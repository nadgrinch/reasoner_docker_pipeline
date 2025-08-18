#!/usr/bin/env python3
import rospy
import numpy as np
import torch
import torchaudio
import soundfile as sf
import io
from collections import deque
from audio_common_msgs.msg import AudioData
from scipy.signal import butter, lfilter
import whisper

class AudioAnalyzer:
  def __init__(self):
    rospy.init_node("microphone_vad_stt_node", anonymous=False)
    self.params = rospy.get_param('microphone_vad_stt')
    
    self.sample_rate = self.params['sample_rate']
    self.chunk_size = self.params['chunk_size']
    self.nchannels = self.params['num_channels']
    self.min_speech = self.params['min_speech_duration_ms']
    self.max_silence = self.params['max_silence_duration']
    self.max_buffer = self.params['max_buffer_duration']
    self.audio_topic = self.params['audio_topic']
    self.pre_filter = self.params['pre_filter']
    self.filter_order = self.params['filter_order']
    self.low_cutoff = self.params['low_cutoff_frequency']
    self.high_cutoff = self.params['high_cutoff_frequency']

    self.pre_speech_buffer = deque(maxlen=int(self.sample_rate * 2))
    self.speech_buffer = []
    self.speech_active = False
    self.silence_counter = 0
    self.start_timestamp = None

    if torch.cuda.is_available():
      # Prefer the 3060 (GPU:1 usually, newer 5060Ti is GPU:0)
      self.device = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'
    else:
      self.device = 'cpu'

    self.vad_model, utils = torch.hub.load(
      repo_or_dir='snakers4/silero-vad',
      model='silero_vad',
      force_reload=False
    )
    self.vad_model = self.vad_model.to('cpu')
    self.get_speech_timestamps = utils[0]
    self.whisper_model = whisper.load_model('small', device=self.device)

    rospy.Subscriber(self.audio_topic, AudioData, self.audio_callback)

  def bandpass_filter(self, data):
    nyquist = 0.5 * self.sample_rate
    low = self.low_cutoff / nyquist
    high = self.high_cutoff / nyquist
    b, a = butter(self.filter_order, [low, high], btype='band')
    return lfilter(b, a, data)

  def audio_callback(self, msg):
    audio_np = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0

    # Optional band-pass filter
    if self.pre_filter:
      audio_np = self.bandpass_filter(audio_np)

    self.pre_speech_buffer.extend(audio_np)

    # Run VAD
    audio_tensor = torch.from_numpy(audio_np).to('cpu')
    speech_timestamps = self.get_speech_timestamps(
      audio_tensor,
      self.vad_model,
      sampling_rate=self.sample_rate,
      min_speech_duration_ms=self.min_speech,
      return_seconds=True
    )

    if speech_timestamps:  # TODO / NOTE timestamps when audio analyzed, want when was recorded
      if not self.speech_active:
        self.speech_active = True
        self.start_timestamp = rospy.Time.now()
        self.speech_buffer.extend(self.pre_speech_buffer)
        rospy.loginfo(f"Speech started at {self.start_timestamp.to_sec():.2f}")
      self.speech_buffer.extend(audio_np)
      self.silence_counter = 0
    else:
      if self.speech_active:
        self.silence_counter += len(audio_np) / self.sample_rate
        if self.silence_counter > self.max_silence:
          self.end_speech()

  def end_speech(self):
    end_timestamp = rospy.Time.now()
    rospy.loginfo(f"Speech ended at {end_timestamp.to_sec():.2f}")

    # Convert to numpy and transcribe
    segment_audio = np.array(self.speech_buffer, dtype=np.float32)
    text = self.transcribe(segment_audio)

    rospy.loginfo(f"Transcription: {text}")

    # Reset
    self.speech_buffer = []
    self.speech_active = False
    self.silence_counter = 0
    self.start_timestamp = None
    self.pre_speech_buffer.clear()

  def transcribe(self, audio_np):
    buf = io.BytesIO()
    sf.write(buf, audio_np, self.sample_rate, format='WAV')
    buf.seek(0)
    result = self.whisper_model.transcribe(buf, fp16=False)
    return result['text']

  def run(self):
    rospy.spin()


if __name__ == '__main__':
  analyzer = AudioAnalyzer()
  analyzer.run()

#!/usr/bin/env python3

import rospy
import wave
import os
import time
import numpy as np
import scipy.signal as signal
from audio_common_msgs.msg import AudioData
from collections import deque

class AnalyzerNode:
  def __init__(self):
    rospy.init_node('microphone_analyzer_node')

    self.params = rospy.get_param('microphone_analyzer')

    self.sample_rate = self.params['sample_rate']
    self.chunk_size = self.params['chunk_size']
    self.fft_size = self.params['fft_size']
    self.threshold = self.params['energy_threshold']
    self.min_speech = self.params['min_speech_duration']
    self.silence = self.params['silence_duration_sec']
    self.buffer_max = self.params['buffer_max_sec']
    self.topic = self.params['audio_topic']
    
    # rospy.loginfo("""[DEBUG] Loaded parameters:\nsample_rate: %d
    #               chunk_size: %d
    #               energy threshold: %d
    #               minimal speech duration: %.2f
    #               silence duration: %.2f
    #               buffer max duration: %.2f
    #               AudioData topic: %s""", 
    #               self.sample_rate, self.chunk_size, self.threshold,
    #               self.min_speech, self.silence, self.buffer_max, self.topic)

    profile_path = os.path.join(
      os.path.dirname(__file__), '../config/noise_profile.npy')
    if not os.path.exists(profile_path):
      rospy.logerr("Noise profile not found at %s", profile_path)
      exit(1)
    self.noise_spectrum = np.load(profile_path)

    self.audio_buffer = deque()
    self.buffer_start_time = None
    self.speech_start_time = None
    self.speech_latest_time = None
    self.speech_detected = False
    self.recording = False

    self.recordings_dir = os.path.join(
      os.path.dirname(__file__), '../recordings')
    os.makedirs(self.recordings_dir, exist_ok=True)

    rospy.Subscriber(self.topic, AudioData, self.audio_callback)
    rospy.loginfo("Microphone Analyzer node started")
    rospy.spin()

  def audio_callback(self, msg):
    audio = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32)
    timestamp = rospy.Time.now().to_sec()
    _, _, Zxx = signal.stft(audio, fs=self.sample_rate, nperseg=self.fft_size)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    mag = np.maximum(mag - self.noise_spectrum[:, np.newaxis], 0.0)
    cleaned = np.clip(signal.istft(
      mag * np.exp(1j * phase), 
      fs=self.sample_rate, nperseg=self.fft_size)[1],
      -32768, 32767).astype(np.int16)
    energy = np.sqrt(np.mean(cleaned.astype(np.float32) ** 2))

    self.audio_buffer.append((timestamp, cleaned))
    if not self.buffer_start_time:
      self.buffer_start_time = timestamp

    while (self.audio_buffer and timestamp - 
            self.audio_buffer[0][0]) > self.buffer_max:
      self.audio_buffer.popleft()
      self.buffer_start_time = self.audio_buffer[0][0] if self.audio_buffer else None

    if energy > self.threshold:
      rospy.loginfo(f"[DEBUG] AudioData received, energy: {energy:.1f}")
      if self.recording:
        return
      if not self.speech_detected:
        self.speech_start_time = timestamp
        self.speech_detected = True
      elif (timestamp - self.speech_start_time) >= self.min_speech:
        rospy.loginfo("Starting recording")
        self.recording = True
    else:
      if self.speech_detected and (timestamp - self.speech_latest_time) > self.silence:
        self.speech_detected = False
      if self.recording and (timestamp - self.speech_latest_time) > self.silence:
        rospy.loginfo(f"Saving recording, low energy levels for longer then {self.silence} s")
        self.save_recording(self.speech_start_time, timestamp)
        self.recording = False

  def save_recording(self, start_time, end_time):
      voice_data = []
      for ts, chunk in self.audio_buffer:
        if start_time <= ts <= end_time:
          voice_data.append(chunk)
      if not voice_data:
        return
      data = np.concatenate(voice_data)
      filename = time.strftime('%Y%m%d_%H%M%S') + '.wav'
      filepath = os.path.join(self.recordings_dir, filename)

      with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(self.params['num_channels'])
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data.tobytes())

      rospy.loginfo("Saved recording: %s", filepath)

if __name__ == "__main__":
    AnalyzerNode()

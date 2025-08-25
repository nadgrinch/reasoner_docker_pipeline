#!/usr/bin/env python3
import os
import io
import wave
import time
import yaml
import rospy
import webrtcvad
import numpy as np
from scipy.signal import resample_poly
from audio_common_msgs.msg import AudioData
from std_msgs.msg import Header
from threading import Lock

def int16_bytes_to_numpy(data_bytes):
  return np.frombuffer(data_bytes, dtype=np.int16)

def numpy_to_int16_bytes(arr):
  return arr.astype(np.int16).tobytes()

def write_wav(path, samples, sample_rate, channels=1):
  samples_i16 = samples.astype(np.int16)
  with wave.open(path, "wb") as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(samples_i16.tobytes())

def resample_to_target(x_int16, src_rate, tgt_rate):
  if src_rate == tgt_rate:
    return x_int16
  # use polyphase resampling
  gcd = np.gcd(src_rate, tgt_rate)
  up = tgt_rate // gcd
  down = src_rate // gcd
  y = resample_poly(x_int16.astype(np.float32), up, down)
  # clip to int16
  y = np.clip(np.round(y), -32768, 32767).astype(np.int16)
  return y

class VADNode:
  def __init__(self):
    rospy.init_node("vad_node", anonymous=False)

    pkg_dir = rospy.get_param("~pkg_dir", os.path.dirname(__file__))
    rospy.loginfo(f"pkg_dir: {pkg_dir}")
    self.cfg = rospy.get_param("vad_stt")

    # config
    self.input_rate = int(self.cfg.get("input_sample_rate", 48000))
    self.vad_rate = int(self.cfg.get("vad_sample_rate", 16000))
    self.frame_ms = int(self.cfg.get("frame_ms", 30))
    self.vad_mode = int(self.cfg.get("vad_mode", 2))
    self.prebuffer_s = float(self.cfg.get("prebuffer_seconds", 1.0))
    self.hang_in_ms = int(self.cfg.get("hang_in_ms", 50))
    self.hang_out_ms = int(self.cfg.get("hang_out_ms", 200))
    self.min_speech_ms = int(self.cfg.get("min_speech_ms", 200))
    self.merge_gap_ms = int(self.cfg.get("merge_gap_ms", 300))
    self.num_channels = int(self.cfg.get("num_channels", 1))
    self.audio_topic = self.cfg.get("audio_topic", "/audio/raw_andrea")
    self.recordings_path = os.path.join(pkg_dir, self.cfg.get("recordings_path", "recordings"))
    os.makedirs(self.recordings_path, exist_ok=True)

    # derived
    self.frame_samples_in = int(self.input_rate * (self.frame_ms / 1000.0))
    # webrtcvad supports frames of 10/20/30 ms at supported rates (we use vad_rate)
    assert self.frame_ms in (10, 20, 30), "frame_ms must be 10,20,30 for webrtcvad"
    self.vad_frame_samples = int(self.vad_rate * (self.frame_ms / 1000.0))

    # buffers and state
    self.prebuffer_max_frames = int(np.ceil(self.prebuffer_s * self.input_rate / self.frame_samples_in))
    self.raw_prebuffer = []  # tuples of (timestamp, int16 numpy)
    self.lock = Lock()

    # vad and state machine
    self.vad = webrtcvad.Vad(self.vad_mode)
    self.state = "IDLE"
    self.speech_frames = []  # list of (timestamp, int16 numpy at input_rate)
    self.last_voice_time = None

    # frame counters for hang logic (in ms)
    self.hang_in_frames = int(np.ceil(self.hang_in_ms / self.frame_ms))
    self.hang_out_frames = int(np.ceil(self.hang_out_ms / self.frame_ms))
    self.min_speech_frames = int(np.ceil(self.min_speech_ms / self.frame_ms))
    self.merge_gap_frames = int(np.ceil(self.merge_gap_ms / self.frame_ms))

    # subscription
    rospy.Subscriber(self.audio_topic, AudioData, self.audio_cb, queue_size=50)
    rospy.loginfo(f"VAD node initialized. Listening to {self.audio_topic}")

  def audio_cb(self, msg: AudioData):
    # msg.data is bytes of int16 PCM (interleaved if multi-channel)
    tstamp = rospy.Time.now()
    arr = int16_bytes_to_numpy(msg.data)
    if self.num_channels > 1:
      # pick first channel for now
      arr = arr.reshape(-1, self.num_channels)[:, 0]
    # push into prebuffer
    with self.lock:
      self.raw_prebuffer.append((tstamp, arr))
      # keep prebuffer limited
      if len(self.raw_prebuffer) > self.prebuffer_max_frames * 10:
        # safety trim
        self.raw_prebuffer = self.raw_prebuffer[-self.prebuffer_max_frames:]

    # process frame for VAD: resample this frame to vad_rate and run webrtcvad
    # note: msg likely contains small chunks; we assume each arrival is one frame or multiple concatenated frames.
    # For generality, process in contiguous blocks of frame_samples_in
    samples = arr
    offset = 0
    while offset + self.frame_samples_in <= len(samples):
      block = samples[offset:offset + self.frame_samples_in]
      offset += self.frame_samples_in
      self.process_frame(block, tstamp)

  def process_frame(self, in_frame_int16, ros_time):
    # resample to vad_rate
    frame_vad = resample_to_target(in_frame_int16, self.input_rate, self.vad_rate)
    # webrtcvad expects raw bytes 16-bit little-endian
    try:
      is_speech = self.vad.is_speech(frame_vad.tobytes(), sample_rate=self.vad_rate)
    except Exception as e:
      rospy.logwarn_throttle(5, f"VAD error: {e}")
      is_speech = False

    # state machine and buffering
    with self.lock:
      if is_speech:
        self.last_voice_time = ros_time

      if self.state == "IDLE":
        if is_speech:
          # enter possible speech
          self.state = "POSSIBLE_SPEECH"
          self.speech_frames = [(ros_time, in_frame_int16)]
          self.possible_count = 1
        # else remain IDLE
      elif self.state == "POSSIBLE_SPEECH":
        if is_speech:
          self.speech_frames.append((ros_time, in_frame_int16))
          self.possible_count += 1
          if self.possible_count >= self.hang_in_frames:
            self.state = "SPEECH"
        else:
          # false alarm, go back to IDLE
          self.state = "IDLE"
          self.speech_frames = []
      elif self.state == "SPEECH":
        if is_speech:
          self.speech_frames.append((ros_time, in_frame_int16))
        else:
          # start possible silence
          if not hasattr(self, "silence_count"):
            self.silence_count = 1
          else:
            self.silence_count += 1
          self.speech_frames.append((ros_time, in_frame_int16))
          if self.silence_count >= self.hang_out_frames:
            # finalize segment
            self.finalize_segment()
            self.state = "IDLE"
            self.speech_frames = []
            self.silence_count = 0
        # continue in SPEECH
      else:
        self.state = "IDLE"

  def finalize_segment(self):
    if not self.speech_frames:
      return
    # assemble numpy audio from speech_frames (they are at input_rate)
    times, frames = zip(*self.speech_frames)
    audio = np.concatenate(frames).astype(np.int16)
    duration_ms = int(len(audio) / self.input_rate * 1000)
    if duration_ms < self.min_speech_ms:
      rospy.loginfo_throttle(10, f"Ignored short segment {duration_ms}ms")
      return

    # compute start and end times using last frame timestamps and pre-roll
    start_time = times[0] - rospy.Duration(self.hang_in_ms / 1000.0)
    end_time = times[-1] + rospy.Duration(self.hang_out_ms / 1000.0)

    # save wav
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    fname = f"speech_{ts}_{int(start_time.to_sec()*1000)}_{int(end_time.to_sec()*1000)}.wav"
    fpath = os.path.join(self.recordings_path, fname)
    try:
      write_wav(fpath, audio, self.input_rate, channels=1)
      rospy.loginfo(f"Saved speech segment {fpath} dur={duration_ms}ms start={start_time.to_sec():.3f} end={end_time.to_sec():.3f}")
    except Exception as e:
      rospy.logerr(f"Failed to save wav: {e}")

    # publish a simple ROS log entry and optionally you could publish a custom message here.
    # TODO: publish SpeechSegment msg if you have one.

  def spin(self):
    rospy.spin()

if __name__ == "__main__":
  node = VADNode()
  node.spin()

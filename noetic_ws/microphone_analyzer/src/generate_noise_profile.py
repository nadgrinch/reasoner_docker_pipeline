#!/usr/bin/env python3

import rospy
import os
import yaml
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "../config/params.yaml")
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    return full_config.get("microphone_analyzer", {})

def main():
    rospy.init_node("generate_noise_profile")
    params = load_config()

    wav_path = rospy.get_param("~input_wav", "")
    if not wav_path or not os.path.exists(wav_path):
        rospy.logerr("You must provide a valid .wav file path via _input_wav param.")
        return

    rate, data = wav.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]

    if rate != params["sample_rate"]:
        rospy.logwarn("Sample rate mismatch: %d (expected %d)", rate, params["sample_rate"])

    data = data.astype(np.float32)
    _, _, Zxx = signal.stft(data, fs=rate, nperseg=params["fft_size"])
    mean_spectrum = np.mean(np.abs(Zxx), axis=1)

    out_path = os.path.join(os.path.dirname(__file__), "../config/noise_profile.npy")
    np.save(out_path, mean_spectrum)
    rospy.loginfo("Noise profile saved to: %s", out_path)

if __name__ == "__main__":
    main()

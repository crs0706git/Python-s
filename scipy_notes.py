import math

import numpy as np
from scipy.io import wavfile
from scipy import signal


def reading_audio(in_path):
    """
    wavfile.read returns as follows
    - The data type and the sampling rate of the audio do not change
    - audio shape: waveform x channel
        - If mono, then channel does not exist
    """
    sr, audio = wavfile.read(in_path)
    return sr, audio


def writing_audio(in_path, in_sr, in_audio):
    """
    - Determines data type that is already saved in the audio-data
    - in_audio shape should be: waveform x channel
    """
    wavfile.write(in_path, in_sr, in_audio)


def resampling_audio(in_audio, in_before_sr, in_after_sr):
    """
    signal.resample should get the final data-length after resampling
    - duration (sec) * desired_sampling_rate
    - duration (sec) = data-length / sampling_rate

    After the resampling, the data type of the output is float64 - should always check
    - Only the data-type of the audio changed, the values inside does not drastically changed
    """
    orig_dtype = in_audio.dtype
    out_audio = signal.resample(in_audio, math.floor(in_audio.shape[0]/in_before_sr*in_after_sr))
    out_audio = np.array(out_audio, dtype=orig_dtype)
    return out_audio


if __name__ == "__main__":
    sample_path= "sample_audio.wav"

    sr, sample_audio = reading_audio(sample_path)
    resampled_audio = resampling_audio(sample_audio, sr, 16000)
    writing_audio("sample_audio_writing.wav", 16000, resampled_audio)

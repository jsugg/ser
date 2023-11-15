from typing import Any

from numpy import ndarray
import numpy as np
import librosa
import soundfile as sf
import time
import warnings

from ser.config import AUDIO_READ_CONFIG


def read_audio_file(file_path: str) -> tuple[np.ndarray | Any, float]:
    """
    Read an audio file.

    Parameters
    ----------
    file_path : str
        Path to the audio file.

    Returns
    -------
    np.ndarray
        Audio data.
    """
    for i in range(AUDIO_READ_CONFIG['max_retries']):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                audiofile, current_sample_rate = librosa.load(file_path, sr=None)
            audiofile: ndarray = np.array(audiofile, dtype=np.float32)

            max_abs_value = np.max(np.abs(audiofile))
            if max_abs_value == 0:
                audiofile = np.zeros_like(audiofile)
            else:
                audiofile /= max_abs_value

            return audiofile, current_sample_rate

        except Exception as e:
            print(f"Error occurred: {e}")
            print(f"Falling back to soundfile...")
            try:
                with sf.SoundFile(file_path) as sound_file:
                    audiofile = sound_file.read(dtype='float32')
                    current_sample_rate = sound_file.samplerate

                max_abs_value = np.max(np.abs(audiofile))
                if max_abs_value == 0:
                    audiofile = np.zeros_like(audiofile)
                else:
                    audiofile /= max_abs_value

                return audiofile, current_sample_rate

            except Exception as err:
                print(f"Error with soundfile: {err}")
                print(f"Retrying with librosa in {AUDIO_READ_CONFIG['retry_delay']} seconds...")
                time.sleep(AUDIO_READ_CONFIG['retry_delay'])
    raise IOError(f"Failed to read audio file {file_path} after {AUDIO_READ_CONFIG['max_retries']} retries.")

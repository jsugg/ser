import warnings
from halo import Halo
import stable_whisper

from ser.config import MODELS_CONFIG

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

def format_transcript(result) -> list:
    """
    Formats the transcript into a list of tuples containing the word, 
    start time, and end time.

    Parameters
    ----------
    result : dict
        The transcript result.

    Returns
    -------
    List[Tuple[str, float, float]]
        Formatted transcript with timestamps.
    """
    text_with_timestamps: list = []
    words = result.all_words()

    for word in words:
        text_with_timestamps.append((word.word, word.start, word.end))

    return text_with_timestamps

def extract_transcript(filename: str, language: str) -> list:
    """
    Extracts the transcript from the audio file.

    Parameters
    ----------
    filename : str
        Path to the audio file.
    language : str
        Language of the audio file.

    Returns
    -------
    List[Tuple[str, float, float]]
        Transcript with word timestamps.
    """
    with Halo(
        text='Loading the speech recognition model...',
        spinner='dots', text_color='green'):
        model_download_path: str = f"{MODELS_CONFIG['models_folder']}/{MODELS_CONFIG['whisper_model']['path']}"
        model = stable_whisper.load_model(
            name=MODELS_CONFIG['whisper_model']['name'],
            device="cpu", dq=False, download_root=model_download_path,
            in_memory=True)

    with Halo(text='Generating the transcript...',
              spinner='dots', text_color='green'):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            transcript: dict = model.transcribe(
                audio=filename, language=language,
                verbose=False, word_timestamps=True,
                demucs=True, vad=True,
                condition_on_previous_text=True)
        formatted_transcript: list = format_transcript(transcript)

    return formatted_transcript

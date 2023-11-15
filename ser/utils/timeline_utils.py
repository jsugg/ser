import csv
from colored import attr, bg, fg
from typing import List, Tuple
from halo import Halo
from ser.config import TIMELINE_CONFIG


def display_elapsed_time(
        elapsed_time: float,
        _format: str = 'long'
    ) -> str:
    """
    Returns the elapsed time in seconds in long or short format.

    Parameters
    ----------
    elapsed_time : Union[int, float]
        Elapsed time in seconds.
    format : str, optional
        Format of the elapsed time ('long' or 'short'), by default 'long'.

    Returns
    -------
    str
        Formatted elapsed time.
    """
    minutes, seconds = divmod(int(elapsed_time), 60)
    if _format == 'long':
        return f"{minutes} min {seconds} seconds" \
            if minutes else f"{elapsed_time} seconds"
    return f"{minutes}m{seconds}s" if minutes else f"{elapsed_time:.2f}s"


def build_timeline(
        text_with_timestamps, emotion_with_timestamps
    ) -> List[Tuple[float, str, str]]:
    """
    Builds a timeline from text and emotion data.

    Parameters
    ----------
    text_with_timestamps : List[Tuple[str, float, float]]
        Transcript data with timestamps.
    emotion_with_timestamps : List[Tuple[str, float, float]]
        Emotion data with timestamps.

    Returns
    -------
    List[Tuple[float, str, str]]
        Combined timeline with timestamps, emotions, and text.
    """
    timeline: List[Tuple[float, str, str]] = []
    all_timestamps: List[float] = sorted(set(
        [t for _, t, _ in text_with_timestamps] + [t for _, t, _ in emotion_with_timestamps]))

    text_dict: dict = {t: text for text, _, t in text_with_timestamps}
    emotion_dict: dict = {t: emotion for emotion, _, t in emotion_with_timestamps}

    for timestamp in all_timestamps:
        text: str = text_dict.get(timestamp, '')
        emotion: str = emotion_dict.get(timestamp, '')
        timeline.append((timestamp, emotion, text))

    return timeline

def color_txt(string: str, fg_color: str, bg_color: str, padding: int = 0) -> str:
    """
    Colorizes a string.

    Parameters
    ----------
    string : str
        String to be colorized.
    fg_color : str
        Foreground color.
    bg_color : str
        Background color.

    Returns
    -------
    str
        Colorized string.
    """
    if padding:
        string = string.ljust(padding)
    
    return f"{fg(fg_color)}{bg(bg_color)}{string}{attr('reset')}"

def print_timeline(timeline: List[Tuple]) -> None:
    """
    Prints the ASCII timeline vertically.

    Parameters
    ----------
    timeline : List[Tuple[Union[int, float], str, str]]
        ASCII timeline.
    """
    # Calculate maximum width for each column
    max_time_width: int = max(len(display_elapsed_time(
        float(ts), _format='short')) for ts, _, _ in timeline)
    max_emotion_width: int = max(len(em.capitalize()) for _, em, _ in timeline)
    max_text_width: int = max(len(txt.strip()) for _, _, txt in timeline)

    # Header
    print(color_txt("Time", "white", "green", max_time_width), end = '')
    print(color_txt("Emotion", "black", "yellow", max_time_width), end = '')
    print(color_txt("Speech", "white", "blue", max_time_width))

    # Print each entry vertically
    for ts, em, txt in timeline:
        time_str: str = f"{display_elapsed_time(float(ts), _format='short')}".ljust(max_time_width)
        emotion_str: str = f"{em.capitalize()}".ljust(max_emotion_width)
        text_str :str = f"{txt.strip()}".ljust(max_text_width)

        print(f"{time_str} {emotion_str} {text_str}")


def save_timeline_to_csv(
        timeline: List[Tuple],
        file_name: str
        ) -> str:
    """
    Saves the timeline to a CSV file.

    Parameters
    ----------
    timeline : List[Tuple[Union[int, float], str, str]]
        The timeline data to be saved.
    file_name : str
        The name of the file to save the timeline to.
    """
    file_name = file_name.split('/')[-1]
    file_name = '.'.join(['/'.join([TIMELINE_CONFIG['folder'], 
                                    file_name.split('.')[0]]), 'csv'])

    with Halo(
        text=f"Saving transcript to {file_name}",
        spinner='dots', text_color='green'):
        with open(file_name, mode='w', newline='',
                  encoding='utf-8') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(['Time (s)', 'Emotion', 'Speech'])

            # Write the data
            for time, emotion, speech in timeline:
                time: float = round(float(time), 2)
                writer.writerow([time, emotion, speech])
    
    return file_name

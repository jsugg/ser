def display_elapsed_time(elapsed_time: float, _format: str = 'long') -> str:
    """
    Returns the elapsed time in seconds in long or short format.

    Arguments:
        elapsed_time (float): Elapsed time in seconds.
        format (str, optional): Format of the elapsed time 
            ('long' or 'short'), by default 'long'.

    Returns:
        str: Formatted elapsed time.
    """
    minutes, seconds = divmod(int(elapsed_time), 60)
    if _format == 'long':
        return f"{minutes} min {seconds} seconds"
    return f"{minutes}m{seconds}s"

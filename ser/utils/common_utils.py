"""General utility helpers shared across CLI output paths."""

def display_elapsed_time(elapsed_time: float, _format: str = "long") -> str:
    """Formats elapsed seconds as either verbose or compact text.

    Args:
        elapsed_time: Elapsed time in seconds.
        _format: Output style, either `"long"` or `"short"`.

    Returns:
        Human-readable elapsed time text.
    """
    minutes, seconds = divmod(int(elapsed_time), 60)
    if _format == "long":
        return f"{minutes} min {seconds} seconds"
    return f"{minutes}m{seconds}s"

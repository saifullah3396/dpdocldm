from pathlib import Path
from typing import Union


def make_dir(path: Union[str, Path]) -> None:
    """
    Creates a directory if it does not exist.

    Args:
        path (Union[str, Path]): The path of the directory.
    """
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)

"""Argument parsing utils."""
from argparse import ArgumentTypeError
from ast import literal_eval
from typing import Any, Tuple, Union


def str2bool(value: Union[str, bool]) -> bool:
    """Convert input string to boolean."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def comma_separated(value: str) -> Tuple[int, ...]:
    """Convert comma-separated values to tuple."""
    converted = [int(v) for v in value.split(",") if v]
    return tuple(converted)


def eq2kwargs(kwargs_str) -> Tuple[str, Any]:
    """Parse a key-value pair separated by '='."""
    key, value = kwargs_str.split("=")
    return key.strip(), literal_eval(value.strip())

from typing import Callable, Any


def _raise_conversion_error(value: str, target_type: str):
    """Raise a ValueError for invalid conversions."""
    raise ValueError(f"Cannot convert {value} to {target_type}.")


class StringConverter:
    """Convert strings to Python types, with optional type enforcement.
    """

    DEFAULT_TYPE_MAP = {
        "bool": lambda v: v == "True" if v in {"True", "False"} else _raise_conversion_error(v, "bool"),
        "int": lambda v: int(v),
        "float": lambda v: float(v),
        "str": lambda v: v
    }

    def __init__(self, type_map: dict = {}):
        self.type_map = type_map if type_map else self.DEFAULT_TYPE_MAP

    def _auto_convert(self, value: str) -> Any:
        """Automatically convert a string to its most appropriate Python type."""
        if value == "True":
            return True
        elif value == "False":
            return False
        try:
            if '.' in value or 'e' in value or 'E' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value

    def string_to_any(self, value: str, enforce_type: str = "auto") -> Any:
        """
        Convert a string to its appropriate Python type, with an optional enforce_type.

        enforce_type can be 'auto', 'str', 'int', 'float', or 'bool'.
        """
        if enforce_type in self.type_map:
            return self.type_map[enforce_type](value)
        elif enforce_type == "auto":
            return self._auto_convert(value)
        else:
            raise ValueError(f"Unknown enforce_type: {enforce_type}")

    def get_type_options(self) -> list[str]:
        """Return a list of valid enforce_type options."""
        return ["auto"] + list(self.type_map.keys())

    def __call__(self, value: str, enforce_type: str = "auto") -> Any:
        """Allow the class instance to be called like a function."""
        return self.string_to_any(value, enforce_type)


def string_to_dtype(input_str: str, enforce_type: str = "auto") -> Callable[[str], Any]:
    """
    Convert a string to a function that converts strings to the appropriate Python type.

    Args:
        input_str (str): The string to convert.
        enforce_type (str): The type to enforce. Can be 'auto', 'str', 'int', 'float', or 'bool'.

    Returns:
        Callable[[str], Any]: The function that converts strings to the appropriate Python type.
    """
    converter = StringConverter()
    return converter(input_str, enforce_type)
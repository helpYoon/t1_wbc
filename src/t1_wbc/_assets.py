"""Package-relative asset paths (self-contained — assets ship inside the package)."""
from importlib.resources import files


def asset(*parts: str) -> str:
    """Absolute path to a vendored asset, e.g. asset('robot', 't1.xml')."""
    return str(files("t1_wbc").joinpath("assets", *parts))

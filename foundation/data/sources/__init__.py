"""Source registry. Each adapter module registers itself on import (Tasks 7+)."""
from .base import (REGISTRY, ManualDownloadRequired, RawBatch, Source,  # noqa: F401
                   fetch, register)
from . import rruff  # noqa: E402,F401

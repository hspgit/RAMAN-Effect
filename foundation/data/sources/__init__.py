"""Source registry. Each adapter module registers itself on import (Tasks 7+)."""
from .base import (REGISTRY, ManualDownloadRequired, RawBatch, Source,  # noqa: F401
                   fetch, register)
from . import rruff  # noqa: E402,F401
from . import api_flanagan  # noqa: E402,F401
from . import airborne  # noqa: E402,F401
from . import zenodo_guo  # noqa: E402,F401

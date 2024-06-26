import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


from .annotate_integrated import annotate_integrated
from .annotate_single import annotate_single
from .build_integrated_references import IntegratedReferences, build_integrated_references
from .build_single_reference import build_single_reference
from .classify_integrated_references import classify_integrated_references
from .classify_single_reference import classify_single_reference
from .get_classic_markers import get_classic_markers, number_of_classic_markers

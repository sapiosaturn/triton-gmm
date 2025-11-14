"""Triton Grouped Matrix Multiplication - Drop-in replacement for torch._grouped_mm."""

from .op import grouped_mm

# Export tgmm as the main API, matching torch._grouped_mm usage
tgmm = grouped_mm

__all__ = ["tgmm", "grouped_mm", "tgmm_autopatch"]


def __getattr__(name: str):
    """Lazy import for tgmm_autopatch that triggers auto-patching."""
    if name == "tgmm_autopatch":
        # Import the autopatch module, which applies the patch on import
        from . import autopatch  # noqa: F401
        
        # Return a sentinel object to indicate patching was applied
        return autopatch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


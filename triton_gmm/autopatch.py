"""Auto-patch module that replaces torch._grouped_mm when imported."""

import torch
from .op import grouped_mm

# Apply the patch automatically when this module is imported
torch._grouped_mm = grouped_mm


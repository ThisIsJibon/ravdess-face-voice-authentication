"""Face + voice verification research pipeline."""
import os as _os

# Notebook hosts (e.g. Google Colab) export MPLBACKEND=module://matplotlib_inline...,
# which the project's separate virtual environment cannot load. Plots are only
# saved to files, so fall back to the non-interactive Agg backend.
if _os.environ.get("MPLBACKEND", "").startswith("module://"):
    _os.environ["MPLBACKEND"] = "Agg"

__version__ = "1.0.0"

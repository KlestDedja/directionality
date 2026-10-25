"""EDGEHOG image directionality analysis.

The original main_code imports remain available for existing scripts.
"""

from main_code.hog_analysis_script import HOGAnalysis
from main_code.pipeline_utils import HOGDescriptor

__all__ = ["HOGAnalysis", "HOGDescriptor"]

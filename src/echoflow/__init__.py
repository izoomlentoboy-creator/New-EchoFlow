"""EchoFlow - voice pathology detection AI.

A hybrid deep-learning pipeline that classifies a voice recording as healthy
or as one of several dysphonia categories, designed from the ground up to be
exported to Core ML and embedded in an iOS application.
"""
from .config import Config, DEFAULT_CONFIG, CLASSES, NUM_CLASSES

__all__ = ["Config", "DEFAULT_CONFIG", "CLASSES", "NUM_CLASSES"]
__version__ = "0.1.0"

"""
VML - Machine Learning for Vejvejr

A package for road temperature prediction using machine learning.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import main classes/functions for easy access
try:
    from .dataset import Dataset
    from .model import *
except ImportError:
    # Handle case where dependencies aren't installed yet
    pass

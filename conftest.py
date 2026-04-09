"""
Root conftest.py — ensures the project root is on sys.path so that
'slicegan' and 'utils' are importable regardless of how pytest is invoked
(e.g. `pytest` from project root, or `pytest tests/` from a subdirectory).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

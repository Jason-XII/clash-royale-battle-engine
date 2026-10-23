"""The tests import the programs at the top level and the cr package: the repository root goes on the path.

    OMP_NUM_THREADS=2 python -m pytest tests
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

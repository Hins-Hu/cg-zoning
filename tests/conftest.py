import os
import sys

# Make the repo root importable so the `cg` and `cliquegen` packages resolve
# regardless of how pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

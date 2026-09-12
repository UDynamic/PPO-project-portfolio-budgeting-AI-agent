# src/conftest.py
# Ensures env/ is on sys.path before any test imports,
# so env.py's bare `import helper` resolves correctly.
import sys
from pathlib import Path

env_dir = Path(__file__).parent / "env"
if str(env_dir) not in sys.path:
    sys.path.insert(0, str(env_dir))
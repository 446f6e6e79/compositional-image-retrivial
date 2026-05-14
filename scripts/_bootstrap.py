"""Shared sys.path bootstrap for the runnable scripts.

Each script does ``import _bootstrap  # noqa`` before importing
``compositional_retrieval`` so the package resolves from the repo root regardless
of where Python is invoked from. This avoids needing a pyproject.toml or pip
install just to run the scripts.
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

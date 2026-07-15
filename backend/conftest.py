"""Root conftest: makes `backend/` importable and guarantees the one env var
Settings requires exists before any test imports app modules. Individual
tests override it (and everything else) through fixtures."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")

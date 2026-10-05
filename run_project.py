"""Convenience launcher for the complete STAAR-IDS research pipeline."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cmd = [sys.executable, str(ROOT / 'run_pipeline.py')] + sys.argv[1:]
raise SystemExit(subprocess.call(cmd, cwd=ROOT))

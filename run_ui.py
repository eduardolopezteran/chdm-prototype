"""
Milestone 3B — top-level Streamlit launch script.

Run with:  streamlit run run_ui.py   (from the chdm-engine/ directory)

This tiny wrapper exists only so ui/app.py's relative imports
(`from . import actions, ...`) resolve correctly -- Streamlit's script
runner executes the target file directly, without package context, so the
actual application logic is imported normally here as `ui.app` instead.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ui.app import main

main()

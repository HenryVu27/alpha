"""Development mode: run all components in a single process."""
import sys

sys.argv = ["main.py", "--config", "config.yaml"]

from main import main

main()

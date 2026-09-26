"""Entry point the graders run: `python tdma_schedule.py ...` (created in Division 1).

All the work happens in the tdma_brain package; this file only calls tdma_brain.cli.main.
"""
import sys

from tdma_brain.cli import main

if __name__ == "__main__":
    sys.exit(main())

"""Entry point for running the Wayfinder agent via the SDK.

Usage:
    uv run main.py --agent=wayfinder --game=ls20
"""

from agents.wayfinder.cli import main

if __name__ == "__main__":
    main()

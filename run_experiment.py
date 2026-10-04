"""Compatibility wrapper for the noninteractive ``facets`` CLI."""

from facets.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

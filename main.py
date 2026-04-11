"""Compatibility wrapper for the server entry point."""

from server.app import app, main, start_server


if __name__ == "__main__":
    main()
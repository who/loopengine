"""Command-line interface for LoopEngine."""

import argparse
import sys

import uvicorn


def main(args: list[str] | None = None) -> int:
    """Run the LoopEngine server.

    Args:
        args: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        prog="loopengine",
        description="LoopEngine - Universal Agent Simulation Framework",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    parsed = parser.parse_args(args)

    print(f"Starting LoopEngine server at http://{parsed.host}:{parsed.port}")
    print("Press Ctrl+C to stop")

    uvicorn.run(
        "loopengine.server.app:app",
        host=parsed.host,
        port=parsed.port,
        reload=parsed.reload,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

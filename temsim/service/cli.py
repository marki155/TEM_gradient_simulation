r"""Part IV, Section 6: the command line.

``python -m temsim.service serve`` starts a core; ``python -m temsim.service ping``
asks a running core for its description.  The core binds to ``127.0.0.1`` unless
told otherwise (see the note in :mod:`temsim.service.protocol`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List, Optional

from .protocol import endpoint_argument

DEFAULT_ENDPOINT = "tcp://127.0.0.1:5555"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="temsim.service", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run a core")
    serve.add_argument("--endpoint", default=DEFAULT_ENDPOINT, type=endpoint_argument)
    serve.add_argument("--device", default="auto", help="auto, cpu or cuda")
    serve.add_argument("--open", metavar="PATH", help="open a run directory or data folder at start")
    serve.add_argument("--workspace", metavar="DIR",
                       help="directory for opening and saving: clients pick data and runs from it, relative paths "
                            "refer to it, 'save as' writes into it (created if missing)")
    serve.add_argument("--fresh", action="store_true", help="with --open: ignore an existing autosave")
    serve.add_argument("--frame-interval", type=float, default=1.5)
    serve.add_argument("--frame-steps", type=int, default=None,
                       help="while training, send the images of every N-th step (default 10; 0: only paced frames)")
    serve.add_argument("--save-interval", type=float, default=120.0)
    ping = sub.add_parser("ping", help="describe a running core")
    ping.add_argument("--endpoint", default=DEFAULT_ENDPOINT, type=endpoint_argument)
    args = parser.parse_args(argv)

    if args.command == "ping":
        from .client import Client
        with Client(args.endpoint, timeout=5.0) as client:
            print(json.dumps(client.info(), indent=2, default=str))
        return 0

    from .jobs import Clock
    from .server import Server
    options = {} if args.frame_steps is None else {"frame_steps": args.frame_steps}
    options["workspace"] = args.workspace
    server = Server(args.endpoint, args.device, Clock(args.frame_interval, save_interval=args.save_interval), **options)
    endpoint = server.start()
    print(f"temsim core on {endpoint} (device {server.session.device})", flush=True)
    if args.open:
        info = server.session.open_dataset(args.open, resume=not args.fresh)
        print(f"opened {args.open}: {info['kind']}, {sum(l['atoms'] for l in info['layers'])} atoms", flush=True)
        for note in info.get("notes", []):
            print(f"  {note}", flush=True)
    try:
        while server._thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

import sys

if len(sys.argv) > 1 and sys.argv[1] == "run":
    from .runner import main
    sys.exit(main(sys.argv[2:]))
from .window import launch_window
sys.exit(launch_window(sys.argv[1:]))

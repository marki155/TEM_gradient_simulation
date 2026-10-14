"""``python -m temsim serve|ping`` is the same as ``python -m temsim.service serve|ping``."""

import sys

from .service.cli import main

sys.exit(main())

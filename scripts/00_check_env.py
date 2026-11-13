"""Environment gate: `uv run python scripts/00_check_env.py [--cpu-ok] [--json] [--no-import]`.

Thin wrapper; the logic lives in ``etfcorr.envcheck`` so it is typed and unit-tested.
Exit codes: 0 ok (or CPU fallback accepted with --cpu-ok), 1 core environment broken,
2 GPU unusable.
"""

import sys

from etfcorr.envcheck import main

if __name__ == "__main__":
    sys.exit(main())

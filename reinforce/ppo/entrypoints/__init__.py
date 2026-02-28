from __future__ import annotations

import os

# Enable colored warning/error logs by default for all `python -m reinforce.ppo.entrypoints.*`.
# Users can still disable colors with NO_COLOR or override FORCE_COLOR explicitly.
os.environ.setdefault("FORCE_COLOR", "1")

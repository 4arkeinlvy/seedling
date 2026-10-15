"""seedling: train models on pooled, unreliable hardware by sharing only seeds and scores.

Keep this file free of heavy imports (torch): multi-process tests spawn many small
processes that only need numpy and the socket code.
"""

__version__ = "0.0.1"

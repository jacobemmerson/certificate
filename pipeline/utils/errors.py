"""Safe exception rendering for logs and persisted evaluation records."""

from __future__ import annotations


def safe_exception(error: BaseException, context: str = "operation failed") -> str:
    """Return exception type plus a generic message, never provider payload text."""
    return f"{type(error).__name__}: {context}; sensitive details omitted"

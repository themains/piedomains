"""Outcome taxonomy and run reporting for batch classification.

Every result row carries a machine-readable ``status``, ``stage`` and
``error_code`` so that failures across a large URL list can be grouped without
string-matching on human messages. :func:`build_report` turns a list of rows
into an aggregate report naming exactly which domains produced nothing.

Example:
    >>> rows = [
    ...     {"domain": "a.com", "category": "news", "confidence": 0.9},
    ...     {"domain": "b.com", "category": None, "error": "boom"},
    ... ]
    >>> rows = [annotate(r) for r in rows]
    >>> report = build_report(rows, run_id="demo")
    >>> report["classified"], report["failed"], report["missing"]
    (1, 1, ['b.com'])
"""

from __future__ import annotations

import socket
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

__all__ = [
    "ErrorCode",
    "Stage",
    "Status",
    "annotate",
    "build_report",
    "classify_exception",
    "failure",
]


class Status(StrEnum):
    """Terminal outcome for a single domain."""

    OK = "ok"
    FAILED = "failed"


class Stage(StrEnum):
    """Pipeline stage a domain reached."""

    VALIDATE = "validate"
    FETCH = "fetch"
    PROCESS = "process"
    INFER = "infer"


class ErrorCode(StrEnum):
    """Stable, groupable reason a domain produced no classification.

    These strings are part of the public output contract: they are safe to
    aggregate on and will not change meaning between releases.
    """

    INVALID_DOMAIN = "invalid_domain"
    DNS_ERROR = "dns_error"
    CONNECTION_ERROR = "connection_error"
    TIMEOUT = "timeout"
    HTTP_ERROR = "http_error"
    ROBOTS_BLOCKED = "robots_blocked"
    CONTENT_TYPE_REJECTED = "content_type_rejected"
    CONTENT_TOO_LARGE = "content_too_large"
    NO_ARCHIVE_SNAPSHOT = "no_archive_snapshot"
    ARCHIVE_RATE_LIMITED = "archive_rate_limited"
    EMPTY_TEXT = "empty_text"
    MISSING_INPUT_PATH = "missing_input_path"
    MISSING_SCREENSHOT = "missing_screenshot"
    MODEL_LOAD_ERROR = "model_load_error"
    MODEL_ERROR = "model_error"
    LLM_ERROR = "llm_error"
    #: An anti-bot challenge or interstitial was served instead of the page.
    BOT_BLOCKED = "bot_blocked"
    #: Fetched and parsed, but too little text to classify honestly.
    THIN_CONTENT = "thin_content"
    #: Umbrella terminal state: we refuse to label this domain. Callers that do
    #: not want to enumerate every cause can branch on this one code.
    CANNOT_CLASSIFY = "cannot_classify"
    UNKNOWN = "unknown"


#: Codes worth retrying — transient network/rate-limit conditions.
RETRYABLE: frozenset[ErrorCode] = frozenset(
    {
        ErrorCode.DNS_ERROR,
        ErrorCode.CONNECTION_ERROR,
        ErrorCode.TIMEOUT,
        ErrorCode.HTTP_ERROR,
        ErrorCode.ARCHIVE_RATE_LIMITED,
        # Worth another attempt from a different IP, or via the archive.
        ErrorCode.BOT_BLOCKED,
    }
)


def classify_exception(exc: BaseException) -> ErrorCode:
    """Map an exception to a stable :class:`ErrorCode`.

    Matching is done on exception type where possible and on message text only
    as a fallback, so that swapping HTTP or browser libraries does not silently
    reclassify failures.

    Args:
        exc: The exception raised while handling a domain.

    Returns:
        ErrorCode: The best-matching code, or ``UNKNOWN``.
    """
    if isinstance(exc, TimeoutError):
        return ErrorCode.TIMEOUT
    if isinstance(exc, socket.gaierror):
        return ErrorCode.DNS_ERROR
    if isinstance(exc, ConnectionError):
        return ErrorCode.CONNECTION_ERROR
    if isinstance(exc, FileNotFoundError):
        return ErrorCode.MISSING_INPUT_PATH

    name = type(exc).__name__.lower()
    text = f"{name} {exc}".lower()

    if "timeout" in text:
        return ErrorCode.TIMEOUT
    if "name or service not known" in text or "nodename nor servname" in text:
        return ErrorCode.DNS_ERROR
    if "robots" in text:
        return ErrorCode.ROBOTS_BLOCKED
    if "429" in text or "rate limit" in text:
        return ErrorCode.ARCHIVE_RATE_LIMITED
    if "no snapshot" in text or "no archive" in text or "no memento" in text:
        return ErrorCode.NO_ARCHIVE_SNAPSHOT
    if "connection" in text:
        return ErrorCode.CONNECTION_ERROR
    if "status" in text and "http" in text:
        return ErrorCode.HTTP_ERROR
    return ErrorCode.UNKNOWN


def failure(
    code: ErrorCode,
    stage: Stage,
    message: str,
    **extra: Any,
) -> dict[str, Any]:
    """Build the annotation fields describing a failed domain.

    Args:
        code: The stable reason the domain failed.
        stage: The pipeline stage it failed in.
        message: Human-readable detail, kept alongside the code.
        **extra: Additional fields merged into the result.

    Returns:
        dict[str, Any]: Fields to merge into the result row.
    """
    return {
        "status": Status.FAILED.value,
        "stage": stage.value,
        "error_code": code.value,
        "error": message,
        "retryable": code in RETRYABLE,
        **extra,
    }


def annotate(row: dict[str, Any], *, stage: Stage = Stage.INFER) -> dict[str, Any]:
    """Ensure a result row carries the outcome fields, inferring them if absent.

    Rows produced before annotation was threaded through a code path still get
    a coherent ``status``/``stage``/``error_code`` so the report is never
    partially populated.

    Args:
        row: A result row, mutated in place and returned.
        stage: Stage to attribute an un-annotated failure to.

    Returns:
        dict[str, Any]: The same row, with outcome fields filled in.
    """
    if row.get("status") is None:
        succeeded = row.get("category") is not None and not row.get("error")
        row["status"] = Status.OK.value if succeeded else Status.FAILED.value

    if row["status"] == Status.OK.value:
        row.setdefault("stage", Stage.INFER.value)
        row.setdefault("error_code", None)
        row.setdefault("retryable", False)
        return row

    row.setdefault("stage", stage.value)
    if row.get("error_code") is None:
        row["error_code"] = ErrorCode.UNKNOWN.value
    try:
        row.setdefault("retryable", ErrorCode(row["error_code"]) in RETRYABLE)
    except ValueError:
        row.setdefault("retryable", False)
    return row


def build_report(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> dict[str, Any]:
    """Aggregate annotated rows into a run report.

    Args:
        rows: Result rows, ideally already passed through :func:`annotate`.
        run_id: Identifier correlating the report with the log stream.
        started_at: When the run began.
        finished_at: When the run ended; defaults to now.

    Returns:
        dict[str, Any]: Counts by reason and stage, plus the domains that
        produced no classification.
    """
    finished = finished_at or datetime.now(UTC)
    by_reason: dict[str, int] = {}
    by_stage: dict[str, int] = {}
    missing: list[str] = []

    for row in rows:
        annotate(row)
        if row["status"] == Status.OK.value:
            continue
        code = row.get("error_code") or ErrorCode.UNKNOWN.value
        by_reason[code] = by_reason.get(code, 0) + 1
        stage = row.get("stage") or Stage.INFER.value
        by_stage[stage] = by_stage.get(stage, 0) + 1
        missing.append(row.get("domain") or row.get("url") or "")

    failed = len(missing)
    report: dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at.isoformat() if started_at else None,
        "finished_at": finished.isoformat(),
        "elapsed_ms": (
            int((finished - started_at).total_seconds() * 1000) if started_at else None
        ),
        "total": len(rows),
        "classified": len(rows) - failed,
        "failed": failed,
        "by_reason": dict(sorted(by_reason.items(), key=lambda kv: -kv[1])),
        "by_stage": dict(sorted(by_stage.items(), key=lambda kv: -kv[1])),
        "missing": missing,
    }
    return report

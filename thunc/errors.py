class ThuncError(RuntimeError):
    """Any failure to get a usable answer from the model."""


class TransientError(ThuncError):
    """A failure that asking again may fix: a timeout, a lost connection, a rate limit, a server
    error, or a CLI call that ended in an error. An agent run retries the step; thunc.call doesn't."""


# HTTP statuses worth asking again for: timeout, conflict, rate limit, and server errors (529 included).
TRANSIENT_STATUS = frozenset({408, 409, 429})


def transient_status(status: int | None) -> bool:
    return status is not None and (status in TRANSIENT_STATUS or status >= 500)


# The Claude API's error types worth asking again for, as its error bodies name them.
TRANSIENT_CLAUDE_ERRORS = frozenset({"overloaded_error", "api_error", "rate_limit_error", "timeout_error"})


def transient_claude_error(status: int | None, body: object) -> bool:
    """Whether a Claude API error is worth asking again for. An error sent while a reply streams
    (`event: error`) comes with the stream's HTTP status, 200, so its body's type decides."""
    if transient_status(status):
        return True
    error = body.get("error") if isinstance(body, dict) else None
    return isinstance(error, dict) and error.get("type") in TRANSIENT_CLAUDE_ERRORS

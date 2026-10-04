class ThuncError(RuntimeError):
    """Any failure to get a usable answer from the model."""


class TransientError(ThuncError):
    """A failure that asking again may fix: a timeout, a lost connection, a rate limit, a server
    error, or a CLI call that ended in an error. An agent run retries the step; thunc.call doesn't."""


# HTTP statuses worth asking again for: timeout, conflict, rate limit, and server errors (529 included).
TRANSIENT_STATUS = frozenset({408, 409, 429})


def transient_status(status: int | None) -> bool:
    return status is not None and (status in TRANSIENT_STATUS or status >= 500)

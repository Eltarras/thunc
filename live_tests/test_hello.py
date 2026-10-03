import thunc


@thunc.function
def hello():
    """Hello"""
    ...


def test_hello_returns_text():
    reply = hello()
    assert isinstance(reply, str)
    assert reply.strip()

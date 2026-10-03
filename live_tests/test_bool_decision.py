import thunc


@thunc.function
def bool_decision() -> bool:
    """is 1 =3?"""
    ...


@thunc.function
def is_true(statement: str) -> bool:
    """Is this statement true?"""
    ...


def test_false_statement():
    assert bool_decision() is False


def test_true_statement():
    # Without a true case, a model that always answers False would pass.
    assert is_true("2 + 2 = 4") is True

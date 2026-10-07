"""thunc/source.py: putting a written body into a file, and compiling it from there."""

import textwrap

import pytest

from thunc.source import SourceError, find, first_line, load, splice

BODY = "return len(text)\n"


def code(text):
    return textwrap.dedent(text).lstrip("\n")


def test_splice_writes_the_body_and_removes_the_decorator():
    text = code('''
        """Demo."""

        import os

        import thunc


        @thunc.function(write=True)
        def size(text: str) -> int:
            """Count the characters."""
            ...


        print(size("abc"))
    ''')
    new, start = splice(
        text, "size", 8, "return len(re.sub(r'\\s', '', text))\n", ["import re", "import os"], note="Written."
    )
    assert new == code('''
        """Demo."""

        import os

        import thunc
        import re


        def size(text: str) -> int:
            """Count the characters."""
            # Written.
            return len(re.sub(r'\\s', '', text))


        print(size("abc"))
    ''')
    assert new.splitlines()[start - 1] == "def size(text: str) -> int:"


def test_splice_leaves_the_thunc_import_for_the_rest_of_the_file():
    text = code('''
        import thunc


        @thunc.function
        def other(x: str) -> str:
            """Other."""


        @thunc.function(write=True)
        def size(text: str) -> int:
            """Count."""
    ''')
    new, _ = splice(text, "size", 9, BODY)
    assert new.startswith("import thunc\n")
    assert "@thunc.function\ndef other" in new
    assert "@thunc.function(write=True)" not in new
    assert new.endswith('    """Count."""\n    return len(text)\n')


def test_splice_handles_methods_stacked_and_multiline_decorators():
    text = code('''
        import thunc


        class Box:
            @staticmethod
            @thunc.function(
                write=True,
                retries=1,
            )  # it writes itself
            def size(text: str) -> int:
                """Count."""
                ...  # the model's
    ''')
    new, start = splice(text, "Box.size", 6, BODY)
    assert new == code('''
        import thunc


        class Box:
            @staticmethod
            def size(text: str) -> int:
                """Count."""
                return len(text)
    ''')
    assert new.splitlines()[start - 1].strip() == "@staticmethod"


def test_splice_adds_doctest_examples_escaping_backslashes():
    text = code('''
        @thunc.function(write=True)
        def size(text: str) -> int:
            """Count the characters."""
            ...
    ''')
    new, _ = splice(text, "size", 1, BODY, examples=[("size('a\\\\b')", "3"), ("size('x')", "1")])
    assert new == code('''
        def size(text: str) -> int:
            """Count the characters.

            >>> size('a\\\\\\\\b')
            3
            >>> size('x')
            1
            """
            return len(text)
    ''')


def test_splice_adds_examples_to_a_multiline_docstring_and_keeps_raw_ones_raw():
    text = code('''
        @thunc.function(write=True)
        def size(text: str) -> int:
            r"""Count the characters.

            Like \\w does.
            """
            ...
    ''')
    new, _ = splice(text, "size", 1, BODY, examples=[("size('a\\\\b')", "3")])
    assert 'Like \\w does.\n\n    >>> size(\'a\\\\b\')\n    3\n    """\n    return len(text)\n' in new


def test_splice_keeps_crlf_and_tabs():
    lines = ["import thunc", "", "class A:", "\t@thunc.function(write=True)", "\tdef f(self, x: str) -> int:"]
    text = "\r\n".join([*lines, '\t\t"""Doc."""', "\t\t...", ""])
    new, _ = splice(text, "A.f", 4, "if x:\n    return 1\nreturn 0\n")
    body = ["\t\tif x:", "\t\t    return 1", "\t\treturn 0", ""]
    head = ["import thunc", "", "class A:", "\tdef f(self, x: str) -> int:", '\t\t"""Doc."""']
    assert new == "\r\n".join([*head, *body])


def test_splice_puts_imports_after_the_module_docstring_when_there_are_none():
    text = '#!/usr/bin/env python3\n"""Doc."""\n\n@thunc.function(write=True)\ndef f(x: str) -> str:\n    """D."""\n'
    new, start = splice(text, "f", 4, "return x\n", ["import re"])
    assert new.startswith('#!/usr/bin/env python3\n"""Doc."""\nimport re\n\ndef f')
    assert start == 5


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('@thunc.function(write=True)\n@other\ndef f(x: str) -> str:\n    """D."""\n', "nearest the def"),
        ('@thunc.function\ndef f(x: str) -> str:\n    """D."""\n', "can't find @thunc.function"),
        ("@thunc.function(write=True)\ndef f(x: str) -> str:\n    ...\n", "needs a docstring"),
        ('@thunc.function(write=True)\ndef f(x: str) -> str: """D."""\n', "lines of their own"),
        ('@thunc.function(write=True)\ndef f(x: str) -> str:\n    """D."""; ...\n', "lines of their own"),
    ],
)
def test_splice_refuses_what_it_cant_edit_cleanly(text, message):
    with pytest.raises(SourceError, match=message):
        splice(text, "f", 1, "return x\n")


def test_splice_refuses_a_body_that_indenting_would_change():
    text = '@thunc.function(write=True)\ndef f(x: str) -> str:\n    """D."""\n'
    with pytest.raises(SourceError, match="multi-line string"):
        splice(text, "f", 1, 'return """a\nb"""\n')


def test_find_picks_between_definitions_by_line():
    text = code('''
        if X:
            @thunc.function(write=True)
            def f(x: str) -> str:
                """One."""
        else:
            @thunc.function(write=True)
            def f(x: str) -> str:
                """Two."""
    ''')
    import ast

    tree = ast.parse(text)
    assert first_line(find(tree, "f", 6)) == 6
    with pytest.raises(SourceError, match="single definition"):
        find(tree, "f")


def test_load_compiles_against_the_namespace_with_the_files_line_numbers():
    text = code('''
        from __future__ import annotations

        import re


        def words(text: str) -> Later:
            """Count words."""
            return len(re.findall(r"\\w+", text)) // 0
    ''')
    namespace = {"__name__": "demo", "re": __import__("re")}
    func = load(text, "demo.py", "words", 6, namespace)
    assert func.__annotations__["return"] == "Later"  # the file's __future__ import applies
    assert func.__globals__ is namespace
    with pytest.raises(ZeroDivisionError) as info:
        func("a b")
    assert info.traceback[-1].lineno + 1 == 8


def test_load_compiles_methods_with_name_mangling_and_super():
    class Base:
        def greet(self):
            return "base"

    class Box(Base):
        __secret = 7

    text = code("""
        class Box(Base):
            __secret = 7

            def greet(self):
                return super().greet() + str(self.__secret)
    """)
    func = load(text, "demo.py", "Box.greet", 4, {"Base": Base}, owner=Box)
    Box.greet = func
    assert Box().greet() == "base7"
    assert func.__qualname__ == "Box.greet"

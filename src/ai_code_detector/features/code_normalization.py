"""Reversible-ish code-string normalization passes for input-format ablations.

The goal of these helpers is to remove a *specific* signal from the raw code
string so the ablation experiment can isolate how much the model relies on
that signal. Each transform is intentionally small and dumb: we use string-
and regex-level operations rather than a real Python parser so the same
helpers work even on syntactically-broken AI samples.

Available transforms:

- ``strip_comments``: remove hash line comments and triple-quoted docstrings.
- ``collapse_whitespace``: replace runs of horizontal whitespace with a
  single space and runs of blank lines with one.
- ``strip_blank_lines``: drop lines that are entirely whitespace.

These are deliberately not destructive of meaning at the token level (they
remove style, not semantics), so post-transform code should still be
classifiable.
"""

from __future__ import annotations

import re

# Match either a triple-quoted string (greedy across lines) OR a hash line
# comment. The regex is anchored to the start-of-token because we only want
# to strip comments that are not inside a regular string literal -- this is
# a heuristic, not a parser, but works on well-formed snippets.
_TRIPLE_QUOTE_OR_COMMENT_RE = re.compile(
    r"""
    (?P<triple>           # triple-quoted strings (the docstring convention)
        '{3}.*?'{3}
        |
        \"{3}.*?\"{3}
    )
    |
    (?P<comment>          # line comments (ignore inside string literals)
        \#[^\n]*
    )
    """,
    re.DOTALL | re.VERBOSE,
)

_HORIZONTAL_WS_RE = re.compile(r"[ \t]+")
_MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")
_BLANK_LINE_RE = re.compile(r"^\s*$\n?", re.MULTILINE)


def strip_comments(code: str) -> str:
    """Remove hash line comments and triple-quoted docstrings."""
    return _TRIPLE_QUOTE_OR_COMMENT_RE.sub("", code)


def collapse_whitespace(code: str) -> str:
    """Replace runs of horizontal whitespace with a single space."""
    code = _HORIZONTAL_WS_RE.sub(" ", code)
    code = _MULTI_BLANK_LINE_RE.sub("\n\n", code)
    return code.strip("\n")


def strip_blank_lines(code: str) -> str:
    """Remove lines that are entirely whitespace."""
    return _BLANK_LINE_RE.sub("", code)


# Named transforms exposed via the ablation script. The "raw" identity is
# included so the ablation always has a baseline row.
TRANSFORMS = {
    "raw": lambda s: s,
    "no_comments": strip_comments,
    "no_comments_no_blank_lines": lambda s: strip_blank_lines(strip_comments(s)),
    "minified": lambda s: collapse_whitespace(strip_blank_lines(strip_comments(s))),
}

from __future__ import annotations

import re


_INVISIBLE = dict.fromkeys(map(ord, ["\u200b", "\ufeff"]), None)
_MULTISPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    s = s.translate(_INVISIBLE)
    s = s.strip()
    s = _MULTISPACE.sub(" ", s)
    return s


"""Turn an Advisor AI summary into a speakable script. Rule-based: figures are reworded, never changed."""
from __future__ import annotations

import re

DISCLAIMER = "These results are projections, not guarantees, and not investment advice."
MAX_SCRIPT_CHARS = 6_000

_SYMBOLS = {"&": " and ", "→": " to ", "≈": " about ", "±": " plus or minus ", "×": " times ", "—": ", ", "–": " to "}


def _strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)                     # code blocks
    text = re.sub(r"<tool_call>.*?</tool_call>", " ", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)                       # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)                    # links -> label
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", line):  # table separator
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
            line = ", ".join(cells)
        line = re.sub(r"^#{1,6}\s*", "", line)                               # headings
        line = re.sub(r"^(\s*[-*+]|\s*\d+[.)])\s+", "", line)               # bullets
        line = re.sub(r"^>\s*", "", line)
        if line and not re.search(r"[.!?:;,]$", line):
            line += "."
        if line:
            lines.append(line)
    text = " ".join(lines)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"(\*\*|__|\*|_)(\S.*?\S|\S)\1", r"\2", text)
    return text


def _speak_numbers(text: str) -> str:
    def money(m: re.Match) -> str:
        sign, amount, suffix = m.group(1) or "", m.group(2), (m.group(3) or "").upper()
        word = {"K": " thousand", "M": " million", "B": " billion"}.get(suffix, "")
        prefix = "minus " if sign in ("-", "−") else ""
        return f"{prefix}{amount}{word} dollars"

    text = re.sub(r"([-−])?\$\s?(\d[\d,]*(?:\.\d+)?)\s?([KkMmBb])?\b", money, text)
    text = re.sub(r"\+(\d+(?:\.\d+)?)\s?pp\b", r"up \1 percentage points", text)
    text = re.sub(r"[-−](\d+(?:\.\d+)?)\s?pp\b", r"down \1 percentage points", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s?pp\b", r"\1 percentage points", text)
    text = re.sub(r"\+(\d+(?:\.\d+)?)\s?%", r"up \1 percent", text)
    text = re.sub(r"(?<![\w.])[-−](\d+(?:\.\d+)?)\s?%", r"down \1 percent", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s?%", r"\1 percent", text)
    text = re.sub(r"\bP(10|25|50|75|90)\b", r"\1th percentile", text)
    return text


def speakable(text: str) -> str:
    """Reword symbols and figures for speech (e.g. "$1,200" -> "1,200 dollars"); values are unchanged."""
    for symbol, word in _SYMBOLS.items():
        text = text.replace(symbol, word)
    return re.sub(r"\s+", " ", _speak_numbers(text)).strip()


def build_script(summary: str, client_name: str | None = None, advisor: str | None = None) -> str:
    """Opening + cleaned summary + fixed disclaimer. The disclaimer is always present."""
    body = _strip_markdown(summary or "")
    body = body.replace("⚠️", "").replace("✅", "")
    body = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", body)       # emoji
    for symbol, word in _SYMBOLS.items():
        body = body.replace(symbol, word)
    body = _speak_numbers(body)
    body = body.replace(DISCLAIMER, "")
    body = re.sub(r"\s+", " ", body).strip(" .") + "." if body.strip(" .") else ""
    body = re.sub(r"\s+([.,;:!?])", r"\1", body)
    body = re.sub(r"\.{2,}", ".", body)
    body = body.replace("*", "").replace("_", " ")
    if not re.search(r"[A-Za-z0-9]", body):
        raise ValueError("There is no summary text to turn into audio.")
    opening = f"Retirement plan update for {client_name}." if client_name else "Retirement plan update."
    closing = f"Prepared by your advisor, {advisor}." if advisor else ""
    budget = MAX_SCRIPT_CHARS - len(opening) - len(closing) - len(DISCLAIMER) - 3
    if len(body) > budget:
        cut = body[:budget]
        body = cut[: cut.rfind(".") + 1] if "." in cut else cut
    return " ".join(p for p in (opening, body, closing, DISCLAIMER) if p)

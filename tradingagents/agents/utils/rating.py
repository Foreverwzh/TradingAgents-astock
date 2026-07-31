"""Shared 5-tier rating vocabulary and a deterministic heuristic parser.

The same five-tier scale (Buy, Overweight, Hold, Underweight, Sell) is used by:
- The Research Manager (investment plan recommendation)
- The Portfolio Manager (final position decision)
- The signal processor (rating extracted for downstream consumers)
- The memory log (rating tag stored alongside each decision entry)

Centralising it here avoids drift between those call sites.

Why this parser is defensive
----------------------------
The PM is *supposed* to emit a structured ``**Rating**: X`` header via typed
output.  In practice that guarantee breaks (observed 2026-07: the header
disappeared for ~9 consecutive runs and the PM fell back to free-form prose,
apparently because the LLM relay in use does not reliably support structured
output).  When it breaks the decision is still stated plainly in the text --
just in a shape the old parser could not see:

    ### 最终交易决策：持有 (Hold)      <- value in parens; old strip() kept "(hold)"
    ### **FINAL RATING: 减持**        <- English label, *Chinese* value

Both used to fall through to ``default`` and be recorded as "Hold", which is
far worse than returning nothing: a real "reduce" verdict became a neutral one,
so the field silently mixed genuine Holds with parse failures.  Hence:

* Chinese labels *and* Chinese rating words are recognised.
* The word-scan fallback strips brackets/quotes, not just ``*:.,``.
* ``default`` is now ``None`` -- a failed parse must be distinguishable from a
  real Hold, because ``rating_score`` feeds downstream factor research and a
  systematic bias toward neutral is worse than a missing value.
"""

from __future__ import annotations

import re

# Canonical, ordered 5-tier scale (most bullish to most bearish).
RATINGS_5_TIER: tuple[str, ...] = (
    "Buy", "Overweight", "Hold", "Underweight", "Sell",
)

_RATING_SET = {r.lower() for r in RATINGS_5_TIER}

# Chinese rating vocabulary -> canonical 5-tier label.  The PM writes these when
# it drops out of structured output and answers in the configured output
# language (TRADINGAGENTS_OUTPUT_LANGUAGE=Chinese).
# NOTE: only words that *are* a rating belong here.  A verb that commonly
# precedes one does not -- see the 维持 discussion in `_first_rating`.
_CN_RATING: dict[str, str] = {
    "买入": "Buy", "强烈买入": "Buy", "加仓": "Buy",
    "增持": "Overweight", "超配": "Overweight", "适度增持": "Overweight",
    "持有": "Hold", "观望": "Hold", "中性": "Hold", "标配": "Hold",
    "减持": "Underweight", "低配": "Underweight", "减仓": "Underweight",
    "卖出": "Sell", "清仓": "Sell", "离场": "Sell",
}

# Label line, e.g. "**Rating**: Underweight" / "FINAL RATING: 减持" /
# "最终交易决策：维持 `Underweight`（减持）评级".  A separator is *required*
# (keeps prose like "研究经理的卖出决策" from matching); ``[\s*]*`` tolerates
# markdown bold.  Everything after the separator is captured and then scanned
# for an actual rating -- see `parse_rating` for why we do not just grab the
# next token.
_RATING_LABEL_RE = re.compile(
    r"(?:最终交易决策|投资建议|投资评级|rating|评级|决策|建议)"
    r"[\s*]*[:：\-](.*)",
    re.IGNORECASE,
)

# Punctuation to peel off a bare word before matching.  The original only
# stripped ``*:.,`` which meant "(Hold)" was never recognised.
_STRIP_CHARS = "*:.,()（）【】[]「」『』<>《》、；;：!！?？\"'`"


def _first_rating(s: str) -> str | None:
    """Return the earliest *recognised* rating word in ``s``, or None.

    "Earliest recognised" rather than "next token" matters.  Real output looks
    like::

        **最终交易决策：维持 `Underweight`（减持）评级**

    Grabbing the token straight after the separator yields 维持 ("maintain"),
    a verb, not a rating -- and "维持 X 评级" is one of the most common phrasings
    in Chinese analyst writing, so mapping 维持 to Hold silently flattened every
    such verdict to neutral regardless of X.  That is the same directional bias
    this module was rewritten to eliminate, reintroduced through a different
    door.  Scanning for a word that is actually in the vocabulary skips the verb
    and finds Underweight.

    Matching is index-based (not token-based) so backticks, full-width parens
    and markdown around the value do not hide it -- ``` `Underweight` ``` and
    ``Underweight（减持）`` both used to fall through.
    """
    best: tuple[int, str] | None = None

    for word in RATINGS_5_TIER:
        m = re.search(rf"\b{word}\b", s, re.IGNORECASE)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), word)

    for cn, rating in _CN_RATING.items():
        i = s.find(cn)
        # Longer Chinese terms win ties naturally: "强烈买入" starts two chars
        # before the "买入" inside it, so it compares as earlier.
        if i >= 0 and (best is None or i < best[0]):
            best = (i, rating)

    return best[1] if best else None


def parse_rating(text: str, default: str | None = None) -> str | None:
    """Heuristically extract a 5-tier rating from prose text.

    Three passes, most explicit first:

    1. An explicit label line ("Rating: X" / "最终交易决策：X"), where X is
       found by scanning the rest of the line for a known rating word.
    2. The earliest known rating word anywhere in the text.
    3. ``default`` (``None`` by default -- see module docstring).

    Returns a Title-cased canonical rating, or ``default`` if nothing matched.
    """
    if not text:
        return default

    # 1. explicit label; the value may sit behind a verb, backticks or brackets
    for line in text.splitlines():
        m = _RATING_LABEL_RE.search(line)
        if not m:
            continue
        found = _first_rating(m.group(1))
        if found:
            return found
        # A label whose remainder holds no rating ("最终交易决策：明确、可执行的
        # 操作指令") is a section heading, not the verdict -- keep looking.

    # 2. earliest rating word anywhere
    return _first_rating(text) or default

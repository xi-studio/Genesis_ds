"""
Heuristic token estimate in line with DeepSeek-V3-style BPE (no extra deps).

Splits **chars per token** for CJK-style codepoints vs Latin / symbols / whitespace,
which tracks ``tokenizer.json`` far better than one global ``len / 2.5``. Defaults
were tuned on short mixed EN/ZH/code strings against the official vocab; API
``usage.prompt_tokens`` still refines both ratios via :func:`update_ratio_from_usage`.
"""
from __future__ import annotations

# Mutable; tuned vs deepseek_v3 tokenizer.json on mixed EN/ZH/code snippets (mean rel
# error ~28% on a small panel; pathological repeats still diverge without real BPE).
_CJK_CHARS_PER_TOKEN: float = 1.9
_OTHER_CHARS_PER_TOKEN: float = 4.0

_CJK_PT_MIN, _CJK_PT_MAX = 1.15, 2.35
_OTHER_PT_MIN, _OTHER_PT_MAX = 3.2, 6.0

_CALIB_ALPHA = 0.35


def _is_cjk_style(ch: str) -> bool:
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF  # CJK Unified
        or 0x3400 <= o <= 0x4DBF  # Extension A
        or 0xF900 <= o <= 0xFAFF  # Compatibility ideographs
        or 0x3040 <= o <= 0x30FF  # Hiragana / Katakana
        or 0xAC00 <= o <= 0xD7AF  # Hangul syllables
        or 0x3000 <= o <= 0x303F  # CJK symbols and punctuation
        or 0xFF00 <= o <= 0xFFEF  # Fullwidth forms
    )


def _count_cjk_other(text: str) -> tuple[int, int]:
    cjk = sum(1 for ch in text if _is_cjk_style(ch))
    return cjk, len(text) - cjk


def _raw_estimate(text: str) -> float:
    if not text:
        return 0.0
    cjk, other = _count_cjk_other(text)
    return cjk / _CJK_CHARS_PER_TOKEN + other / _OTHER_CHARS_PER_TOKEN


def update_ratio_from_usage(prompt_text: str, prompt_tokens: int) -> None:
    """Scale both ratios when the API reports prompt token usage.

    ``prompt_text`` must be the same string used for local length estimates (e.g.
    ``json.dumps(messages)`` for infer calibration).
    """
    global _CJK_CHARS_PER_TOKEN, _OTHER_CHARS_PER_TOKEN
    if not prompt_text or prompt_tokens <= 0:
        return
    est = _raw_estimate(prompt_text)
    if est <= 0.25:
        return
    scale = est / float(prompt_tokens)
    adj = 1.0 + _CALIB_ALPHA * (scale - 1.0)
    adj = max(0.82, min(1.22, adj))
    ncjk = _CJK_CHARS_PER_TOKEN * adj
    nother = _OTHER_CHARS_PER_TOKEN * adj
    _CJK_CHARS_PER_TOKEN = max(_CJK_PT_MIN, min(_CJK_PT_MAX, ncjk))
    _OTHER_CHARS_PER_TOKEN = max(_OTHER_PT_MIN, min(_OTHER_PT_MAX, nother))


def count_tokens(text: str) -> int:
    """Approximate token count for ``text``."""
    if not text:
        return 0
    return max(1, round(_raw_estimate(text)))

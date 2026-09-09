"""
Token counting — exact via the DeepSeek tokenizer (HF ``tokenizers`` Rust lib),
with a heuristic fallback when the tokenizer file is unavailable.

**All models are approximated with a single universal tokenizer**
``agent/tokenizer_data/deepseek.json`` (deepseek-ai/DeepSeek-V3). Rationale:
window trimming only cares about *aggregate* counts, and a side-by-side
measurement on real mixed log/code/Chinese corpora showed the GLM tokenizer
differs by only ~0.4% in aggregate (up to ±18% per message, which washes out
once summed). Keeping one dictionary also drops the 19MB ``glm.json`` and stays
conservative for code-heavy content (DeepSeek over-counts code vs GLM).

``count_tokens`` caches the loaded instance and falls back to the CJK/Latin
heuristic when the file is missing or loading fails.

Heuristic (fallback only): CJK-style codepoints vs Latin/symbols/whitespace with
separate chars-per-token ratios; ``update_ratio_from_usage`` still refines the
fallback ratios from API ``usage.prompt_tokens`` so even the no-tokenizer path
improves over time.
"""
from __future__ import annotations

import os
import threading
from typing import Any

# ---------------------------------------------------------------------------
# Heuristic fallback (used only when no tokenizer file is available)
# ---------------------------------------------------------------------------

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
    """Refine heuristic fallback ratios from API prompt-token usage.

    No-op effect on the exact-tokenizer path (which doesn't use these ratios),
    but keeps the fallback path self-correcting. Safe to call regardless of which
    counting path is active.
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


# ---------------------------------------------------------------------------
# Exact tokenizer selection (HF ``tokenizers`` Rust lib)
# ---------------------------------------------------------------------------

_TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer_data")

# All models are approximated with the single DeepSeek tokenizer (module docstring
# explains why). ``Config.tokenizer_reference_model`` no longer affects selection.
_UNIVERSAL_TOKENIZER_FILE = "deepseek.json"

_lock = threading.Lock()
_tok_cache: dict[str, Any] = {}      # filename → Tokenizer | None
_tokenizers_import_failed = False    # remember if the lib itself is unavailable


def _load_tokenizer(filename: str) -> Any | None:
    """Lazy-load and cache a Tokenizer by filename; None if unavailable."""
    global _tokenizers_import_failed
    if filename in _tok_cache:
        return _tok_cache[filename]
    with _lock:
        if filename in _tok_cache:
            return _tok_cache[filename]
        tok = None
        if not _tokenizers_import_failed:
            path = os.path.join(_TOKENIZER_DIR, filename)
            if os.path.isfile(path):
                try:
                    from tokenizers import Tokenizer
                    tok = Tokenizer.from_file(path)
                except ImportError:
                    _tokenizers_import_failed = True
                    tok = None
                except Exception:
                    tok = None
        _tok_cache[filename] = tok
        return tok


def _tokenizer_for_current_model() -> Any | None:
    """The universal tokenizer (deepseek.json) approximating every model."""
    return _load_tokenizer(_UNIVERSAL_TOKENIZER_FILE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Exact token count for ``text`` via the current model's tokenizer.

    Falls back to the CJK/Latin heuristic when no tokenizer file matches the
    configured model or the ``tokenizers`` library is unavailable.
    """
    if not text:
        return 0
    tok = _tokenizer_for_current_model()
    if tok is not None:
        try:
            return max(1, len(tok.encode(text).ids))
        except Exception:
            pass  # fall through to heuristic
    return max(1, round(_raw_estimate(text)))


def active_counter() -> str:
    """Diagnostic: which counting path is active."""
    if _load_tokenizer(_UNIVERSAL_TOKENIZER_FILE) is not None:
        return f"exact:{_UNIVERSAL_TOKENIZER_FILE}"
    return "heuristic"

"""
Content search (grep) — behavior aligned with nanobot GrepTool semantics.

Read-only walk under a file or directory; skips binary, huge files, and noise dirs.
"""
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

T = TypeVar("T")

_DEFAULT_HEAD_LIMIT = 250
_MAX_RESULT_CHARS = 128_000
_MAX_FILE_BYTES = 2_000_000

_IGNORE_DIRS = frozenset({
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "dist",
    "build",
})

_TYPE_GLOB_MAP: dict[str, tuple[str, ...]] = {
    "py": ("*.py", "*.pyi"),
    "python": ("*.py", "*.pyi"),
    "js": ("*.js", "*.jsx", "*.mjs", "*.cjs"),
    "ts": ("*.ts", "*.tsx", "*.mts", "*.cts"),
    "json": ("*.json",),
    "md": ("*.md", "*.mdx"),
    "markdown": ("*.md", "*.mdx"),
    "go": ("*.go",),
    "rs": ("*.rs",),
    "rust": ("*.rs",),
    "java": ("*.java",),
    "sh": ("*.sh", "*.bash"),
    "yaml": ("*.yaml", "*.yml"),
    "yml": ("*.yaml", "*.yml"),
    "toml": ("*.toml",),
    "sql": ("*.sql",),
    "html": ("*.html", "*.htm"),
    "css": ("*.css", "*.scss", "*.sass"),
}


def _normalize_pattern(pattern: str) -> str:
    return pattern.strip().replace("\\", "/")


def _match_glob(rel_path: str, name: str, pattern: str) -> bool:
    normalized = _normalize_pattern(pattern)
    if not normalized:
        return False
    if "/" in normalized or normalized.startswith("**"):
        return PurePosixPath(rel_path).match(normalized)
    return fnmatch.fnmatch(name, normalized)


def _is_binary(raw: bytes) -> bool:
    if b"\x00" in raw:
        return True
    sample = raw[:4096]
    if not sample:
        return False
    non_text = sum(byte < 9 or 13 < byte < 32 for byte in sample)
    return (non_text / len(sample)) > 0.2


def _paginate(items: list[T], limit: int | None, offset: int) -> tuple[list[T], bool]:
    if limit is None:
        return items[offset:], False
    sliced = items[offset : offset + limit]
    truncated = len(items) > offset + limit
    return sliced, truncated


def _pagination_note(limit: int | None, offset: int, truncated: bool) -> str | None:
    if truncated:
        if limit is None:
            return f"(pagination: offset={offset})"
        return f"(pagination: limit={limit}, offset={offset})"
    if offset > 0:
        return f"(pagination: offset={offset})"
    return None


def _matches_type(name: str, file_type: str | None) -> bool:
    if not file_type:
        return True
    lowered = file_type.strip().lower()
    if not lowered:
        return True
    patterns = _TYPE_GLOB_MAP.get(lowered, (f"*.{lowered}",))
    return any(fnmatch.fnmatch(name.lower(), pattern.lower()) for pattern in patterns)


def _display_path(file_path: Path, root: Path) -> str:
    try:
        return file_path.relative_to(root).as_posix()
    except ValueError:
        return file_path.as_posix()


def _iter_files(target: Path) -> Any:
    if target.is_file():
        yield target
        return
    root = target
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _IGNORE_DIRS)
        current = Path(dirpath)
        for filename in sorted(filenames):
            yield current / filename


def _format_block(
    display_path: str,
    lines: list[str],
    match_line: int,
    before: int,
    after: int,
) -> str:
    start = max(1, match_line - before)
    end = min(len(lines), match_line + after)
    block = [f"{display_path}:{match_line}"]
    for line_no in range(start, end + 1):
        marker = ">" if line_no == match_line else " "
        block.append(f"{marker} {line_no}| {lines[line_no - 1]}")
    return "\n".join(block)


def run_grep(args: dict[str, Any]) -> str:
    """Execute grep tool; paths are resolved from process cwd (same as ``read_file``)."""
    pattern = str(args.get("pattern") or "").strip()
    if not pattern:
        return "Error: pattern is required"

    path_arg = str(args.get("path") or ".").strip() or "."
    glob_pat = args.get("glob")
    glob_pat = str(glob_pat).strip() if glob_pat else None
    file_type = args.get("type")
    file_type = str(file_type).strip() if file_type else None

    case_insensitive = bool(args.get("case_insensitive", False))
    fixed_strings = bool(args.get("fixed_strings", False))
    output_mode = str(args.get("output_mode") or "files_with_matches").strip()
    if output_mode not in {"content", "files_with_matches", "count"}:
        output_mode = "files_with_matches"

    context_before = min(20, max(0, int(args.get("context_before") or 0)))
    context_after = min(20, max(0, int(args.get("context_after") or 0)))
    offset = min(100_000, max(0, int(args.get("offset") or 0)))

    head_limit = args.get("head_limit")
    max_matches = args.get("max_matches")
    max_results = args.get("max_results")
    if head_limit is not None:
        limit = None if int(head_limit) == 0 else int(head_limit)
    elif output_mode == "content" and max_matches is not None:
        limit = int(max_matches)
    elif output_mode != "content" and max_results is not None:
        limit = int(max_results)
    else:
        limit = _DEFAULT_HEAD_LIMIT

    try:
        target = Path(path_arg).expanduser().resolve()
    except OSError as e:
        return f"Error: {e}"

    if not target.exists():
        return f"Error: Path not found: {path_arg}"
    if not (target.is_dir() or target.is_file()):
        return f"Error: Unsupported path: {path_arg}"

    flags = re.IGNORECASE if case_insensitive else 0
    try:
        needle = re.escape(pattern) if fixed_strings else pattern
        regex = re.compile(needle, flags)
    except re.error as e:
        return f"Error: invalid regex pattern: {e}"

    blocks: list[str] = []
    result_chars = 0
    seen_content_matches = 0
    truncated = False
    size_truncated = False
    skipped_binary = 0
    skipped_large = 0
    matching_files: list[str] = []
    counts: dict[str, int] = {}
    file_mtimes: dict[str, float] = {}

    root = target if target.is_dir() else target.parent

    for file_path in _iter_files(target):
        rel_path = file_path.relative_to(root).as_posix()
        if glob_pat and not _match_glob(rel_path, file_path.name, glob_pat):
            continue
        if not _matches_type(file_path.name, file_type):
            continue

        try:
            raw = file_path.read_bytes()
        except OSError:
            skipped_binary += 1
            continue
        if len(raw) > _MAX_FILE_BYTES:
            skipped_large += 1
            continue
        if _is_binary(raw):
            skipped_binary += 1
            continue
        try:
            mtime = file_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped_binary += 1
            continue

        lines = content.splitlines()
        display_path = _display_path(file_path, root)
        file_had_match = False
        for idx, line in enumerate(lines, start=1):
            if not regex.search(line):
                continue
            file_had_match = True

            if output_mode == "count":
                counts[display_path] = counts.get(display_path, 0) + 1
                continue
            if output_mode == "files_with_matches":
                if display_path not in matching_files:
                    matching_files.append(display_path)
                    file_mtimes[display_path] = mtime
                break

            seen_content_matches += 1
            if seen_content_matches <= offset:
                continue
            if limit is not None and len(blocks) >= limit:
                truncated = True
                break
            block = _format_block(
                display_path,
                lines,
                idx,
                context_before,
                context_after,
            )
            extra_sep = 2 if blocks else 0
            if result_chars + extra_sep + len(block) > _MAX_RESULT_CHARS:
                size_truncated = True
                break
            blocks.append(block)
            result_chars += extra_sep + len(block)
        if output_mode == "count" and file_had_match:
            if display_path not in matching_files:
                matching_files.append(display_path)
                file_mtimes[display_path] = mtime
        if output_mode in {"count", "files_with_matches"} and file_had_match:
            continue
        if truncated or size_truncated:
            break

    if output_mode == "files_with_matches":
        if not matching_files:
            result = f"No matches found for pattern '{pattern}' in {path_arg}"
        else:
            ordered_files = sorted(
                matching_files,
                key=lambda name: (-file_mtimes.get(name, 0.0), name),
            )
            paged, truncated = _paginate(ordered_files, limit, offset)
            result = "\n".join(paged)
    elif output_mode == "count":
        if not counts:
            result = f"No matches found for pattern '{pattern}' in {path_arg}"
        else:
            ordered_files = sorted(
                matching_files,
                key=lambda name: (-file_mtimes.get(name, 0.0), name),
            )
            ordered, truncated = _paginate(ordered_files, limit, offset)
            result = "\n".join(f"{name}: {counts[name]}" for name in ordered)
    else:
        if not blocks:
            result = f"No matches found for pattern '{pattern}' in {path_arg}"
        else:
            result = "\n\n".join(blocks)

    notes: list[str] = []
    if output_mode == "content" and truncated:
        notes.append(f"(pagination: limit={limit}, offset={offset})")
    elif output_mode == "content" and size_truncated:
        notes.append("(output truncated due to size)")
    elif truncated and output_mode in {"count", "files_with_matches"}:
        note = _pagination_note(limit, offset, truncated)
        if note:
            notes.append(note)
    elif output_mode in {"count", "files_with_matches"} and offset > 0:
        notes.append(f"(pagination: offset={offset})")
    elif output_mode == "content" and offset > 0 and blocks:
        notes.append(f"(pagination: offset={offset})")
    if skipped_binary:
        notes.append(f"(skipped {skipped_binary} binary/unreadable files)")
    if skipped_large:
        notes.append(f"(skipped {skipped_large} large files)")
    if output_mode == "count" and counts:
        notes.append(f"(total matches: {sum(counts.values())} in {len(counts)} files)")
    if notes:
        result += "\n\n" + "\n".join(notes)
    return result

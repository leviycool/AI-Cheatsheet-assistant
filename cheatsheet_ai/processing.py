"""Cleaning and chunking helpers for messy study materials."""

from __future__ import annotations

import re
from collections import Counter


def clean_extracted_text(text: str) -> str:
    """Remove obvious extraction noise while keeping headings, bullets, and formulas."""
    if not text:
        return ""

    prepared_lines = [_normalize_line(line) for line in text.splitlines()]
    repeated_counter = Counter()

    for line in prepared_lines:
        plain = _plain_line(line)
        if _is_repeated_noise_candidate(plain):
            repeated_counter[plain] += 1

    repeated_noise = {line for line, count in repeated_counter.items() if count >= 3}

    cleaned: list[str] = []
    previous_plain = ""

    for raw_line in prepared_lines:
        if not raw_line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue

        plain = _plain_line(raw_line)
        if _is_obvious_noise(plain):
            continue
        if plain in repeated_noise and not _looks_contentful(raw_line):
            continue
        if plain == previous_plain:
            continue

        if cleaned and cleaned[-1] != "" and _should_join_lines(cleaned[-1], raw_line):
            cleaned[-1] = _repair_spacing(f"{cleaned[-1]} {raw_line}")
            previous_plain = _plain_line(cleaned[-1])
            continue

        cleaned.append(_repair_spacing(raw_line))
        previous_plain = plain

    compacted: list[str] = []
    for line in cleaned:
        if line == "" and compacted and compacted[-1] == "":
            continue
        compacted.append(line)

    return "\n".join(compacted).strip()


def chunk_text(text: str, max_chars: int = 4500, overlap_chars: int = 300) -> list[str]:
    """Split long text into paragraph-friendly chunks with light overlap."""
    if not text.strip():
        return []

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            for piece in _split_large_paragraph(paragraph, max_chars):
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk).strip())
                    current_chunk = []
                    current_length = 0
                chunks.append(piece.strip())
            continue

        projected = current_length + len(paragraph) + 2
        if current_chunk and projected > max_chars:
            chunks.append("\n\n".join(current_chunk).strip())
            current_chunk = _build_overlap(current_chunk, overlap_chars)
            current_length = sum(len(item) + 2 for item in current_chunk)

        current_chunk.append(paragraph)
        current_length += len(paragraph) + 2

    if current_chunk:
        chunks.append("\n\n".join(current_chunk).strip())

    return [chunk for chunk in chunks if chunk]


def _normalize_line(line: str) -> str:
    line = line.replace("\xa0", " ")
    line = re.sub(r"\s+", " ", line).strip()
    return line


def _plain_line(line: str) -> str:
    plain = re.sub(r"^#+\s*", "", line)
    plain = re.sub(r"^[-*]\s*", "", plain)
    plain = re.sub(r"^\d+\.\s*", "", plain)
    plain = plain.strip().lower()
    return re.sub(r"\s+", " ", plain)


def _is_obvious_noise(plain: str) -> bool:
    if not plain:
        return False

    noise_patterns = [
        r"^\d+$",
        r"^page\s+\d+(\s+of\s+\d+)?$",
        r"^slide\s+\d+(\s+of\s+\d+)?$",
        r"^\[\d+\]$",
        r"^\d+/\d+$",
        r"^copyright\b.*$",
        r"^all rights reserved$",
        r"^confidential$",
        r"^www\.[^\s]+$",
        r"^https?://[^\s]+$",
    ]
    return any(re.match(pattern, plain) for pattern in noise_patterns)


def _is_repeated_noise_candidate(plain: str) -> bool:
    if not plain or len(plain) > 80:
        return False
    if len(plain.split()) > 12:
        return False
    if any(symbol in plain for symbol in ("=", "<", ">", "+", "-", "/", "*")):
        return False
    return True


def _looks_contentful(line: str) -> bool:
    lowered = line.lower()
    content_keywords = (
        "definition",
        "theorem",
        "formula",
        "equation",
        "algorithm",
        "example",
        "proof",
        "assumption",
        "model",
        "framework",
    )
    if any(keyword in lowered for keyword in content_keywords):
        return True
    if re.search(r"[=<>%]", line):
        return True
    if ":" in line and len(line.split(":")[0].split()) <= 8:
        return True
    return False


def _repair_spacing(line: str) -> str:
    line = re.sub(r"\s+([,:;])", r"\1", line)
    return line


def _should_join_lines(previous_line: str, current_line: str) -> bool:
    previous = previous_line.strip()
    current = current_line.strip()
    if not previous or not current:
        return False

    if previous.startswith(("#", "-", "*", "|")) and current.startswith(("#", "-", "*", "|")):
        return False
    if current.startswith(("#", "-", "*", "|", ">", ".")):
        return False
    if re.match(r"^\d+\.\s+", current):
        return False
    if previous.endswith((".", "?", "!", ":", ";")):
        return False
    if re.search(r"[=<>|]", previous) or re.search(r"[=<>|]", current):
        return False
    if previous.lower().startswith(("input dataset", "output dataset", "summary statistics", "variable |")):
        return False
    if re.match(r"^[a-z(\[]", current):
        return True
    if previous.endswith((",", "--")):
        return True
    return False


def _split_large_paragraph(paragraph: str, max_chars: int) -> list[str]:
    lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
    pieces: list[str] = []
    current: list[str] = []
    current_length = 0

    for line in lines or [paragraph]:
        if current and current_length + len(line) + 1 > max_chars:
            pieces.append("\n".join(current))
            current = [line]
            current_length = len(line)
        else:
            current.append(line)
            current_length += len(line) + 1

    if current:
        pieces.append("\n".join(current))

    return pieces


def _build_overlap(paragraphs: list[str], overlap_chars: int) -> list[str]:
    overlap: list[str] = []
    total = 0

    for paragraph in reversed(paragraphs):
        projected = total + len(paragraph) + 2
        if overlap and projected > overlap_chars:
            break
        overlap.insert(0, paragraph)
        total = projected

    return overlap

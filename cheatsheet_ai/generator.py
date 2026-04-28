"""Chunk summarization and final cheat sheet generation helpers."""

from __future__ import annotations

import os
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - safe fallback when dependency is absent
    OpenAI = None  # type: ignore[assignment]

try:
    import streamlit as st
except Exception:  # pragma: no cover - safe fallback when dependency is absent
    st = None  # type: ignore[assignment]


DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"

CHEATSHEET_SECTION_ORDER = [
    "1. Core Concepts",
    "2. Key Measures / Formulas",
    "3. Must-Know Distinctions",
    "4. Classic Examples / Findings",
    "5. Exam Traps",
]


@dataclass
class GenerationOptions:
    course_name: str
    output_language: str
    target_length: str
    focus_style: str
    include_examples: bool
    include_formulas: bool
    include_exam_questions: bool
    density: str
    variant: int = 0


@dataclass
class UsageStats:
    api_calls: int = 0
    usage_available_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    def add(self, other: "UsageStats") -> None:
        self.api_calls += other.api_calls
        self.usage_available_calls += other.usage_available_calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.reasoning_tokens += other.reasoning_tokens

    @classmethod
    def from_response(cls, response) -> "UsageStats":
        usage = extract_usage(response)
        return cls(
            api_calls=1,
            usage_available_calls=1 if usage["available"] else 0,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            cached_input_tokens=usage["cached_input_tokens"],
            reasoning_tokens=usage["reasoning_tokens"],
        )


def extract_usage(response: Any) -> dict[str, int | bool]:
    """Extract token usage from either Responses API or Chat Completions metadata."""
    usage = _read_response_field(response, "usage")
    if usage is None:
        return {
            "available": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
        }

    input_tokens = _read_response_field(usage, "input_tokens")
    output_tokens = _read_response_field(usage, "output_tokens")
    total_tokens = _read_response_field(usage, "total_tokens")

    if input_tokens is None and output_tokens is None:
        input_tokens = _read_response_field(usage, "prompt_tokens")
        output_tokens = _read_response_field(usage, "completion_tokens")

    input_details = _read_response_field(usage, "input_tokens_details")
    output_details = _read_response_field(usage, "output_tokens_details")

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "available": any(value is not None for value in (input_tokens, output_tokens, total_tokens)),
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(total_tokens or 0),
        "cached_input_tokens": int(_read_response_field(input_details, "cached_tokens") or 0),
        "reasoning_tokens": int(_read_response_field(output_details, "reasoning_tokens") or 0),
    }


def is_openai_configured() -> bool:
    """Return True when the OpenAI SDK and API key are both available."""
    return OpenAI is not None and bool(get_openai_api_key())


def get_openai_api_key() -> str:
    """Read the OpenAI API key from env vars or Streamlit secrets."""
    return _get_runtime_config("OPENAI_API_KEY")


def get_openai_model() -> str:
    """Read the preferred model from env vars or Streamlit secrets."""
    return _get_runtime_config("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL


def summarize_chunks(chunks: list[str], options: GenerationOptions) -> tuple[list[str], UsageStats]:
    """Summarize each chunk before the final aggregation step."""
    usage_totals = UsageStats()
    if not chunks:
        return [], usage_totals

    summaries: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        if is_openai_configured():
            try:
                summary, chunk_usage = _summarize_chunk_with_openai(chunk, options, index, len(chunks))
                summaries.append(summary)
                usage_totals.add(chunk_usage)
                continue
            except Exception:
                pass

        summaries.append(_heuristic_chunk_summary(chunk, options, index))

    return summaries, usage_totals


def generate_cheatsheet(
    chunk_summaries: list[str],
    options: GenerationOptions,
    source_text: str | None = None,
) -> tuple[str, UsageStats]:
    """Combine chunk summaries into one condensed, exam-oriented cheat sheet."""
    if is_openai_configured():
        try:
            return _generate_cheatsheet_with_openai(chunk_summaries, options)
        except Exception:
            pass

    combined_source = "\n\n".join(chunk_summaries) if chunk_summaries else (source_text or "")
    return _generate_cheatsheet_heuristic(combined_source, options, source_text or combined_source), UsageStats()


def audit_cheatsheet(
    cheatsheet_markdown: str,
    chunk_summaries: list[str],
    options: GenerationOptions,
) -> tuple[str, UsageStats]:
    """Audit and revise the cheatsheet so every bullet is accurate and exam-useful."""
    if is_openai_configured():
        try:
            return _audit_cheatsheet_with_openai(cheatsheet_markdown, chunk_summaries, options)
        except Exception:
            pass

    return _audit_cheatsheet_heuristic(cheatsheet_markdown), UsageStats()


def _summarize_chunk_with_openai(
    chunk: str,
    options: GenerationOptions,
    chunk_index: int,
    chunk_total: int,
) -> tuple[str, UsageStats]:
    system_prompt = (
        "You are extracting candidate concepts from lecture slides for a one-page exam cheatsheet. "
        "Your highest priority is factual accuracy. Use only information explicitly supported by the uploaded slides in this chunk. "
        "Do not invent, generalize, repair missing meaning, or add outside knowledge. "
        "If a fragment is uncertain, broken, duplicated, decorative, or incomplete OCR, leave it out."
    )
    user_prompt = f"""
Slide chunk {chunk_index} of {chunk_total}

Course/topic: {options.course_name or "Infer from material"}
Output language: {options.output_language}
Include formulas: {options.include_formulas}
Include possible exam questions: {options.include_exam_questions}
Include examples/findings: {options.include_examples}

Extract candidate concepts, not candidate sentences.

A candidate concept can be:
- term or construct
- measure or formula
- model or method
- important distinction
- example, finding, or dataset
- exam trap or interpretation rule

Extraction rules:
- Do not include slide headings unless they teach a concept.
- Do not include half-sentences or broken OCR.
- Preserve exact technical terms.
- Preserve formulas and numerical values exactly when present.
- Compress wording without changing meaning.
- Omit anything uncertain.

Output format:
- Use short markdown bullets only.
- One bullet per concept.
- Each bullet should include as many of these as the chunk supports: concept name, definition, interpretation, distinction, example, or exam trap.
- Do not output random copied fragments.

Source chunk:
{chunk}
""".strip()

    return _call_openai(system_prompt, user_prompt, max_output_tokens=1400)


def _generate_cheatsheet_with_openai(
    chunk_summaries: list[str], options: GenerationOptions
) -> tuple[str, UsageStats]:
    system_prompt = (
        "You are creating a one-page exam cheatsheet from lecture slides. "
        "You are not summarizing slides. You are teaching the important concepts in compact form. "
        "Your highest priority is factual accuracy. Use only information explicitly supported by the extracted slide notes. "
        "Do not invent, generalize, or add outside knowledge. If support is uncertain, leave it out. "
        "Every bullet must be complete, non-duplicated, understandable without the slides, and useful for exam review."
    )
    word_budget = _target_word_budget(options.target_length, options.density)
    language_hint = _language_instruction(options.output_language)

    user_prompt = f"""
Create one polished, one-page A4 exam cheatsheet in markdown.

Course/topic: {options.course_name or "Infer from the summaries"}
Output language: {options.output_language}
Target length: {options.target_length}
Approximate word budget: {word_budget}
Include examples: {options.include_examples}
Include formulas: {options.include_formulas}
Include possible exam questions: {options.include_exam_questions}
Density preference: {options.density}

Core requirement:
- A good cheatsheet explains concepts clearly and fits them on one A4 page.
- It should not simply list slide headings or copied fragments.
- Prioritize concepts over slide order.
- Merge repeated content.
- If the draft is too long, keep the highest-value concepts and compress wording.
- Do not delete definitions before deleting examples.

For each concept, prefer this teaching format:
- Concept name: one clear sentence explaining what it means.
- Why it matters / how to identify it: one short sentence.
- Exam trap: one short sentence only if useful.

Remove all bullets that are:
- incomplete sentences
- slide agenda items without explanation
- generic headings
- OCR artifacts
- questions without answers
- duplicated ideas

Final quality check for every bullet:
- Does this explain a concept?
- Would a student understand it without the slides?
- Is it useful for a quiz/exam?
- Is it short enough for A4?
If no, revise or delete it.

Output format:
- Start with: `# [Course / Lecture Title] - A4 Cheatsheet`
- Use the exact section headings below, in this order, when supported:
{chr(10).join(f"## {heading}" for heading in CHEATSHEET_SECTION_ORDER)}
- Under `## 1. Core Concepts`, explain the key concepts clearly.
- Under `## 2. Key Measures / Formulas`, include formula if available and say what high/low values mean when the slides support that interpretation.
- Under `## 3. Must-Know Distinctions`, prefer a compact markdown table with columns `Concept A | Concept B | Difference`.
- Under `## 4. Classic Examples / Findings`, include short examples only when they help clarify the concept.
- Under `## 5. Exam Traps`, write likely multiple-choice logic, common misinterpretations, or cautions only when directly supported.
- Do not output long paragraphs.
- Do not output generic section labels or raw extraction fragments.
- {language_hint}

Extracted slide notes:
{chr(10).join(chunk_summaries)}
""".strip()

    return _call_openai(system_prompt, user_prompt, max_output_tokens=2600)


def _call_openai(system_prompt: str, user_prompt: str, max_output_tokens: int) -> tuple[str, UsageStats]:
    client = OpenAI(api_key=get_openai_api_key())
    response = client.responses.create(
        model=get_openai_model(),
        max_output_tokens=max_output_tokens,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ],
    )
    usage_stats = UsageStats.from_response(response)

    output_text = getattr(response, "output_text", "")
    if output_text:
        return output_text.strip(), usage_stats

    parts: list[str] = []
    for item in getattr(response, "output", []):
        for content in getattr(item, "content", []):
            text = getattr(content, "text", "")
            if text:
                parts.append(text)

    return "\n".join(parts).strip(), usage_stats


def _audit_cheatsheet_with_openai(
    cheatsheet_markdown: str,
    chunk_summaries: list[str],
    options: GenerationOptions,
) -> tuple[str, UsageStats]:
    system_prompt = (
        "You are an accuracy auditor for concept-based exam cheatsheets. "
        "Revise the draft so that it teaches concepts clearly in one-page form, while staying fully grounded in the lecture-slide notes. "
        "Every bullet must be directly supported, complete, non-duplicated, correctly classified, and useful for exam review. "
        "If a claim is unsupported or uncertain, delete it instead of repairing it with outside knowledge."
    )
    language_hint = _language_instruction(options.output_language)
    user_prompt = f"""
Audit the draft cheatsheet for accuracy.

Check for:
1. Unsupported claims
2. Incomplete sentences
3. Duplicated points
4. Generic filler
5. OCR artifacts
6. Missing formulas or definitions that are clearly present in the extracted notes
7. Misclassified content, such as putting examples under formulas

Revise the cheatsheet so that every bullet is accurate, complete, and exam-useful.

Revision rules:
- Use only information directly supported by the extracted slide notes.
- Remove unsupported, broken, generic, duplicated, or OCR-corrupted bullets.
- Move bullets into the correct section when they are misclassified.
- Add back missing formulas or definitions only when they are clearly present in the extracted notes.
- Preserve exact technical terms, formulas, and important numerical values.
- Keep each bullet complete and grammatically understandable.
- Keep the final result compact enough for one A4 page.
- Do not leave behind slide titles without explanation, questions without answers, or copied fragments.
- Ensure every bullet teaches a concept or clarifies an interpretation.
- Use the same exact section headings when supported:
{chr(10).join(f"## {heading}" for heading in CHEATSHEET_SECTION_ORDER)}
- Keep the title in the format `# [Course / Lecture Title] - A4 Cheatsheet`.
- Omit a section rather than padding it with filler.
- {language_hint}

Extracted slide notes:
{chr(10).join(chunk_summaries)}

Draft cheatsheet:
{cheatsheet_markdown}
""".strip()

    return _call_openai(system_prompt, user_prompt, max_output_tokens=2200)


def _get_runtime_config(name: str) -> str:
    env_value = os.getenv(name, "").strip()
    if env_value:
        return env_value

    secret_value = _read_streamlit_secret(name)
    if secret_value:
        return secret_value

    nested_key = {
        "OPENAI_API_KEY": "api_key",
        "OPENAI_MODEL": "model",
    }.get(name)
    if nested_key:
        return _read_streamlit_secret("openai", nested_key)

    return ""


def _read_response_field(obj: Any, name: str):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _read_streamlit_secret(*keys: str) -> str:
    if st is None:
        return ""

    try:
        value = st.secrets
        for key in keys:
            value = value[key]
    except Exception:
        return ""

    if value is None:
        return ""

    return str(value).strip()


def _heuristic_chunk_summary(chunk: str, options: GenerationOptions, chunk_index: int) -> str:
    candidates = _collect_candidates(chunk)
    caps = _section_caps(options)
    concept_items = _dedupe_lines(candidates["definitions"] + candidates["concepts"] + candidates["methods"])

    del chunk_index
    lines: list[str] = []
    lines.extend(_format_plain_bullets(concept_items, caps["concepts"]))

    if options.include_formulas:
        lines.extend(_format_plain_bullets(candidates["formulas"], caps["formulas"]))

    lines.extend(_format_plain_bullets(candidates["comparisons"], caps["comparisons"]))

    if options.include_exam_questions:
        lines.extend(_format_plain_bullets(candidates["exam"], caps["exam"]))

    if options.include_examples:
        lines.extend(_format_plain_bullets(candidates["examples"], caps["examples"]))

    return "\n".join(_dedupe_lines(lines)).strip()


def _generate_cheatsheet_heuristic(
    summary_text: str,
    options: GenerationOptions,
    source_text: str,
) -> str:
    candidates = _collect_candidates(summary_text + "\n" + source_text)
    labels = _section_labels(options.output_language)
    caps = _section_caps(options)
    title = _resolve_display_title(source_text, options)
    core_items = _dedupe_lines(candidates["definitions"] + candidates["concepts"] + candidates["methods"])
    exam_items = candidates["exam"]

    sections: list[str] = [f"# {title} - A4 Cheatsheet", ""]
    sections.extend(_section_block(labels["concepts"], core_items, caps["concepts"]))

    if options.include_formulas:
        sections.extend(_section_block(labels["formulas"], candidates["formulas"], caps["formulas"]))

    sections.extend(_section_block(labels["distinctions"], candidates["comparisons"], caps["comparisons"]))

    if options.include_examples:
        sections.extend(_section_block(labels["examples"], candidates["examples"], caps["examples"]))

    if options.include_exam_questions:
        sections.extend(_section_block(labels["exam"], exam_items, caps["exam"]))

    return "\n".join(sections).strip()


def _audit_cheatsheet_heuristic(cheatsheet_markdown: str) -> str:
    audited_lines: list[str] = []
    seen_bullets: set[str] = set()

    for raw_line in cheatsheet_markdown.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if audited_lines and audited_lines[-1] != "":
                audited_lines.append("")
            continue

        if stripped.startswith("#"):
            if audited_lines and audited_lines[-1] == "":
                audited_lines.pop()
            audited_lines.append(stripped)
            continue

        if not stripped.startswith("- "):
            continue

        bullet = _compact_line(stripped[2:])
        if not bullet or _looks_like_generic_filler(bullet) or _looks_incomplete(bullet):
            continue

        bullet_key = bullet.lower()
        if bullet_key in seen_bullets:
            continue

        seen_bullets.add(bullet_key)
        audited_lines.append(f"- {bullet}")

    while audited_lines and audited_lines[-1] == "":
        audited_lines.pop()

    return "\n".join(audited_lines)


def _collect_candidates(text: str) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = defaultdict(list)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        plain = _strip_markdown(line)
        if not plain or plain.lower().startswith("source:"):
            continue

        if line.startswith("#"):
            categories["headings"].append(plain)
            categories["concepts"].append(plain)
            continue

        if _looks_like_formula(plain):
            categories["formulas"].append(plain)
        if _looks_like_definition(plain):
            categories["definitions"].append(plain)
        if _looks_like_comparison(plain):
            categories["comparisons"].append(plain)
        if _looks_like_method(plain):
            categories["methods"].append(plain)
        if _looks_like_exam_signal(plain):
            categories["exam"].append(plain)
        if _looks_like_example(plain):
            categories["examples"].append(plain)

        if line.startswith(("- ", "* ")) or re.match(r"^\d+\.\s+", line):
            categories["concepts"].append(plain)
        elif len(plain.split()) <= 16 and not plain.endswith(":"):
            categories["concepts"].append(plain)

    for key, items in list(categories.items()):
        categories[key] = _dedupe_lines(items)

    return categories


def _section_caps(options: GenerationOptions) -> dict[str, int]:
    base = {
        "1-page A4": 4,
        "2-page A4": 7,
        "concise summary": 5,
        "detailed summary": 9,
    }.get(options.target_length, 6)

    modifier = {
        "More concise": -1,
        "Balanced": 0,
        "More detailed": 2,
    }.get(options.density, 0)

    return {
        "concepts": max(4, base + modifier),
        "definitions": max(3, base - 1 + modifier),
        "formulas": max(3, base - 1 + modifier),
        "comparisons": max(3, base - 2 + modifier),
        "methods": max(3, base - 1 + modifier),
        "exam": max(3, base - 2 + modifier),
        "examples": max(2, base - 3 + modifier),
        "checklist": max(5, base + modifier),
    }


def _format_bullets(title: str, items: list[str], limit: int) -> list[str]:
    if not items:
        return []
    lines = [f"### {title}"]
    for item in items[:limit]:
        lines.append(f"- {_compact_line(item)}")
    return lines


def _format_plain_bullets(items: list[str], limit: int) -> list[str]:
    return [f"- {_compact_line(item)}" for item in items[:limit]]


def _section_block(
    title: str,
    items: list[str],
    limit: int,
) -> list[str]:
    selected = [
        compacted
        for item in items[:limit]
        for compacted in [_compact_line(item)]
        if compacted and not _looks_like_generic_filler(compacted)
    ]
    if not selected:
        return []

    section = [f"## {title}"]
    section.extend(f"- {item}" for item in selected)
    section.append("")
    return section


def _build_checklist(candidates: dict[str, list[str]], options: GenerationOptions) -> list[str]:
    checklist: list[str] = []
    prefix = {
        "English": "Verify",
        "Chinese": "检查",
        "Bilingual": "Check / 检查",
    }.get(options.output_language, "Verify")

    for concept in candidates.get("concepts", [])[:3]:
        checklist.append(f"{prefix} you can explain: {_compact_line(concept, 18)}")
    for definition in candidates.get("definitions", [])[:2]:
        checklist.append(f"{prefix} the exact distinction in: {_compact_line(definition, 18)}")
    if options.include_formulas:
        for formula in candidates.get("formulas", [])[:2]:
            checklist.append(f"{prefix} when to use: {_compact_line(formula, 16)}")

    return _dedupe_lines(checklist)


def _resolve_title(source_text: str, options: GenerationOptions) -> str:
    if options.course_name.strip():
        return options.course_name.strip()

    for line in source_text.splitlines():
        candidate = _strip_markdown(line.strip())
        if 3 <= len(candidate) <= 80 and len(candidate.split()) <= 12:
            return candidate

    return ""


def _resolve_display_title(source_text: str, options: GenerationOptions) -> str:
    resolved = _resolve_title(source_text, options).strip()
    return resolved or "Exam Cheatsheet"


def _section_labels(language: str) -> dict[str, str]:
    if language == "Chinese":
        return {
            "concepts": "1. 核心概念",
            "formulas": "2. 公式 / 指标",
            "distinctions": "3. 必会区分",
            "examples": "4. 经典例子 / 发现",
            "exam": "5. 考试陷阱",
        }
    if language == "Bilingual":
        return {
            "concepts": "1. 核心概念 / Core Concepts",
            "formulas": "2. 公式 / 指标 / Key Measures / Formulas",
            "distinctions": "3. 必会区分 / Must-Know Distinctions",
            "examples": "4. 经典例子 / 发现 / Classic Examples / Findings",
            "exam": "5. 考试陷阱 / Exam Traps",
        }
    return {
        "concepts": "1. Core Concepts",
        "formulas": "2. Key Measures / Formulas",
        "distinctions": "3. Must-Know Distinctions",
        "examples": "4. Classic Examples / Findings",
        "exam": "5. Exam Traps",
    }


def _language_instruction(language: str) -> str:
    if language == "Chinese":
        return "Write the cheat sheet in Chinese."
    if language == "Bilingual":
        return "Write a bilingual cheat sheet with compact English and Chinese phrasing."
    return "Write the cheat sheet in English."


def _target_word_budget(target_length: str, density: str) -> str:
    base = {
        "1-page A4": "450-650",
        "2-page A4": "800-1200",
        "concise summary": "350-550",
        "detailed summary": "900-1400",
    }.get(target_length, "500-900")

    if density == "More concise":
        return base.split("-")[0] + "-" + str(max(int(base.split("-")[0]) + 120, int(base.split("-")[1]) - 120))
    if density == "More detailed":
        return str(max(int(base.split("-")[0]), int(base.split("-")[0]) + 100)) + "-" + str(int(base.split("-")[1]) + 200)
    return base


def _strip_markdown(line: str) -> str:
    line = re.sub(r"^#+\s*", "", line)
    line = re.sub(r"^[-*]\s*", "", line)
    line = re.sub(r"^\d+\.\s*", "", line)
    return line.strip()


def _looks_like_formula(text: str) -> bool:
    if len(text) > 140:
        return False
    formula_patterns = (
        "=",
        "->",
        "<=",
        ">=",
        "f(",
        "p(",
        "o(",
        "theta",
        "lambda",
        "sigma",
        "delta",
        "sum",
        "mean",
        "%",
    )
    return any(pattern in text.lower() for pattern in formula_patterns) or bool(re.search(r"\d+\s*[\+\-\*/]\s*\d+", text))


def _looks_like_definition(text: str) -> bool:
    lowered = text.lower()
    if lowered.startswith(("definition", "defn")):
        return True
    if ":" in text and len(text.split(":")[0].split()) <= 7:
        return True
    return any(keyword in lowered for keyword in ("defined as", "refers to", "means", "is the"))


def _looks_like_comparison(text: str) -> bool:
    lowered = text.lower()
    keywords = ("vs", "versus", "difference", "compare", "compared with", "whereas", "unlike")
    return any(keyword in lowered for keyword in keywords)


def _looks_like_method(text: str) -> bool:
    lowered = text.lower()
    if re.match(r"^(step\s*\d+|\d+\.)", lowered):
        return True
    return any(keyword in lowered for keyword in ("procedure", "algorithm", "workflow", "first", "then", "finally", "solve by"))


def _looks_like_exam_signal(text: str) -> bool:
    lowered = text.lower()
    keywords = ("exam", "trap", "pitfall", "mistake", "important", "remember", "beware", "frequently", "commonly")
    return any(keyword in lowered for keyword in keywords)


def _looks_like_example(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in ("example", "e.g.", "for instance", "sample"))


def _dedupe_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for line in lines:
        normalized = re.sub(r"\s+", " ", line.strip().lower())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(line.strip())

    return result


def _compact_line(text: str, max_words: int = 24) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    shortened = " ".join(words[:max_words])
    return textwrap.shorten(shortened, width=160, placeholder=" ...")


def _looks_like_generic_filler(text: str) -> bool:
    lowered = text.lower()
    filler_phrases = (
        "lecture title not clearly identified",
        "verify you can explain",
        "the exact distinction in:",
        "when to use:",
        "check / 检查",
    )
    return any(phrase in lowered for phrase in filler_phrases)


def _looks_incomplete(text: str) -> bool:
    if len(text.split()) < 2:
        return True
    if text.endswith((":","/","-","(","[","{")):
        return True
    if text.count("(") != text.count(")"):
        return True
    if text.count("[") != text.count("]"):
        return True
    return False

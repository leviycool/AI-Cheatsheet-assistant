"""Chunk summarization and final cheat sheet generation helpers."""

from __future__ import annotations

import os
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - safe fallback when dependency is absent
    OpenAI = None  # type: ignore[assignment]


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


def is_openai_configured() -> bool:
    """Return True when the OpenAI SDK and API key are both available."""
    return OpenAI is not None and bool(os.getenv("OPENAI_API_KEY"))


def summarize_chunks(chunks: list[str], options: GenerationOptions) -> list[str]:
    """Summarize each chunk before the final aggregation step."""
    if not chunks:
        return []

    summaries: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        if is_openai_configured():
            try:
                summaries.append(_summarize_chunk_with_openai(chunk, options, index, len(chunks)))
                continue
            except Exception:
                pass

        summaries.append(_heuristic_chunk_summary(chunk, options, index))

    return summaries


def generate_cheatsheet(
    chunk_summaries: list[str],
    options: GenerationOptions,
    source_text: str | None = None,
) -> str:
    """Combine chunk summaries into one condensed, exam-oriented cheat sheet."""
    if is_openai_configured():
        try:
            return _generate_cheatsheet_with_openai(chunk_summaries, options)
        except Exception:
            pass

    combined_source = "\n\n".join(chunk_summaries) if chunk_summaries else (source_text or "")
    return _generate_cheatsheet_heuristic(combined_source, options, source_text or combined_source)


def _summarize_chunk_with_openai(
    chunk: str,
    options: GenerationOptions,
    chunk_index: int,
    chunk_total: int,
) -> str:
    system_prompt = (
        "You are an exam-preparation compression assistant. Summarize source material into high-density, "
        "exam-relevant study notes. Preserve definitions, formulas, distinctions, procedures, traps, and examples. "
        "Use markdown bullets and headings. Avoid long paragraphs."
    )
    user_prompt = f"""
Chunk {chunk_index} of {chunk_total}

Course/topic: {options.course_name or "Infer from material"}
Output language: {options.output_language}
Target length: {options.target_length}
Focus style: {options.focus_style}
Include examples: {options.include_examples}
Include formulas: {options.include_formulas}
Include possible exam questions: {options.include_exam_questions}
Density preference: {options.density}

Please summarize this chunk into compact study notes with the sections below when relevant:
- Key concepts
- Definitions
- Formulas / models / frameworks
- Methods / procedures
- Comparisons
- Exam traps / likely testable distinctions
- Mini examples

Keep it compact and source-grounded.

Source chunk:
{chunk}
""".strip()

    return _call_openai(system_prompt, user_prompt, max_output_tokens=1400)


def _generate_cheatsheet_with_openai(chunk_summaries: list[str], options: GenerationOptions) -> str:
    system_prompt = (
        "You are Cheatsheet AI, a high-density exam cheat sheet generator. Produce markdown that is compact, "
        "structured, and print-friendly. Prioritize likely exam content, formulas, procedures, comparisons, "
        "and common traps. Prefer bullets, compact tables, and short lines over prose."
    )
    word_budget = _target_word_budget(options.target_length, options.density)
    language_hint = _language_instruction(options.output_language)
    variation_hint = "Use a fresh structure emphasis." if options.variant else "Use the strongest default structure."

    user_prompt = f"""
Create one final cheat sheet in markdown.

Course/topic: {options.course_name or "Infer from the summaries"}
Output language: {options.output_language}
Target length: {options.target_length}
Approximate word budget: {word_budget}
Focus style: {options.focus_style}
Include examples: {options.include_examples}
Include formulas: {options.include_formulas}
Include possible exam questions: {options.include_exam_questions}
Density preference: {options.density}
Variation note: {variation_hint}

Formatting requirements:
- Highly condensed but readable
- Clear section headings
- Prefer bullets, mini tables, formulas, and short explanations
- Avoid fluffy prose
- Make the result suitable for an A4 exam cheat sheet
- Keep the structure below in this exact order when the content exists

Required structure:
1. Course / Topic Title
2. Key Concepts
3. Important Definitions
4. Core Formulas / Models / Frameworks
5. Key Comparisons
6. Step-by-step Methods / Procedures
7. Common Exam Questions or Traps
8. Mini Examples if useful
9. Final Quick Review Checklist

Language instruction:
{language_hint}

Chunk summaries:
{chr(10).join(chunk_summaries)}
""".strip()

    return _call_openai(system_prompt, user_prompt, max_output_tokens=2600)


def _call_openai(system_prompt: str, user_prompt: str, max_output_tokens: int) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
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

    output_text = getattr(response, "output_text", "")
    if output_text:
        return output_text.strip()

    parts: list[str] = []
    for item in getattr(response, "output", []):
        for content in getattr(item, "content", []):
            text = getattr(content, "text", "")
            if text:
                parts.append(text)

    return "\n".join(parts).strip()


def _heuristic_chunk_summary(chunk: str, options: GenerationOptions, chunk_index: int) -> str:
    candidates = _collect_candidates(chunk)
    caps = _section_caps(options)

    lines = [f"## Chunk {chunk_index} Highlights"]
    lines.extend(_format_bullets("Key concepts", candidates["concepts"], caps["concepts"]))
    lines.extend(_format_bullets("Definitions", candidates["definitions"], caps["definitions"]))

    if options.include_formulas:
        lines.extend(_format_bullets("Formulas", candidates["formulas"], caps["formulas"]))

    lines.extend(_format_bullets("Methods", candidates["methods"], caps["methods"]))
    lines.extend(_format_bullets("Comparisons", candidates["comparisons"], caps["comparisons"]))

    if options.include_exam_questions:
        lines.extend(_format_bullets("Exam signals", candidates["exam"], caps["exam"]))

    if options.include_examples:
        lines.extend(_format_bullets("Examples", candidates["examples"], caps["examples"]))

    return "\n".join(lines).strip()


def _generate_cheatsheet_heuristic(
    summary_text: str,
    options: GenerationOptions,
    source_text: str,
) -> str:
    candidates = _collect_candidates(summary_text + "\n" + source_text)
    labels = _section_labels(options.output_language)
    caps = _section_caps(options)
    title = _resolve_title(source_text, options)

    sections: list[str] = [f"# {title}", ""]
    sections.extend(_section_block(labels["concepts"], candidates["concepts"], caps["concepts"], options.output_language))
    sections.extend(_section_block(labels["definitions"], candidates["definitions"], caps["definitions"], options.output_language))

    if options.include_formulas:
        sections.extend(_section_block(labels["formulas"], candidates["formulas"], caps["formulas"], options.output_language))

    sections.extend(
        _section_block(
            labels["comparisons"],
            candidates["comparisons"],
            caps["comparisons"],
            options.output_language,
            fallback=_fallback_line("comparisons", options.output_language),
        )
    )
    sections.extend(
        _section_block(
            labels["methods"],
            candidates["methods"],
            caps["methods"],
            options.output_language,
            fallback=_fallback_line("methods", options.output_language),
        )
    )

    if options.include_exam_questions:
        sections.extend(
            _section_block(
                labels["exam"],
                candidates["exam"],
                caps["exam"],
                options.output_language,
                fallback=_fallback_line("exam", options.output_language),
            )
        )

    if options.include_examples:
        sections.extend(
            _section_block(
                labels["examples"],
                candidates["examples"],
                caps["examples"],
                options.output_language,
                fallback=_fallback_line("examples", options.output_language),
            )
        )

    checklist_items = _build_checklist(candidates, options)
    sections.extend(_section_block(labels["checklist"], checklist_items, caps["checklist"], options.output_language))

    return "\n".join(sections).strip()


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


def _section_block(
    title: str,
    items: list[str],
    limit: int,
    language: str,
    fallback: str | None = None,
) -> list[str]:
    section = [f"## {title}"]
    selected = [_compact_line(item) for item in items[:limit]]

    if not selected and fallback:
        selected = [fallback]
    if not selected:
        selected = [_fallback_line("generic", language)]

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

    return "Exam Cheat Sheet"


def _section_labels(language: str) -> dict[str, str]:
    if language == "Chinese":
        return {
            "concepts": "核心概念",
            "definitions": "重要定义",
            "formulas": "核心公式 / 模型 / 框架",
            "comparisons": "关键比较",
            "methods": "步骤 / 方法",
            "exam": "常见考试题型 / 易错点",
            "examples": "小例题 / 示例",
            "checklist": "考前速查清单",
        }
    if language == "Bilingual":
        return {
            "concepts": "核心概念 / Key Concepts",
            "definitions": "重要定义 / Important Definitions",
            "formulas": "核心公式 / 模型 / 框架 / Core Formulas / Models / Frameworks",
            "comparisons": "关键比较 / Key Comparisons",
            "methods": "步骤 / 方法 / Step-by-step Methods / Procedures",
            "exam": "常见考试题型 / 易错点 / Common Exam Questions or Traps",
            "examples": "小例题 / 示例 / Mini Examples",
            "checklist": "考前速查清单 / Final Quick Review Checklist",
        }
    return {
        "concepts": "Key Concepts",
        "definitions": "Important Definitions",
        "formulas": "Core Formulas / Models / Frameworks",
        "comparisons": "Key Comparisons",
        "methods": "Step-by-step Methods / Procedures",
        "exam": "Common Exam Questions or Traps",
        "examples": "Mini Examples",
        "checklist": "Final Quick Review Checklist",
    }


def _language_instruction(language: str) -> str:
    if language == "Chinese":
        return "Write the cheat sheet in Chinese."
    if language == "Bilingual":
        return "Write a bilingual cheat sheet with compact English and Chinese phrasing."
    return "Write the cheat sheet in English."


def _fallback_line(section: str, language: str) -> str:
    fallback_map = {
        "English": {
            "comparisons": "Compare look-alike terms, assumptions, and when each method applies.",
            "methods": "List the main problem-solving sequence, trigger conditions, and final checks.",
            "exam": "Flag definitions, boundary cases, and formula-selection mistakes that are easy to test.",
            "examples": "Add one micro-example that shows how the rule or formula is used.",
            "generic": "No strong signal was extracted from the source for this section.",
        },
        "Chinese": {
            "comparisons": "比较容易混淆的术语、前提条件，以及各方法的适用场景。",
            "methods": "列出主要解题步骤、触发条件和最后检查点。",
            "exam": "标记定义、边界条件和公式选择错误等高频考点。",
            "examples": "补一个能体现规则或公式用途的微型示例。",
            "generic": "该部分在原始材料中没有提取到明显信号。",
        },
        "Bilingual": {
            "comparisons": "Compare similar terms and method assumptions / 比较相近术语与方法前提。",
            "methods": "List the main solving steps and checks / 列出主要步骤与检查点。",
            "exam": "Flag easy-to-test traps and boundary cases / 标记常考陷阱与边界条件。",
            "examples": "Add one micro-example / 补一个微型示例。",
            "generic": "No strong signal extracted / 该部分暂无明显提取信号。",
        },
    }
    return fallback_map.get(language, fallback_map["English"]).get(section, fallback_map["English"]["generic"])


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

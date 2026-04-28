"""Chunk summarization and final cheat sheet generation helpers."""

from __future__ import annotations

import os
import re
import textwrap
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
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
FALLBACK_OPENAI_MODELS = ("gpt-4.1-mini", "gpt-4o-mini")

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
    use_web_search: bool = False
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


@dataclass
class LLMCallResult:
    text: str
    usage: dict[str, int | bool]
    model: str
    raw_usage: Any = None
    parsed_json: Any = None
    sources: list[dict[str, str]] = field(default_factory=list)
    response_keys: list[str] = field(default_factory=list)
    usage_keys: list[str] = field(default_factory=list)
    response_type: str = ""

    def to_usage_stats(self) -> UsageStats:
        return UsageStats(
            api_calls=1,
            usage_available_calls=1 if self.usage.get("available", False) else 0,
            input_tokens=int(self.usage.get("input_tokens", 0)),
            output_tokens=int(self.usage.get("output_tokens", 0)),
            total_tokens=int(self.usage.get("total_tokens", 0)),
            cached_input_tokens=int(self.usage.get("cached_input_tokens", 0)),
            reasoning_tokens=int(self.usage.get("reasoning_tokens", 0)),
        )


@dataclass
class ConceptRecord:
    concept: str
    category: str
    slide_context: str
    appears_in_slides: bool = True
    importance: str = "medium"
    needs_web_clarification: bool = False
    reason_for_clarification: str = ""
    final_explanation: str = ""
    definition_from_slides: str = ""
    why_it_matters: str = ""
    formula_or_measure: str = ""
    distinction: str = ""
    example_or_finding: str = ""
    exam_trap: str = ""
    web_definition: str = ""
    final_definition: str = ""
    web_source_title: str = ""
    web_source_url: str = ""
    sources: list[dict[str, str]] = field(default_factory=list)
    kind: str = ""


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
    if input_details is None:
        input_details = _read_response_field(usage, "prompt_tokens_details")
    if output_details is None:
        output_details = _read_response_field(usage, "completion_tokens_details")

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


def extract_concepts(
    chunks: str | list[str],
    options: GenerationOptions,
    source_text: str | None = None,
) -> tuple[list[dict[str, Any]], UsageStats]:
    """Extract a raw concept inventory from slide text."""
    usage_totals = UsageStats()
    if isinstance(chunks, str):
        source_text = source_text or chunks
        chunks = [chunks]
    if not chunks:
        return [], usage_totals

    if is_openai_configured():
        chunk_candidates: list[dict[str, Any]] = []
        batched_chunks = _group_chunks_for_concept_extraction(chunks)
        for index, chunk in enumerate(batched_chunks, start=1):
            try:
                result = _extract_chunk_concepts_with_openai(chunk, options, index, len(batched_chunks))
                usage_totals.add(result.to_usage_stats())
                parsed = result.parsed_json or {}
                normalized = _normalize_concept_records(parsed.get("concepts", []))
                if not normalized:
                    _record_pipeline_error(
                        "extraction",
                        RuntimeError("OpenAI concept extraction returned no parseable concepts."),
                        {"batch_index": index, "batch_total": len(batched_chunks), "output_preview": result.text[:400]},
                    )
                    continue
                chunk_candidates.extend(normalized)
            except Exception:
                continue
        if chunk_candidates:
            return chunk_candidates, usage_totals

    return _extract_concepts_heuristic(source_text or "\n\n".join(chunks), options), usage_totals


def clean_concepts(
    concept_inventory: list[dict[str, Any]],
    options: GenerationOptions,
    source_text: str | None = None,
) -> tuple[list[dict[str, Any]], UsageStats]:
    """Clean the concept inventory and keep only high-value A4-worthy concepts."""
    usage_totals = UsageStats()
    if not concept_inventory:
        return [], usage_totals

    if is_openai_configured():
        try:
            result = _clean_concepts_with_openai(concept_inventory, options, source_text or "")
            usage_totals.add(result.to_usage_stats())
            cleaned = _normalize_concept_records((result.parsed_json or {}).get("concepts", []))
            cleaned = _prioritize_concepts(_finalize_concept_records(cleaned), options)
            if cleaned:
                return cleaned, usage_totals
            _record_pipeline_error(
                "cleaning",
                RuntimeError("OpenAI concept cleaning returned no usable concepts."),
                {"candidate_count": len(concept_inventory), "output_preview": result.text[:400]},
            )
        except Exception:
            pass

    return _clean_concepts_heuristic(concept_inventory, options), usage_totals


def clarify_concepts(
    concepts: list[dict[str, Any]],
    options: GenerationOptions,
) -> tuple[list[dict[str, Any]], UsageStats]:
    """Optionally clarify slide concepts with web definitions, without adding new concepts."""
    usage_totals = UsageStats()
    if not concepts or not options.use_web_search or not is_openai_configured():
        return concepts, usage_totals

    clarified: list[dict[str, Any]] = []
    for concept in concepts:
        if not concept.get("needs_web_clarification"):
            clarified.append(_ensure_final_definition(dict(concept)))
            continue

        try:
            result = _clarify_concept_with_web_openai(concept, options)
            usage_totals.add(result.to_usage_stats())
            parsed = result.parsed_json or {}
            updated = dict(concept)
            primary_source = result.sources[0] if result.sources else {}
            updated["web_definition"] = _safe_text(parsed.get("web_definition"))
            updated["final_definition"] = _safe_text(parsed.get("final_definition")) or _safe_text(
                concept.get("definition_from_slides")
            )
            updated["why_it_matters"] = _safe_text(parsed.get("why_it_matters")) or _safe_text(
                concept.get("why_it_matters")
            )
            updated["final_explanation"] = _safe_text(parsed.get("final_definition")) or _safe_text(
                parsed.get("final_explanation")
            ) or _safe_text(updated.get("final_definition"))
            updated["web_source_title"] = _safe_text(primary_source.get("title"))
            updated["web_source_url"] = _safe_text(primary_source.get("url"))
            updated["sources"] = result.sources
            clarified.append(_ensure_final_definition(updated))
        except Exception:
            clarified.append(_ensure_final_definition(dict(concept)))

    return _prioritize_concepts(_finalize_concept_records(clarified), options), usage_totals


def clarify_concepts_with_web(
    concepts: list[dict[str, Any]],
    options: GenerationOptions,
) -> tuple[list[dict[str, Any]], UsageStats]:
    """Backward-compatible alias for clarify_concepts."""
    return clarify_concepts(concepts, options)


def summarize_chunks(chunks: list[str], options: GenerationOptions) -> tuple[list[str], UsageStats]:
    """Summarize each chunk before the final aggregation step."""
    usage_totals = UsageStats()
    if not chunks:
        return [], usage_totals

    summaries: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        if is_openai_configured():
            try:
                result = _summarize_chunk_with_openai(chunk, options, index, len(chunks))
                if not result.text.strip():
                    _record_pipeline_error(
                        "extraction",
                        RuntimeError("OpenAI chunk summarization returned empty output."),
                        {"chunk_index": index, "chunk_total": len(chunks)},
                    )
                    raise RuntimeError("Empty chunk summary.")
                summaries.append(result.text)
                usage_totals.add(result.to_usage_stats())
                continue
            except Exception:
                pass

        summaries.append(_heuristic_chunk_summary(chunk, options, index))

    return summaries, usage_totals


def generate_cheatsheet(
    chunk_summaries: list[str] | list[dict[str, Any]],
    options: GenerationOptions,
    source_text: str | None = None,
) -> tuple[str, UsageStats]:
    """Generate a cheatsheet from either legacy chunk summaries or cleaned concept records."""
    if chunk_summaries and isinstance(chunk_summaries[0], dict):
        return generate_cheatsheet_from_concepts(chunk_summaries, options, source_text=source_text)

    if is_openai_configured():
        try:
            result = _generate_cheatsheet_with_openai(chunk_summaries, options)
            if _has_substantive_cheatsheet(result.text):
                return result.text, result.to_usage_stats()
            _record_pipeline_error(
                "generation",
                RuntimeError("OpenAI generation returned empty or title-only output."),
                {"mode": "legacy_generation", "output_preview": result.text[:400]},
            )
        except Exception:
            pass

    combined_source = "\n\n".join(chunk_summaries) if chunk_summaries else (source_text or "")
    return _generate_cheatsheet_heuristic(combined_source, options, source_text or combined_source), UsageStats()


def audit_cheatsheet(
    cheatsheet_markdown: str,
    chunk_summaries: list[str] | list[dict[str, Any]],
    options: GenerationOptions,
) -> tuple[str, UsageStats]:
    """Audit a cheatsheet against either legacy summaries or cleaned concept records."""
    if chunk_summaries and isinstance(chunk_summaries[0], dict):
        return audit_cheatsheet_from_concepts(cheatsheet_markdown, chunk_summaries, options)

    if is_openai_configured():
        try:
            result = _audit_cheatsheet_with_openai(cheatsheet_markdown, chunk_summaries, options)
            if _has_substantive_cheatsheet(result.text):
                return result.text, result.to_usage_stats()
            _record_pipeline_error(
                "audit",
                RuntimeError("OpenAI audit returned empty or title-only output."),
                {"mode": "legacy_audit", "output_preview": result.text[:400]},
            )
        except Exception:
            pass

    return _audit_cheatsheet_heuristic(cheatsheet_markdown), UsageStats()


def generate_cheatsheet_from_concepts(
    concepts: list[dict[str, Any]],
    options: GenerationOptions,
    source_text: str | None = None,
) -> tuple[str, UsageStats]:
    """Generate an A4 cheatsheet from cleaned concept records."""
    concepts = _finalize_concept_records(concepts)
    if is_openai_configured():
        try:
            result = _generate_concept_cheatsheet_with_openai(concepts, options)
            if _has_substantive_cheatsheet(result.text):
                return result.text, result.to_usage_stats()
            _record_pipeline_error(
                "generation",
                RuntimeError("OpenAI concept generation returned empty or title-only output."),
                {"mode": "concept_generation", "output_preview": result.text[:400], "concept_count": len(concepts)},
            )
        except Exception:
            pass

    return _generate_concept_cheatsheet_heuristic(concepts, options, source_text or ""), UsageStats()


def audit_cheatsheet_from_concepts(
    cheatsheet_markdown: str,
    concepts: list[dict[str, Any]],
    options: GenerationOptions,
) -> tuple[str, UsageStats]:
    """Audit the final cheatsheet against cleaned concept records."""
    concepts = _finalize_concept_records(concepts)
    if is_openai_configured():
        try:
            result = _audit_concept_cheatsheet_with_openai(cheatsheet_markdown, concepts, options)
            if _has_substantive_cheatsheet(result.text):
                return result.text, result.to_usage_stats()
            _record_pipeline_error(
                "audit",
                RuntimeError("OpenAI concept audit returned empty or title-only output."),
                {"mode": "concept_audit", "output_preview": result.text[:400], "concept_count": len(concepts)},
            )
        except Exception:
            pass

    return _audit_cheatsheet_heuristic(cheatsheet_markdown), UsageStats()


def _extract_chunk_concepts_with_openai(
    chunk: str,
    options: GenerationOptions,
    chunk_index: int,
    chunk_total: int,
) -> LLMCallResult:
    system_prompt = (
        "You are extracting candidate concepts from lecture slides for a one-page exam cheatsheet. "
        "Use only concepts that actually appear in the slides. Return JSON only."
    )
    user_prompt = f"""
Return a JSON object with key `concepts`.

Slide chunk: {chunk_index} of {chunk_total}
Course/topic: {options.course_name or "Infer from material"}
Output language: {options.output_language}

Extract 4 to 10 candidate concepts from this chunk. A concept can be a term, theory, model, measure,
formula, method, distinction, named study/example, or exam trap.

Remove:
- OCR fragments
- incomplete phrases
- generic slide headings
- duplicated concepts
- questions without answers
- decorative text

For each concept object include:
- concept
- category
- slide_context
- appears_in_slides
- importance
- needs_web_clarification
- reason_for_clarification
- definition_from_slides
- why_it_matters
- formula_or_measure
- distinction
- example_or_finding
- exam_trap
- final_explanation

Rules:
- Use only slide-supported information.
- Keep every string short and complete.
- Set `appears_in_slides` to true only when the concept clearly appears in this chunk.
- Set `importance` to high, medium, or low.
- If a field is unsupported, use an empty string except booleans.
- Set `needs_web_clarification` to true only when the concept clearly appears in slides but the slide wording
  is too fragmentary or implicit to produce a clean exam-ready definition.

Source chunk:
{chunk}
""".strip()
    return _call_openai(
        system_prompt,
        user_prompt,
        max_output_tokens=1800,
        step_name="extraction",
        text_format={"type": "json_object"},
    )


def _clean_concepts_with_openai(
    candidate_concepts: list[dict[str, Any]],
    options: GenerationOptions,
    source_text: str,
) -> LLMCallResult:
    system_prompt = (
        "You are cleaning and prioritizing a slide-derived concept inventory for a graduate-level exam cheatsheet. "
        "Use only the supplied concept records. Return JSON only."
    )
    target_limit = _concept_limit(options)
    user_prompt = f"""
Return a JSON object with key `concepts`.

Goal:
- Keep 10 to {target_limit} important concepts.
- Keep only high and strong medium concepts that fit on an A4 exam cheatsheet.
- Prioritize concepts that are defined, repeated, formula-based, contrasted, method-like, example-backed, or exam-relevant.

Remove:
- OCR fragments
- incomplete phrases
- generic headings
- duplicated concepts
- questions without answers
- decorative text

Rules:
- Do not invent concepts.
- Do not add concepts that were not present in the candidate list.
- Merge repeated concepts into one clean record.
- Preserve formulas and numerical values when present.
- Make each record short, complete, conceptual, and exam-useful.
- Set `appears_in_slides` to true only if the concept clearly appears in the supplied inventory.
- Use categories only from: definition, measure, formula, theory, method, distinction, example, exam_trap.
- Set `importance` to high, medium, or low.
- Use `reason_for_clarification` only when `needs_web_clarification` is true.
- Write `final_explanation` as the compact explanation you would want the final cheatsheet to use.

Candidate concept records:
{json.dumps(candidate_concepts, ensure_ascii=True, indent=2)}

Slide title hints:
{_compact_line(source_text, 120)}
""".strip()
    return _call_openai(
        system_prompt,
        user_prompt,
        max_output_tokens=2600,
        step_name="cleaning",
        text_format={"type": "json_object"},
    )


def _clarify_concept_with_web_openai(concept: dict[str, Any], options: GenerationOptions) -> LLMCallResult:
    system_prompt = (
        "You clarify a concept that already appears in lecture slides. "
        "Use web search only to improve wording or provide a standard definition. "
        "If web and slides conflict, follow the slides. Return JSON only."
    )
    user_prompt = f"""
Return a JSON object with keys:
- concept
- web_definition
- final_definition
- why_it_matters
- final_explanation

Concept from slides:
{json.dumps(concept, ensure_ascii=True, indent=2)}

Rules:
- Do not introduce any new concept.
- Do not broaden this into a textbook summary.
- Keep the wording compact and exam-ready.
- If the slides already dominate the meaning, keep `final_definition` aligned with the slides.
- If web search is not helpful, leave `web_definition` empty and simply restate the slide meaning more clearly.
- Prefer university course pages, academic glossaries, official documentation, and reputable educational sources.
- Use Wikipedia only as a fallback for a basic definition.
""".strip()
    return _call_openai(
        system_prompt,
        user_prompt,
        max_output_tokens=900,
        step_name="web_clarification",
        tools=[{"type": "web_search"}],
        include=["web_search_call.action.sources"],
        text_format={"type": "json_object"},
        tool_choice="auto",
    )


def _generate_concept_cheatsheet_with_openai(
    concepts: list[dict[str, Any]],
    options: GenerationOptions,
) -> LLMCallResult:
    system_prompt = (
        "You are a graduate teaching assistant creating a one-page exam cheatsheet from a cleaned concept inventory. "
        "You are not summarizing slides page by page. Teach the most important concepts clearly and compactly."
    )
    word_budget = _target_word_budget(options.target_length, options.density)
    language_hint = _language_instruction(options.output_language)
    user_prompt = f"""
Create one polished, one-page A4 exam cheatsheet in markdown.

Requirements:
- Prioritize concepts over slide order.
- Every bullet must be understandable without the slides.
- Use only the supplied concept records.
- Do not add concepts not present in the records.
- Prefer high-importance concepts, then strong medium-importance concepts.
- Merge repeated ideas.
- Keep it compact enough for one A4 page.
- Keep definitions before minor examples.
- Keep formulas, distinctions, and exam traps when they are high-value.
- Use the exact section headings below when supported:
{chr(10).join(f"## {heading}" for heading in CHEATSHEET_SECTION_ORDER)}
- Under `## 3. Must-Know Distinctions`, prefer a markdown table when possible.
- Do not output long paragraphs.
- Approximate word budget: {word_budget}
- {language_hint}

Writing style:
- `Core Concepts`: Concept: definition + why it matters.
- `Measures / Formulas`: measure/formula + interpretation + high/low meaning if supported.
- `Examples / Findings`: include only examples that clarify major concepts.
- `Exam Traps`: write likely wrong idea -> correct idea when supported.
- Delete any concept that does not teach something exam-useful.

Concept records:
{json.dumps(concepts, ensure_ascii=True, indent=2)}
""".strip()
    return _call_openai(system_prompt, user_prompt, max_output_tokens=2400, step_name="generation")


def _audit_concept_cheatsheet_with_openai(
    cheatsheet_markdown: str,
    concepts: list[dict[str, Any]],
    options: GenerationOptions,
) -> LLMCallResult:
    system_prompt = (
        "You are auditing a concept-first graduate-level exam cheatsheet for accuracy, clarity, and A4 usefulness. "
        "Keep only content that is supported by the supplied concept records."
    )
    language_hint = _language_instruction(options.output_language)
    user_prompt = f"""
Audit and revise the cheatsheet.

Remove or fix anything that fails:
- Does this explain a real concept?
- Does the concept appear in the uploaded slides?
- Is the explanation complete?
- Is it useful for exams?
- Is it short enough for A4?
- Is it free from OCR artifacts?
- Is it understandable without the slides?
- Is it non-duplicated?

Rules:
- Use only the concept records.
- Do not add unrelated concepts.
- Do not leave generic slide headings, unanswered slide questions, or broken fragments.
- Prefer definitions, formulas, distinctions, and high-value exam traps over minor detail.
- Keep the exact section structure when supported:
{chr(10).join(f"## {heading}" for heading in CHEATSHEET_SECTION_ORDER)}
- Keep the title format `# [Course / Lecture Title] - A4 Cheatsheet`.
- {language_hint}

Concept records:
{json.dumps(concepts, ensure_ascii=True, indent=2)}

Draft cheatsheet:
{cheatsheet_markdown}
""".strip()
    return _call_openai(system_prompt, user_prompt, max_output_tokens=2200, step_name="audit")


def _summarize_chunk_with_openai(
    chunk: str,
    options: GenerationOptions,
    chunk_index: int,
    chunk_total: int,
) -> LLMCallResult:
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

    return _call_openai(system_prompt, user_prompt, max_output_tokens=1400, step_name="extraction")


def _generate_cheatsheet_with_openai(
    chunk_summaries: list[str], options: GenerationOptions
) -> LLMCallResult:
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

    return _call_openai(system_prompt, user_prompt, max_output_tokens=2600, step_name="generation")


def _call_openai(
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    step_name: str,
    tools: list[dict[str, Any]] | None = None,
    include: list[str] | None = None,
    text_format: dict[str, Any] | None = None,
    tool_choice: str | None = None,
) -> LLMCallResult:
    client = OpenAI(api_key=get_openai_api_key())
    last_error: Exception | None = None

    for model_name in _candidate_model_names():
        try:
            response = _call_responses_api(
                client=client,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=max_output_tokens,
                tools=tools,
                include=include,
                text_format=text_format,
                tool_choice=tool_choice,
            )
            return _build_llm_call_result(response, model_name, text_format, step_name)
        except Exception as exc:
            last_error = exc
            _record_pipeline_error(
                step_name,
                exc,
                {
                    "api_mode": "responses",
                    "model": model_name,
                    "max_output_tokens": max_output_tokens,
                    "tools": tools or [],
                    "include": include or [],
                    "text_format": text_format or {},
                    "tool_choice": tool_choice or "",
                },
            )

        if tools:
            continue

        try:
            response = _call_chat_completions_api(
                client=client,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=max_output_tokens,
                text_format=text_format,
            )
            return _build_llm_call_result(response, model_name, text_format, step_name)
        except Exception as exc:
            last_error = exc
            _record_pipeline_error(
                step_name,
                exc,
                {
                    "api_mode": "chat_completions",
                    "model": model_name,
                    "max_output_tokens": max_output_tokens,
                    "text_format": text_format or {},
                },
            )

    raise last_error or RuntimeError("OpenAI call failed without an exception.")


def _call_responses_api(
    client: Any,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    tools: list[dict[str, Any]] | None,
    include: list[str] | None,
    text_format: dict[str, Any] | None,
    tool_choice: str | None,
) -> Any:
    request_kwargs: dict[str, Any] = {
        "model": model_name,
        "max_output_tokens": max_output_tokens,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ],
    }
    if tools:
        request_kwargs["tools"] = tools
    if include:
        request_kwargs["include"] = include
    if text_format:
        request_kwargs["text"] = {"format": text_format}
    if tool_choice:
        request_kwargs["tool_choice"] = tool_choice
    return client.responses.create(**request_kwargs)


def _call_chat_completions_api(
    client: Any,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    text_format: dict[str, Any] | None,
) -> Any:
    request_kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_completion_tokens": max_output_tokens,
    }
    if text_format and text_format.get("type") == "json_object":
        request_kwargs["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**request_kwargs)


def _build_llm_call_result(
    response: Any,
    model_name: str,
    text_format: dict[str, Any] | None,
    step_name: str,
) -> LLMCallResult:
    usage = extract_usage(response)
    raw_usage = _serialize_debug_object(_read_response_field(response, "usage"))
    response_keys = _extract_response_keys(response)
    usage_keys = list(raw_usage.keys()) if isinstance(raw_usage, dict) else []
    output_text = _extract_output_text(response)
    sources = _extract_web_sources(response)
    parsed_json = _parse_json_output(output_text) if text_format and text_format.get("type") == "json_object" else None

    result = LLMCallResult(
        text=output_text.strip(),
        usage=usage,
        model=model_name,
        raw_usage=raw_usage,
        parsed_json=parsed_json,
        sources=sources,
        response_keys=response_keys,
        usage_keys=usage_keys,
        response_type=type(response).__name__,
    )
    _record_token_usage(step_name, result)
    return result


def _audit_cheatsheet_with_openai(
    cheatsheet_markdown: str,
    chunk_summaries: list[str],
    options: GenerationOptions,
) -> LLMCallResult:
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

    return _call_openai(system_prompt, user_prompt, max_output_tokens=2200, step_name="audit")


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


def _candidate_model_names() -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for model_name in (get_openai_model(), DEFAULT_OPENAI_MODEL, *FALLBACK_OPENAI_MODELS):
        name = str(model_name or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def _read_response_field(obj: Any, name: str):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _serialize_debug_object(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(key): _serialize_debug_object(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_debug_object(value) for value in obj]

    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return _serialize_debug_object(model_dump())
        except Exception:
            pass

    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return _serialize_debug_object(to_dict())
        except Exception:
            pass

    return str(obj)


def _extract_response_keys(response: Any) -> list[str]:
    if response is None:
        return []
    if isinstance(response, dict):
        return sorted(str(key) for key in response.keys())

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            if isinstance(dumped, dict):
                return sorted(str(key) for key in dumped.keys())
        except Exception:
            pass

    keys: list[str] = []
    for name in dir(response):
        if name.startswith("_"):
            continue
        try:
            value = getattr(response, name)
        except Exception:
            continue
        if callable(value):
            continue
        keys.append(str(name))
    return sorted(set(keys))


def _extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", "") or _read_response_field(response, "output_text") or ""
    if output_text:
        return str(output_text)

    choices = _read_response_field(response, "choices") or []
    if choices:
        message = _read_response_field(choices[0], "message")
        content = _read_response_field(message, "content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = _read_response_field(item, "text") or _read_response_field(item, "content")
                    if text:
                        parts.append(str(text))
                elif isinstance(item, str):
                    parts.append(item)
            if parts:
                return "\n".join(parts)

    parts: list[str] = []
    for item in _read_response_field(response, "output") or []:
        for content in _read_response_field(item, "content") or []:
            text = _read_response_field(content, "text")
            if text:
                parts.append(str(text))
    return "\n".join(parts)


def _parse_json_output(text: str) -> Any:
    candidate = text.strip()
    if not candidate:
        return None

    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)

    try:
        return json.loads(candidate)
    except Exception:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1])
        except Exception:
            return None
    return None


def _extract_web_sources(response: Any) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []

    for item in _read_response_field(response, "output") or []:
        item_type = _read_response_field(item, "type")
        if item_type == "web_search_call":
            action = _read_response_field(item, "action")
            for source in _read_response_field(action, "sources") or []:
                entry = {
                    "title": str(_read_response_field(source, "title") or ""),
                    "url": str(_read_response_field(source, "url") or ""),
                    "type": "web_source",
                }
                if entry["title"] or entry["url"]:
                    sources.append(entry)

        if item_type == "message":
            for content in _read_response_field(item, "content") or []:
                for annotation in _read_response_field(content, "annotations") or []:
                    if _read_response_field(annotation, "type") != "url_citation":
                        continue
                    entry = {
                        "title": str(_read_response_field(annotation, "title") or ""),
                        "url": str(_read_response_field(annotation, "url") or ""),
                        "type": "url_citation",
                    }
                    if entry["title"] or entry["url"]:
                        sources.append(entry)

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        key = (source.get("title", "").strip().lower(), source.get("url", "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def _usage_stats_dict() -> dict[str, int]:
    return asdict(UsageStats())


def _empty_token_usage_payload(model_name: str = "") -> dict[str, Any]:
    return {
        "model": model_name,
        "available": False,
        "extraction": _usage_stats_dict(),
        "cleaning": _usage_stats_dict(),
        "web_clarification": _usage_stats_dict(),
        "generation": _usage_stats_dict(),
        "audit": _usage_stats_dict(),
        "total": _usage_stats_dict(),
        "debug": {
            "extraction": [],
            "cleaning": [],
            "web_clarification": [],
            "generation": [],
            "audit": [],
        },
    }


def _record_pipeline_error(
    step_name: str,
    exc: Exception,
    context: dict[str, Any] | None = None,
) -> None:
    if st is None:
        return

    try:
        payload = st.session_state.get("token_usage")
    except Exception:
        return

    if not isinstance(payload, dict):
        payload = _empty_token_usage_payload(get_openai_model() if is_openai_configured() else "")

    debug_payload = payload.setdefault(
        "debug",
        {
            "extraction": [],
            "cleaning": [],
            "web_clarification": [],
            "generation": [],
            "audit": [],
        },
    )
    step_debug = debug_payload.setdefault(step_name, [])
    step_debug.append(
        {
            "event_type": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "context": _serialize_debug_object(context),
        }
    )
    st.session_state["token_usage"] = payload

    try:
        pipeline_errors = st.session_state.get("pipeline_errors")
    except Exception:
        return

    if not isinstance(pipeline_errors, dict):
        pipeline_errors = {}
    step_errors = pipeline_errors.setdefault(step_name, [])
    if len(step_errors) < 12:
        step_errors.append(
            {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "context": _serialize_debug_object(context),
            }
        )
    st.session_state["pipeline_errors"] = pipeline_errors


def _record_token_usage(step_name: str, result: LLMCallResult) -> None:
    if st is None:
        return

    try:
        payload = st.session_state.get("token_usage")
    except Exception:
        return

    if not isinstance(payload, dict):
        payload = _empty_token_usage_payload(result.model)

    if not payload.get("model"):
        payload["model"] = result.model

    step_stats = asdict(result.to_usage_stats())
    for bucket_name in (step_name, "total"):
        bucket = payload.setdefault(bucket_name, _usage_stats_dict())
        for key, value in step_stats.items():
            bucket[key] = int(bucket.get(key, 0)) + int(value)

    payload["available"] = bool(payload.get("available")) or bool(result.usage.get("available", False))

    debug_payload = payload.setdefault(
        "debug",
        {
            "extraction": [],
            "cleaning": [],
            "web_clarification": [],
            "generation": [],
            "audit": [],
        },
    )
    step_debug = debug_payload.setdefault(step_name, [])
    step_debug.append(
        {
            "event_type": "usage",
            "model": result.model,
            "response_type": result.response_type,
            "response_keys": result.response_keys,
            "usage_keys": result.usage_keys,
            "parsed_usage": dict(result.usage),
            "raw_usage": result.raw_usage,
        }
    )

    st.session_state["token_usage"] = payload


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


def _extract_concepts_heuristic(source_text: str, options: GenerationOptions) -> list[dict[str, Any]]:
    candidates = _collect_candidates(source_text)
    concept_pool = _dedupe_lines(
        candidates["definitions"]
        + candidates["concepts"]
        + candidates["methods"]
        + candidates["formulas"]
        + candidates["comparisons"]
        + candidates["examples"]
        + candidates["exam"]
    )
    if not concept_pool:
        concept_pool = _fallback_concept_candidates(source_text)
    limit = _concept_limit(options)
    records: list[dict[str, Any]] = []

    for item in concept_pool:
        cleaned = _compact_line(item, 28)
        if (
            not cleaned
            or _looks_like_generic_filler(cleaned)
            or _looks_incomplete(cleaned)
            or _looks_like_nonconcept_noise(cleaned)
        ):
            continue

        concept_name = _infer_concept_name(cleaned)
        if not concept_name:
            continue
        category = _infer_concept_category(cleaned)
        record = ConceptRecord(
            concept=concept_name,
            category=category,
            slide_context=cleaned,
            appears_in_slides=True,
            importance=_heuristic_importance(cleaned, category),
            needs_web_clarification=False,
            reason_for_clarification="",
            final_explanation=cleaned,
            definition_from_slides=cleaned if _looks_like_definition(cleaned) else "",
            why_it_matters="",
            formula_or_measure=cleaned if category in {"formula", "measure"} else "",
            distinction=cleaned if category == "distinction" else "",
            example_or_finding=cleaned if category == "example" else "",
            exam_trap=cleaned if category == "exam_trap" else "",
            final_definition=cleaned,
            kind=category,
        )
        records.append(asdict(record))
        if len(records) >= limit:
            break

    return records


def _clean_concepts_heuristic(
    concept_inventory: list[dict[str, Any]],
    options: GenerationOptions,
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for record in _finalize_concept_records(concept_inventory):
        importance = _safe_text(record.get("importance")).lower() or _heuristic_importance(
            _safe_text(record.get("slide_context")),
            _safe_text(record.get("category")) or _safe_text(record.get("kind")),
        )
        if importance == "low":
            continue
        cleaned.append(record)

    return _prioritize_concepts(cleaned, options)


def _generate_concept_cheatsheet_heuristic(
    concepts: list[dict[str, Any]],
    options: GenerationOptions,
    source_text: str,
) -> str:
    concepts = _finalize_concept_records(concepts)
    labels = _section_labels(options.output_language)
    title = _resolve_display_title(source_text, options)
    core_lines: list[str] = []
    formula_lines: list[str] = []
    distinction_rows: list[str] = []
    example_lines: list[str] = []
    exam_lines: list[str] = []

    for concept in concepts:
        name = _safe_text(concept.get("concept"))
        category = (_safe_text(concept.get("category")) or _safe_text(concept.get("kind"))).lower()
        definition = (
            _safe_text(concept.get("final_explanation"))
            or _safe_text(concept.get("final_definition"))
            or _safe_text(concept.get("definition_from_slides"))
        )
        why = _safe_text(concept.get("why_it_matters"))
        formula = _safe_text(concept.get("formula_or_measure"))
        distinction = _safe_text(concept.get("distinction"))
        example = _safe_text(concept.get("example_or_finding"))
        trap = _safe_text(concept.get("exam_trap"))

        core_entry = _build_teaching_bullet(name, definition, why, "")
        if core_entry and category not in {"exam_trap", "example"}:
            core_lines.append(core_entry)

        if formula or category in {"formula", "measure"}:
            formula_lines.append(_build_measure_bullet(name, formula, definition, trap))

        if distinction or category == "distinction":
            left, right, diff = _distinction_row_from_text(name, distinction)
            distinction_rows.append(f"| {left} | {right} | {diff} |")

        if example or category == "example":
            example_lines.append(_build_supporting_bullet(name, example))

        if trap or category == "exam_trap":
            exam_lines.append(_build_supporting_bullet(name, trap))

    if not any((core_lines, formula_lines, distinction_rows, example_lines, exam_lines)):
        fallback_lines = _fallback_concept_candidates(
            source_text
            or "\n".join(_safe_text(concept.get("slide_context")) for concept in concepts),
            max_items=max(12, _section_caps(options)["concepts"] + _section_caps(options)["formulas"]),
        )
        for line in fallback_lines[: _section_caps(options)["concepts"]]:
            core_lines.append(line)

    lines: list[str] = [f"# {title} - A4 Cheatsheet", ""]
    lines.extend(_section_block(labels["concepts"], core_lines, _section_caps(options)["concepts"]))
    lines.extend(_section_block(labels["formulas"], formula_lines, _section_caps(options)["formulas"]))
    if distinction_rows:
        lines.append(f"## {labels['distinctions']}")
        lines.append("| Concept A | Concept B | Difference |")
        lines.append("| --- | --- | --- |")
        lines.extend(distinction_rows[: _section_caps(options)["comparisons"]])
        lines.append("")
    lines.extend(_section_block(labels["examples"], example_lines, _section_caps(options)["examples"]))
    lines.extend(_section_block(labels["exam"], exam_lines, _section_caps(options)["exam"]))
    return "\n".join(lines).strip()


def _normalize_concept_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        concept = _safe_text(record.get("concept"))
        slide_context = _safe_text(record.get("slide_context"))
        if (
            not concept
            or _looks_like_generic_filler(concept)
            or _looks_like_incomplete_concept_name(concept)
            or _looks_like_nonconcept_noise(concept)
            or (slide_context and _looks_like_nonconcept_noise(slide_context))
        ):
            continue

        normalized.append(
            asdict(
                ConceptRecord(
                    concept=concept,
                    category=_normalize_category(_safe_text(record.get("category")) or _safe_text(record.get("kind"))),
                    slide_context=slide_context,
                    appears_in_slides=bool(record.get("appears_in_slides", True)),
                    importance=_normalize_importance(_safe_text(record.get("importance"))),
                    needs_web_clarification=bool(record.get("needs_web_clarification")),
                    reason_for_clarification=_safe_text(record.get("reason_for_clarification")),
                    final_explanation=_safe_text(record.get("final_explanation")),
                    definition_from_slides=_safe_text(record.get("definition_from_slides")),
                    why_it_matters=_safe_text(record.get("why_it_matters")),
                    formula_or_measure=_safe_text(record.get("formula_or_measure")),
                    distinction=_safe_text(record.get("distinction")),
                    example_or_finding=_safe_text(record.get("example_or_finding")),
                    exam_trap=_safe_text(record.get("exam_trap")),
                    web_definition=_safe_text(record.get("web_definition")),
                    final_definition=_safe_text(record.get("final_definition")),
                    web_source_title=_safe_text(record.get("web_source_title")),
                    web_source_url=_safe_text(record.get("web_source_url")),
                    sources=[
                        {
                            "title": _safe_text(source.get("title")),
                            "url": _safe_text(source.get("url")),
                            "type": _safe_text(source.get("type")) or "web_source",
                        }
                        for source in (record.get("sources") or [])
                        if isinstance(source, dict) and (_safe_text(source.get("title")) or _safe_text(source.get("url")))
                    ],
                    kind=_normalize_category(_safe_text(record.get("category")) or _safe_text(record.get("kind"))),
                )
            )
        )
    return normalized


def _finalize_concept_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_record in _normalize_concept_records(records):
        record = _ensure_final_definition(dict(raw_record))
        if not bool(record.get("appears_in_slides", True)):
            continue
        key = record["concept"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _prioritize_concepts(records: list[dict[str, Any]], options: GenerationOptions) -> list[dict[str, Any]]:
    high = [record for record in records if _safe_text(record.get("importance")).lower() == "high"]
    medium = [record for record in records if _safe_text(record.get("importance")).lower() == "medium"]
    prioritized = high + medium
    return prioritized[: _concept_limit(options)]


def _ensure_final_definition(record: dict[str, Any]) -> dict[str, Any]:
    final_definition = _safe_text(record.get("final_definition"))
    if not final_definition:
        final_definition = (
            _safe_text(record.get("definition_from_slides"))
            or _safe_text(record.get("web_definition"))
            or _safe_text(record.get("slide_context"))
        )
    record["final_definition"] = final_definition
    if not _safe_text(record.get("final_explanation")):
        explanation = (
            final_definition
            or _safe_text(record.get("formula_or_measure"))
            or _safe_text(record.get("distinction"))
            or _safe_text(record.get("example_or_finding"))
            or _safe_text(record.get("exam_trap"))
        )
        record["final_explanation"] = explanation
    if not _safe_text(record.get("importance")):
        record["importance"] = _heuristic_importance(
            _safe_text(record.get("slide_context")),
            _safe_text(record.get("category")) or _safe_text(record.get("kind")),
        )
    if not _safe_text(record.get("category")):
        record["category"] = _normalize_category(_safe_text(record.get("kind")))
    if not _safe_text(record.get("kind")):
        record["kind"] = _safe_text(record.get("category"))
    return record


def _concept_limit(options: GenerationOptions) -> int:
    base = {
        "1-page A4": 16,
        "2-page A4": 24,
        "concise summary": 12,
        "detailed summary": 25,
    }.get(options.target_length, 16)
    modifier = {
        "More concise": -2,
        "Balanced": 0,
        "More detailed": 2,
    }.get(options.density, 0)
    return max(10, min(25, base + modifier))


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _infer_concept_name(text: str) -> str:
    if ":" in text:
        head = text.split(":", 1)[0].strip()
        if 1 <= len(head.split()) <= 8:
            return head
    for separator in (" is ", " are ", " refers to ", " means ", " = "):
        if separator in text.lower():
            pattern = re.compile(re.escape(separator), re.IGNORECASE)
            head = pattern.split(text, maxsplit=1)[0].strip()
            if 1 <= len(head.split()) <= 6 and not _looks_like_nonconcept_noise(head):
                return head
    words = text.split()
    if len(words) <= 6 and not _looks_like_nonconcept_noise(text):
        return text.strip()
    return ""


def _infer_concept_category(text: str) -> str:
    if _looks_like_exam_signal(text):
        return "exam_trap"
    if _looks_like_example(text):
        return "example"
    if _looks_like_comparison(text):
        return "distinction"
    if _looks_like_method(text):
        return "method"
    if _looks_like_formula(text):
        return "formula"
    if _looks_like_definition(text):
        return "definition"
    return "theory"


def _infer_concept_kind(text: str) -> str:
    return _infer_concept_category(text)


def _normalize_category(value: str) -> str:
    lowered = value.strip().lower()
    allowed = {"definition", "measure", "formula", "theory", "method", "distinction", "example", "exam_trap"}
    if lowered in allowed:
        return lowered
    aliases = {
        "term": "definition",
        "concept": "definition",
        "model": "theory",
        "study": "example",
        "finding": "example",
        "comparison": "distinction",
        "trap": "exam_trap",
    }
    return aliases.get(lowered, "definition")


def _normalize_importance(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"high", "medium", "low"}:
        return lowered
    return "medium"


def _heuristic_importance(text: str, category: str) -> str:
    category = _normalize_category(category)
    lowered = text.lower()
    if category in {"formula", "measure", "theory", "distinction", "exam_trap"}:
        return "high"
    if any(token in lowered for token in ("important", "remember", "exam", "trap", "formula", "theory", "model")):
        return "high"
    if category in {"definition", "method"}:
        return "medium"
    if category == "example":
        return "medium"
    return "low"


def _build_teaching_bullet(name: str, definition: str, why: str, trap: str) -> str:
    definition = _strip_repeated_label(name, definition)
    why = _strip_repeated_label(name, why)
    trap = _strip_repeated_label(name, trap)
    parts = []
    if name and definition:
        parts.append(f"{name}: {definition}")
    elif definition:
        parts.append(definition)
    if why:
        parts.append(f"Why it matters / how to identify it: {why}")
    if trap:
        parts.append(f"Exam trap: {trap}")
    return " ".join(parts).strip()


def _build_measure_bullet(name: str, formula: str, meaning: str, trap: str) -> str:
    formula = _strip_repeated_label(name, formula)
    meaning = _strip_repeated_label(name, meaning)
    trap = _strip_repeated_label(name, trap)
    parts = []
    if name:
        parts.append(f"{name}:")
    if formula:
        parts.append(formula)
    if meaning:
        parts.append(f"What it means: {meaning}")
    if trap:
        parts.append(f"Exam trap: {trap}")
    return " ".join(parts).strip()


def _build_supporting_bullet(name: str, detail: str) -> str:
    detail = _strip_repeated_label(name, detail)
    if name:
        return f"{name}: {detail}".strip()
    return detail.strip()


def _strip_repeated_label(name: str, text: str) -> str:
    if not name or not text:
        return text.strip()
    pattern = re.compile(rf"^{re.escape(name)}\s*:\s*", re.IGNORECASE)
    return pattern.sub("", text.strip())


def _distinction_row_from_text(name: str, distinction: str) -> tuple[str, str, str]:
    for separator in (" vs ", " versus ", " unlike ", " compared with "):
        if separator in distinction.lower():
            pattern = re.compile(separator, re.IGNORECASE)
            parts = pattern.split(distinction, maxsplit=1)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip(), _safe_text(name) or distinction
    return name or "Concept A", "Concept B", distinction


def _collect_candidates(text: str) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = defaultdict(list)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        plain = _strip_markdown(line)
        if not plain or plain.lower().startswith("source:"):
            continue
        if _looks_like_nonconcept_noise(plain):
            continue

        if line.startswith("#"):
            categories["headings"].append(plain)
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
        if compacted and not _looks_like_generic_filler(compacted) and not _looks_like_nonconcept_noise(compacted)
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
        if not candidate or candidate.lower().startswith("source:"):
            continue
        if _looks_like_formula(candidate) or _looks_like_definition(candidate):
            continue
        if candidate.endswith((".", "!", "?")):
            continue
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
    if re.match(r"^[A-Za-z][A-Za-z0-9\s-]{0,50}\s+(is|are)\s+", text) and len(text.split()[:6]) >= 2:
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


def _looks_like_incomplete_concept_name(text: str) -> bool:
    if not text.strip():
        return True
    if text.endswith((":","/","-","(","[","{")):
        return True
    if text.count("(") != text.count(")"):
        return True
    if text.count("[") != text.count("]"):
        return True
    return False


def _looks_like_nonconcept_noise(text: str) -> bool:
    lowered = text.lower().strip()
    if not lowered:
        return False

    if "http://" in lowered or "https://" in lowered or "www." in lowered:
        return True
    if re.search(r"\(\d{4}\)\s*[:.,]\s*\d", text):
        return True
    if re.search(r"\b(?:doi|issn|isbn)\b", lowered):
        return True
    if lowered.startswith(("repository", "references", "bibliography")):
        return True
    if re.search(r"\b(?:bootstrap|reg|areg|xtreg|logit|probit|summarize|tabulate|egen|gen)\b", lowered) and ":" in lowered:
        return True
    if re.search(r"\b(?:prob>=|prob>|chibar2|lr test|likelihood-ratio test|std\.?\s*err\.?|coef\.?)\b", lowered):
        return True

    tokens = re.findall(r"\b[\w.-]+\b", text)
    if not tokens:
        return False

    digit_tokens = sum(any(character.isdigit() for character in token) for token in tokens)
    numeric_ratio = digit_tokens / len(tokens)
    if numeric_ratio >= 0.45 and not any(
        keyword in lowered for keyword in ("density", "degree", "centrality", "formula", "measure", "probability")
    ):
        return True

    return False


def _group_chunks_for_concept_extraction(chunks: list[str], max_batches: int = 12) -> list[str]:
    if len(chunks) <= max_batches:
        return chunks

    group_size = max(1, math.ceil(len(chunks) / max_batches))
    return [
        "\n\n".join(chunks[index : index + group_size]).strip()
        for index in range(0, len(chunks), group_size)
        if "\n\n".join(chunks[index : index + group_size]).strip()
    ]


def _fallback_concept_candidates(source_text: str, max_items: int = 36) -> list[str]:
    candidates: list[str] = []
    for raw_line in source_text.splitlines():
        plain = _strip_markdown(raw_line)
        if not plain or plain.lower().startswith("source:"):
            continue
        if _looks_like_generic_filler(plain) or _looks_incomplete(plain) or _looks_like_nonconcept_noise(plain):
            continue
        word_count = len(plain.split())
        if 3 <= word_count <= 28:
            candidates.append(plain)
        if len(candidates) >= max_items:
            break
    return _dedupe_lines(candidates)


def _has_substantive_cheatsheet(markdown: str) -> bool:
    if not markdown or not markdown.strip():
        return False

    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    if len(lines) <= 1:
        return False

    bullet_or_table_lines = [
        line
        for line in lines
        if line.startswith("- ")
        or line.startswith("| ")
        or re.match(r"^\|\s*[^|]+\|", line)
    ]
    return len(bullet_or_table_lines) >= 2

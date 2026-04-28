"""Streamlit app for building compact, exam-oriented cheat sheets from course files."""

from __future__ import annotations

import re
from dataclasses import asdict

import streamlit as st

from cheatsheet_ai import generator as generator_module
from cheatsheet_ai.extractors import (
    extract_text_from_docx,
    extract_text_from_pdf,
    extract_text_from_pptx,
    extract_text_from_txt,
)
from cheatsheet_ai.exporters import export_to_docx, export_to_markdown, export_to_pdf
from cheatsheet_ai.processing import chunk_text, clean_extracted_text


GenerationOptions = generator_module.GenerationOptions
UsageStats = generator_module.UsageStats
get_openai_model = generator_module.get_openai_model
is_openai_configured = generator_module.is_openai_configured


SUPPORTED_FILE_TYPES = ["pdf", "pptx", "docx", "txt"]

MODEL_PRICING_PER_MILLION = {
    "gpt-5.2": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
}


st.set_page_config(
    page_title="Cheatsheet AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stTextArea textarea {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.9rem;
    }
    .preview-card {
        border: 1px solid #d9d9d9;
        border-radius: 10px;
        padding: 1rem;
        background: #fafafa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    st.title("Cheatsheet AI Assistant")
    st.caption("Upload messy course materials and turn them into a condensed, exam-ready A4 cheat sheet.")

    with st.sidebar:
        st.header("Controls")
        course_name = st.text_input("Course / topic name", placeholder="e.g. Probability Theory Midterm")
        output_language = st.selectbox("Output language", ["English", "Chinese", "Bilingual"], index=0)
        target_length = st.selectbox(
            "Cheat sheet length",
            ["1-page A4", "2-page A4", "concise summary", "detailed summary"],
            index=0,
        )
        focus_style = st.selectbox(
            "Focus style",
            ["exam-focused", "formula-focused", "concept-focused", "mixed"],
            index=0,
        )
        include_examples = st.checkbox("Include examples", value=True)
        include_formulas = st.checkbox("Include formulas", value=True)
        use_web_search = st.checkbox(
            "Use web search to clarify concepts",
            value=False,
            help="Uses the web only to clarify concepts that already appear in the uploaded slides.",
        )
        density = st.radio("Detail level", ["More concise", "Balanced", "More detailed"], index=1)

        if is_openai_configured():
            st.success(f"OpenAI mode enabled ({get_openai_model()})")
        else:
            st.warning("OpenAI key not found. Running in heuristic prototype mode.")
            st.caption("Set OPENAI_API_KEY in your shell or .streamlit/secrets.toml to enable OpenAI mode.")
            if output_language != "English":
                st.caption("Non-English output works best when an OpenAI API key is available.")
            if use_web_search:
                st.caption("Web clarification requires OpenAI mode and will be skipped in heuristic mode.")

    uploaded_files = st.file_uploader(
        "Upload lecture slides, notes, PDFs, or documents",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
        help="Supported formats: PDF, PPTX, DOCX, TXT",
    )

    if uploaded_files:
        st.write("Uploaded files:")
        for file in uploaded_files:
            st.write(f"- {file.name}")
    else:
        st.info("Upload one or more files to start building a cheat sheet.")

    action_columns = st.columns(5)
    generate_clicked = action_columns[0].button("Generate Cheat Sheet", use_container_width=True, type="primary")
    regenerate_clicked = action_columns[1].button("Regenerate", use_container_width=True)
    concise_clicked = action_columns[2].button("Make More Concise", use_container_width=True)
    detail_clicked = action_columns[3].button("Add More Detail", use_container_width=True)
    bilingual_clicked = action_columns[4].button("Convert to Bilingual", use_container_width=True)

    options = GenerationOptions(
        course_name=course_name.strip(),
        output_language=output_language,
        target_length=target_length,
        focus_style=focus_style,
        include_examples=include_examples,
        include_formulas=include_formulas,
        include_exam_questions=False,
        density=density,
        use_web_search=use_web_search,
        variant=st.session_state.get("generation_variant", 0),
    )

    if generate_clicked:
        if not uploaded_files:
            st.error("Please upload at least one file before generating.")
        else:
            _process_uploads_and_generate(uploaded_files, options)

    if regenerate_clicked and _has_cleaned_text():
        options.variant = st.session_state.get("generation_variant", 0) + 1
        st.session_state["generation_variant"] = options.variant
        _generate_from_existing_text(options)
    elif regenerate_clicked:
        st.error("Generate a cheat sheet first so there is source text to regenerate from.")

    if concise_clicked and _has_cleaned_text():
        options.density = "More concise"
        _generate_from_existing_text(options)
    elif concise_clicked:
        st.error("Generate a cheat sheet first so there is source text to compress.")

    if detail_clicked and _has_cleaned_text():
        options.density = "More detailed"
        _generate_from_existing_text(options)
    elif detail_clicked:
        st.error("Generate a cheat sheet first so there is source text to expand.")

    if bilingual_clicked and _has_cleaned_text():
        options.output_language = "Bilingual"
        _generate_from_existing_text(options)
    elif bilingual_clicked:
        st.error("Generate a cheat sheet first so there is source text to convert.")

    if "generated_markdown" in st.session_state:
        display_result()


def _process_uploads_and_generate(uploaded_files, options: GenerationOptions) -> None:
    with st.spinner("Extracting text, cleaning materials, and generating the cheat sheet..."):
        extracted_by_file, combined_text = parse_slides(uploaded_files)
        if not extracted_by_file:
            st.error("No uploaded files could be parsed. Please try a different file set.")
            return
        _store_extraction_state(extracted_by_file, combined_text)
        _run_generation_pipeline(options)


def _generate_from_existing_text(options: GenerationOptions) -> None:
    with st.spinner("Generating a refreshed cheat sheet..."):
        _run_generation_pipeline(options)


def _run_generation_pipeline(options: GenerationOptions) -> None:
    cleaned_text = st.session_state.get("cleaned_text", "")
    chunks = chunk_text(cleaned_text)
    st.session_state["pipeline_errors"] = {}
    st.session_state["token_usage"] = _empty_token_usage_state(get_openai_model() if is_openai_configured() else "")
    (
        concept_inventory,
        concepts,
        cheatsheet_markdown,
        extraction_usage,
        cleaning_usage,
        web_usage,
        generation_usage,
        audit_usage,
    ) = _generate_with_best_available_pipeline(chunks, cleaned_text, options)

    total_usage = UsageStats()
    total_usage.add(extraction_usage)
    total_usage.add(cleaning_usage)
    total_usage.add(web_usage)
    total_usage.add(generation_usage)
    total_usage.add(audit_usage)
    configured_model = get_openai_model() if is_openai_configured() else ""
    token_usage = st.session_state.get("token_usage", _empty_token_usage_state(configured_model))
    model_name = str(token_usage.get("model") or configured_model)
    debug_info = token_usage.get("debug", _empty_token_usage_state().get("debug", {}))
    visible_error_count = _count_pipeline_errors(
        st.session_state.get("pipeline_errors", {}),
        bool(total_usage.api_calls),
    )

    if is_openai_configured() and total_usage.api_calls == 0 and visible_error_count:
        cheatsheet_markdown = _generation_failed_markdown(
            _first_pipeline_error_message(st.session_state.get("pipeline_errors", {}), False)
        )

    st.session_state["chunk_count"] = len(chunks)
    st.session_state["concept_inventory"] = concept_inventory
    st.session_state["chunk_summaries"] = [concept.get("concept", "") for concept in concepts]
    st.session_state["concept_records"] = concepts
    st.session_state["generated_markdown"] = cheatsheet_markdown
    st.session_state["editable_cheatsheet"] = cheatsheet_markdown
    st.session_state["last_options"] = asdict(options)
    st.session_state["generation_variant"] = options.variant
    st.session_state["sources_used"] = _build_sources_used(concepts)
    st.session_state["token_usage"] = {
        "model": model_name,
        "available": total_usage.usage_available_calls > 0,
        "extraction": asdict(extraction_usage),
        "cleaning": asdict(cleaning_usage),
        "web_clarification": asdict(web_usage),
        "generation": asdict(generation_usage),
        "audit": asdict(audit_usage),
        "total": asdict(total_usage),
        "estimated_cost_usd": _estimate_cost_usd(model_name, total_usage),
        "pricing_configured": _get_pricing_for_model(model_name) is not None,
        "debug": debug_info,
    }


def _generate_with_best_available_pipeline(
    chunks: list[str],
    cleaned_text: str,
    options: GenerationOptions,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    str,
    UsageStats,
    UsageStats,
    UsageStats,
    UsageStats,
    UsageStats,
]:
    extract_concepts_fn = getattr(generator_module, "extract_concepts", None)
    clean_concepts_fn = getattr(generator_module, "clean_concepts", None)
    clarify_concepts_fn = getattr(generator_module, "clarify_concepts", None)
    clarify_with_web_fn = getattr(generator_module, "clarify_concepts_with_web", None)
    generate_from_concepts_fn = getattr(generator_module, "generate_cheatsheet_from_concepts", None)
    audit_from_concepts_fn = getattr(generator_module, "audit_cheatsheet_from_concepts", None)

    if all(
        callable(fn)
        for fn in (
            extract_concepts_fn,
            clean_concepts_fn,
            generate_from_concepts_fn,
            audit_from_concepts_fn,
        )
    ):
        concept_inventory, extraction_usage = extract_concepts_fn(chunks, options, source_text=cleaned_text)
        concepts, cleaning_usage = clean_concepts_fn(concept_inventory, options, source_text=cleaned_text)

        if callable(clarify_concepts_fn):
            concepts, web_usage = clarify_concepts_fn(concepts, options)
        elif callable(clarify_with_web_fn):
            concepts, web_usage = clarify_with_web_fn(concepts, options)
        else:
            web_usage = UsageStats()

        draft_markdown, generation_usage = generate_from_concepts_fn(concepts, options, source_text=cleaned_text)
        cheatsheet_markdown, audit_usage = audit_from_concepts_fn(draft_markdown, concepts, options)
        return (
            concept_inventory,
            concepts,
            cheatsheet_markdown,
            extraction_usage,
            cleaning_usage,
            web_usage,
            generation_usage,
            audit_usage,
        )

    summarize_chunks_fn = getattr(generator_module, "summarize_chunks")
    generate_cheatsheet_fn = getattr(generator_module, "generate_cheatsheet")
    audit_cheatsheet_fn = getattr(generator_module, "audit_cheatsheet")
    summaries, extraction_usage = summarize_chunks_fn(chunks, options)
    cheatsheet_draft, generation_usage = generate_cheatsheet_fn(summaries, options, source_text=cleaned_text)
    cheatsheet_markdown, audit_usage = audit_cheatsheet_fn(cheatsheet_draft, summaries, options)
    concept_inventory = _concept_inventory_from_summaries(summaries)
    concepts = list(concept_inventory)
    return (
        concept_inventory,
        concepts,
        cheatsheet_markdown,
        extraction_usage,
        UsageStats(),
        UsageStats(),
        generation_usage,
        audit_usage,
    )


def _store_extraction_state(extracted_by_file: list[dict[str, str]], combined_text: str) -> None:
    st.session_state["extracted_by_file"] = extracted_by_file
    st.session_state["cleaned_text"] = combined_text
    st.session_state["source_word_count"] = len(combined_text.split())


def display_result() -> None:
    st.divider()
    st.subheader("Generated Cheat Sheet")

    stats_columns = st.columns(3)
    stats_columns[0].metric("Source words", st.session_state.get("source_word_count", 0))
    stats_columns[1].metric("Chunks", st.session_state.get("chunk_count", 0))
    stats_columns[2].metric("Mode", "OpenAI" if is_openai_configured() else "Heuristic")

    token_usage = st.session_state.get("token_usage", {})
    pipeline_error_count = _count_pipeline_errors(
        st.session_state.get("pipeline_errors", {}),
        bool(token_usage.get("total", {}).get("api_calls", 0)),
    )
    if pipeline_error_count:
        first_error = _first_pipeline_error_message(
            st.session_state.get("pipeline_errors", {}),
            bool(token_usage.get("total", {}).get("api_calls", 0)),
        )
        st.warning(
            f"{pipeline_error_count} OpenAI step(s) failed during this run. "
            "The app used fallbacks where possible. Expand `Token Usage` and enable debug info to inspect the errors."
        )
        if first_error:
            st.caption(f"First error: {first_error}")

    tabs = st.tabs(["Edit", "Preview", "Source Preview"])

    with tabs[0]:
        st.text_area(
            "Edit markdown before export",
            height=650,
            key="editable_cheatsheet",
        )

    with tabs[1]:
        st.markdown('<div class="preview-card">', unsafe_allow_html=True)
        st.markdown(st.session_state.get("editable_cheatsheet", ""))
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        for entry in st.session_state.get("extracted_by_file", []):
            with st.expander(entry["name"]):
                st.caption("Cleaned extraction preview")
                st.text_area(
                    f"cleaned-{entry['name']}",
                    value=entry["cleaned_text"][:12000],
                    height=260,
                    disabled=True,
                )

    _render_token_usage()
    _render_sources_used()

    st.subheader("Export")
    export_name = _slugify_filename(
        st.session_state.get("last_options", {}).get("course_name") or "cheatsheet-ai-output"
    )
    current_markdown = st.session_state.get("editable_cheatsheet", "")

    download_columns = st.columns(3)
    download_columns[0].download_button(
        "Download Markdown",
        data=export_to_markdown(current_markdown),
        file_name=f"{export_name}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    try:
        pdf_bytes = export_to_pdf(current_markdown, title=export_name)
        download_columns[1].download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=f"{export_name}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as exc:
        download_columns[1].warning(f"PDF export unavailable: {exc}")

    try:
        docx_bytes = export_to_docx(current_markdown, title=export_name)
        download_columns[2].download_button(
            "Download DOCX",
            data=docx_bytes,
            file_name=f"{export_name}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
    except Exception as exc:
        download_columns[2].warning(f"DOCX export unavailable: {exc}")


def _extract_text(file_name: str, file_bytes: bytes) -> str:
    extension = file_name.rsplit(".", 1)[-1].lower()

    if extension == "pdf":
        return extract_text_from_pdf(file_bytes)
    if extension == "pptx":
        return extract_text_from_pptx(file_bytes)
    if extension == "docx":
        return extract_text_from_docx(file_bytes)
    if extension == "txt":
        return extract_text_from_txt(file_bytes)

    raise ValueError(f"Unsupported file type: {extension}")


def parse_slides(uploaded_files) -> tuple[list[dict[str, str]], str]:
    extracted_by_file: list[dict[str, str]] = []
    cleaned_sections: list[str] = []

    for uploaded_file in uploaded_files:
        try:
            raw_bytes = uploaded_file.getvalue()
            extracted_text = _extract_text(uploaded_file.name, raw_bytes)
            cleaned_text = _focus_slide_text(clean_extracted_text(extracted_text))
        except Exception as exc:
            st.warning(f"Skipping {uploaded_file.name}: {exc}")
            continue

        extracted_by_file.append(
            {
                "name": uploaded_file.name,
                "raw_text": extracted_text,
                "cleaned_text": cleaned_text,
            }
        )
        cleaned_sections.append(f"# Source: {uploaded_file.name}\n{cleaned_text}")

    combined_text = clean_extracted_text("\n\n".join(cleaned_sections)) if cleaned_sections else ""
    return extracted_by_file, combined_text


def _focus_slide_text(cleaned_text: str) -> str:
    if not cleaned_text.strip():
        return cleaned_text

    lines = cleaned_text.splitlines()
    agenda_index = -1
    for index, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped in {"agenda", "## agenda", "# agenda"}:
            agenda_index = index
            break

    if agenda_index <= 0:
        return cleaned_text

    preamble = "\n".join(lines[:agenda_index])
    question_like_preamble = preamble.count("?") >= 3 or sum(
        line.strip().lower().startswith(("does ", "is ", "are ", "can ", "would ", "if "))
        for line in lines[:agenda_index]
    ) >= 5
    if not question_like_preamble:
        return cleaned_text

    title_block: list[str] = []
    for line in lines[:agenda_index]:
        stripped = line.strip()
        if not stripped:
            if title_block:
                break
            continue
        title_block.append(line)
        if len(title_block) >= 4:
            break

    focused_lines: list[str] = []
    if title_block:
        focused_lines.extend(title_block)
        focused_lines.append("")
    focused_lines.extend(lines[agenda_index:])
    return "\n".join(focused_lines).strip()


def _generation_failed_markdown(first_error: str) -> str:
    lines = [
        "# Generation failed",
        "",
        "The app did not generate a cheatsheet because every OpenAI request failed.",
        "Open `Token Usage` and `Show raw usage debug info` to inspect the API error.",
    ]
    if first_error:
        lines.extend(["", f"First error: {first_error}"])
    return "\n".join(lines)


def _concept_inventory_from_summaries(summaries: list[str]) -> list[dict[str, object]]:
    concepts: list[dict[str, object]] = []
    seen: set[str] = set()

    for summary in summaries:
        for raw_line in summary.splitlines():
            line = raw_line.strip()
            if not line.startswith("- "):
                continue
            text = line[2:].strip()
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            concepts.append(
                {
                    "concept": text.split(":", 1)[0].strip() if ":" in text else text[:60].strip(),
                    "category": "definition",
                    "kind": "definition",
                    "slide_context": text,
                    "appears_in_slides": True,
                    "importance": "medium",
                    "needs_web_clarification": False,
                    "reason_for_clarification": "",
                    "final_explanation": text,
                    "definition_from_slides": text,
                    "why_it_matters": "",
                    "formula_or_measure": "",
                    "distinction": "",
                    "example_or_finding": "",
                    "exam_trap": "",
                    "web_definition": "",
                    "final_definition": text,
                    "web_source_title": "",
                    "web_source_url": "",
                    "sources": [],
                }
            )
    return concepts


def _has_cleaned_text() -> bool:
    return bool(st.session_state.get("cleaned_text"))


def _slugify_filename(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip()).strip("-").lower()
    return slug or "cheatsheet-ai-output"


def _render_token_usage() -> None:
    usage = st.session_state.get("token_usage", {})
    total = usage.get("total", {})
    pipeline_errors = st.session_state.get("pipeline_errors", {})
    pipeline_error_count = _count_pipeline_errors(pipeline_errors, bool(total.get("api_calls", 0)))

    if not usage:
        return

    st.subheader("Token Usage")

    with st.expander("Token Usage", expanded=True):
        show_debug = st.checkbox("Show raw usage debug info", key="show_raw_usage_debug")
        first_error = _first_pipeline_error_message(pipeline_errors, bool(total.get("api_calls", 0)))

        if not total.get("api_calls", 0):
            if pipeline_error_count:
                st.warning(
                    f"No API usage was recorded because {pipeline_error_count} OpenAI call(s) failed before "
                    "returning usage metadata. The app fell back to local generation where possible."
                )
                if first_error:
                    st.caption(f"First error: {first_error}")
            else:
                st.info("No API usage was recorded for this run.")
            if show_debug:
                if pipeline_errors:
                    st.markdown("**Pipeline errors**")
                    st.json(pipeline_errors)
                st.markdown("**Usage debug**")
                st.json(usage.get("debug", {}))
            return

        if not usage.get("available", False):
            st.info("Token usage not available for this request.")
            if show_debug:
                if pipeline_errors:
                    st.markdown("**Pipeline errors**")
                    st.json(pipeline_errors)
                st.markdown("**Usage debug**")
                st.json(usage.get("debug", {}))
            return

        top_columns = st.columns(4)
        top_columns[0].metric("Model", usage.get("model") or "Unknown")
        top_columns[1].metric("Input tokens", _format_number(total.get("input_tokens", 0)))
        top_columns[2].metric("Output tokens", _format_number(total.get("output_tokens", 0)))
        top_columns[3].metric("Total tokens", _format_number(total.get("total_tokens", 0)))

        if usage.get("pricing_configured"):
            st.metric("Estimated cost (USD)", _format_cost(usage.get("estimated_cost_usd")))
            if usage.get("web_clarification", {}).get("api_calls", 0):
                st.caption("Estimated cost is token-based and may exclude web-search tool-call fees.")
        else:
            st.caption("Estimated cost unavailable because pricing is not configured for this model.")

        detail_columns = st.columns(4)
        detail_columns[0].metric(
            "Concept extraction",
            _format_number(usage.get("extraction", {}).get("total_tokens", 0)),
        )
        detail_columns[1].metric(
            "Concept cleaning",
            _format_number(usage.get("cleaning", {}).get("total_tokens", 0)),
        )
        detail_columns[2].metric(
            "Web clarification",
            _format_number(usage.get("web_clarification", {}).get("total_tokens", 0)),
        )
        detail_columns[3].metric(
            "Cheatsheet generation",
            _format_number(usage.get("generation", {}).get("total_tokens", 0)),
        )

        audit_columns = st.columns(2)
        audit_columns[0].metric(
            "Audit / revision",
            _format_number(usage.get("audit", {}).get("total_tokens", 0)),
        )
        audit_columns[1].metric(
            "API calls",
            _format_number(total.get("api_calls", 0)),
        )

        extra_columns = st.columns(2)
        extra_columns[0].metric("Cached input", _format_number(total.get("cached_input_tokens", 0)))
        extra_columns[1].metric("Reasoning tokens", _format_number(total.get("reasoning_tokens", 0)))

        if pipeline_error_count:
            st.caption(
                f"{pipeline_error_count} pipeline error(s) were recorded. "
                "Some steps may have fallen back to heuristic generation."
            )
            if first_error:
                st.caption(f"First error: {first_error}")

        if show_debug:
            if pipeline_errors:
                st.markdown("**Pipeline errors**")
                st.json(pipeline_errors)
            st.markdown("**Usage debug**")
            st.json(usage.get("debug", {}))


def _render_sources_used() -> None:
    sources = st.session_state.get("sources_used", {})
    if not sources:
        return

    with st.expander("Sources Used", expanded=False):
        st.caption("Lecture slides decide what belongs in the cheatsheet. Web sources are used only to clarify slide concepts.")

        slide_sources = sources.get("slide_sources", [])
        if slide_sources:
            st.markdown("**Slides**")
            for source in slide_sources:
                st.markdown(f"- {source}")

        web_sources = sources.get("web_sources", [])
        if web_sources:
            st.markdown("**Web Clarification Sources**")
            for source in web_sources:
                concept = source.get("concept", "Concept")
                title = source.get("title") or source.get("url") or "Source"
                url = source.get("url", "")
                if url:
                    st.markdown(f"- **{concept}**: [{title}]({url})")
                else:
                    st.markdown(f"- **{concept}**: {title}")

        if not slide_sources and not web_sources:
            st.caption("No sources were recorded for this run.")


def _build_sources_used(concepts: list[dict[str, object]]) -> dict[str, list[dict[str, str]] | list[str]]:
    slide_sources = [entry["name"] for entry in st.session_state.get("extracted_by_file", []) if entry.get("name")]
    web_sources: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for concept in concepts:
        concept_name = str(concept.get("concept", "")).strip() or "Concept"
        for source in concept.get("sources", []) or []:
            if not isinstance(source, dict):
                continue
            title = str(source.get("title", "")).strip()
            url = str(source.get("url", "")).strip()
            key = (concept_name.lower(), title.lower(), url.lower())
            if key in seen or not (title or url):
                continue
            seen.add(key)
            web_sources.append(
                {
                    "concept": concept_name,
                    "title": title,
                    "url": url,
                }
            )

    return {
        "slide_sources": slide_sources,
        "web_sources": web_sources,
    }


def _blank_usage_bucket() -> dict[str, int]:
    return asdict(UsageStats())


def _empty_token_usage_state(model_name: str = "") -> dict[str, object]:
    return {
        "model": model_name,
        "available": False,
        "extraction": _blank_usage_bucket(),
        "cleaning": _blank_usage_bucket(),
        "web_clarification": _blank_usage_bucket(),
        "generation": _blank_usage_bucket(),
        "audit": _blank_usage_bucket(),
        "total": _blank_usage_bucket(),
        "estimated_cost_usd": None,
        "pricing_configured": False,
        "debug": {
            "extraction": [],
            "cleaning": [],
            "web_clarification": [],
            "generation": [],
            "audit": [],
        },
    }


def _format_number(value: int) -> str:
    return f"{int(value):,}"


def _format_cost(value: float | None) -> str:
    if value is None:
        return "Unavailable"
    return f"${value:,.4f}"


def _estimate_cost_usd(model_name: str, usage: UsageStats) -> float | None:
    pricing = _get_pricing_for_model(model_name)
    if pricing is None or usage.usage_available_calls == 0:
        return None

    cached_input_tokens = min(usage.cached_input_tokens, usage.input_tokens)
    uncached_input_tokens = max(usage.input_tokens - cached_input_tokens, 0)

    input_cost = uncached_input_tokens * pricing["input"] / 1_000_000
    cached_input_cost = cached_input_tokens * pricing.get("cached_input", pricing["input"]) / 1_000_000
    output_cost = usage.output_tokens * pricing["output"] / 1_000_000
    return input_cost + cached_input_cost + output_cost


def _get_pricing_for_model(model_name: str) -> dict[str, float] | None:
    if not model_name:
        return None
    if model_name in MODEL_PRICING_PER_MILLION:
        return MODEL_PRICING_PER_MILLION[model_name]

    for configured_name, pricing in MODEL_PRICING_PER_MILLION.items():
        if model_name.startswith(f"{configured_name}-"):
            return pricing

    return None


def _count_pipeline_errors(errors: object, has_successful_api_calls: bool = False) -> int:
    if not isinstance(errors, dict):
        return 0
    return sum(len(_visible_error_entries(value, has_successful_api_calls)) for value in errors.values() if isinstance(value, list))


def _first_pipeline_error_message(errors: object, has_successful_api_calls: bool = False) -> str:
    if not isinstance(errors, dict):
        return ""
    for value in errors.values():
        if not isinstance(value, list):
            continue
        visible = _visible_error_entries(value, has_successful_api_calls)
        if not visible:
            continue
        first = visible[0]
        if not isinstance(first, dict):
            continue
        error_type = str(first.get("error_type", "")).strip()
        message = str(first.get("message", "")).strip()
        parts = [part for part in (error_type, message) if part]
        if parts:
            return ": ".join(parts)
    return ""


def _visible_error_entries(entries: list[object], has_successful_api_calls: bool) -> list[dict[str, object]]:
    visible: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        message = str(entry.get("message", "")).lower()
        if has_successful_api_calls and "incorrect regional hostname" in message:
            continue
        if has_successful_api_calls and (
            "returned no parseable concepts" in message or "returned no usable concepts" in message
        ):
            continue
        visible.append(entry)
    return visible


if __name__ == "__main__":
    main()

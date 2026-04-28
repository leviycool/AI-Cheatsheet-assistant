"""Streamlit app for building compact, exam-oriented cheat sheets from course files."""

from __future__ import annotations

import re
from dataclasses import asdict

import streamlit as st

from cheatsheet_ai.extractors import (
    extract_text_from_docx,
    extract_text_from_pdf,
    extract_text_from_pptx,
    extract_text_from_txt,
)
from cheatsheet_ai.exporters import export_to_docx, export_to_markdown, export_to_pdf
from cheatsheet_ai.generator import (
    GenerationOptions,
    generate_cheatsheet,
    get_openai_model,
    is_openai_configured,
    summarize_chunks,
)
from cheatsheet_ai.processing import chunk_text, clean_extracted_text


SUPPORTED_FILE_TYPES = ["pdf", "pptx", "docx", "txt"]


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
        include_exam_questions = st.checkbox("Include possible exam questions", value=True)
        density = st.radio("Detail level", ["More concise", "Balanced", "More detailed"], index=1)

        if is_openai_configured():
            st.success(f"OpenAI mode enabled ({get_openai_model()})")
        else:
            st.warning("OpenAI key not found. Running in heuristic prototype mode.")
            st.caption("Set OPENAI_API_KEY in your shell or .streamlit/secrets.toml to enable OpenAI mode.")
            if output_language != "English":
                st.caption("Non-English output works best when an OpenAI API key is available.")

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
        include_exam_questions=include_exam_questions,
        density=density,
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
        _render_results()


def _process_uploads_and_generate(uploaded_files, options: GenerationOptions) -> None:
    with st.spinner("Extracting text, cleaning materials, and generating the cheat sheet..."):
        extracted_by_file: list[dict[str, str]] = []
        cleaned_sections: list[str] = []

        for uploaded_file in uploaded_files:
            try:
                raw_bytes = uploaded_file.getvalue()
                extracted_text = _extract_text(uploaded_file.name, raw_bytes)
                cleaned_text = clean_extracted_text(extracted_text)
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

        if not cleaned_sections:
            st.error("No uploaded files could be parsed. Please try a different file set.")
            return

        combined_text = clean_extracted_text("\n\n".join(cleaned_sections))
        _store_extraction_state(extracted_by_file, combined_text)
        _run_generation_pipeline(options)


def _generate_from_existing_text(options: GenerationOptions) -> None:
    with st.spinner("Generating a refreshed cheat sheet..."):
        _run_generation_pipeline(options)


def _run_generation_pipeline(options: GenerationOptions) -> None:
    cleaned_text = st.session_state.get("cleaned_text", "")
    chunks = chunk_text(cleaned_text)
    summaries = summarize_chunks(chunks, options)
    cheatsheet_markdown = generate_cheatsheet(summaries, options, source_text=cleaned_text)

    st.session_state["chunk_count"] = len(chunks)
    st.session_state["chunk_summaries"] = summaries
    st.session_state["generated_markdown"] = cheatsheet_markdown
    st.session_state["editable_cheatsheet"] = cheatsheet_markdown
    st.session_state["last_options"] = asdict(options)
    st.session_state["generation_variant"] = options.variant


def _store_extraction_state(extracted_by_file: list[dict[str, str]], combined_text: str) -> None:
    st.session_state["extracted_by_file"] = extracted_by_file
    st.session_state["cleaned_text"] = combined_text
    st.session_state["source_word_count"] = len(combined_text.split())


def _render_results() -> None:
    st.divider()
    st.subheader("Generated Cheat Sheet")

    stats_columns = st.columns(3)
    stats_columns[0].metric("Source words", st.session_state.get("source_word_count", 0))
    stats_columns[1].metric("Chunks", st.session_state.get("chunk_count", 0))
    stats_columns[2].metric("Mode", "OpenAI" if is_openai_configured() else "Heuristic")

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


def _has_cleaned_text() -> bool:
    return bool(st.session_state.get("cleaned_text"))


def _slugify_filename(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip()).strip("-").lower()
    return slug or "cheatsheet-ai-output"


if __name__ == "__main__":
    main()

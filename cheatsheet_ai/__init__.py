"""Core helpers for the Cheatsheet AI Assistant prototype."""

from .extractors import (
    extract_text_from_docx,
    extract_text_from_pdf,
    extract_text_from_pptx,
)
from .exporters import export_to_docx, export_to_markdown, export_to_pdf
from .generator import (
    GenerationOptions,
    UsageStats,
    audit_cheatsheet,
    audit_cheatsheet_from_concepts,
    clarify_concepts,
    clarify_concepts_with_web,
    clean_concepts,
    extract_concepts,
    extract_usage,
    generate_cheatsheet,
    generate_cheatsheet_from_concepts,
    is_openai_configured,
    summarize_chunks,
)
from .processing import chunk_text, clean_extracted_text

__all__ = [
    "GenerationOptions",
    "UsageStats",
    "audit_cheatsheet",
    "audit_cheatsheet_from_concepts",
    "chunk_text",
    "clarify_concepts",
    "clarify_concepts_with_web",
    "clean_concepts",
    "clean_extracted_text",
    "extract_concepts",
    "extract_usage",
    "export_to_docx",
    "export_to_markdown",
    "export_to_pdf",
    "extract_text_from_docx",
    "extract_text_from_pdf",
    "extract_text_from_pptx",
    "generate_cheatsheet",
    "generate_cheatsheet_from_concepts",
    "is_openai_configured",
    "summarize_chunks",
]

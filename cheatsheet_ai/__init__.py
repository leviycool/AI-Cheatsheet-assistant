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
    generate_cheatsheet,
    is_openai_configured,
    summarize_chunks,
)
from .processing import chunk_text, clean_extracted_text

__all__ = [
    "GenerationOptions",
    "UsageStats",
    "audit_cheatsheet",
    "chunk_text",
    "clean_extracted_text",
    "export_to_docx",
    "export_to_markdown",
    "export_to_pdf",
    "extract_text_from_docx",
    "extract_text_from_pdf",
    "extract_text_from_pptx",
    "generate_cheatsheet",
    "is_openai_configured",
    "summarize_chunks",
]

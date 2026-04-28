# Cheatsheet AI Assistant

Cheatsheet AI Assistant is a practical Streamlit prototype that turns lecture slides, notes, PDFs, and documents into a compact, exam-oriented cheat sheet.

It is designed for messy student materials:

- Upload one or more `PDF`, `PPTX`, `DOCX`, or `TXT` files
- Extract and clean noisy lecture content
- Chunk long materials before summarizing
- Generate a dense cheat sheet tuned for exam prep
- Preview and edit the result inside the app
- Export the final version as Markdown, PDF, or DOCX

## Features

- Multi-file upload panel
- Text extraction for PDFs, PowerPoint slides, Word documents, and text files
- Cleaning pass to reduce repeated headers, footers, slide numbers, and obvious noise
- Chunk-first pipeline for long materials
- Exam-oriented cheat sheet structure
- User controls for language, target length, focus style, examples, formulas, exam questions, and density
- Editable preview before export
- Markdown export by default
- PDF export with a compact two-column layout
- DOCX export for further editing

## Project Structure

```text
app.py
cheatsheet_ai/
  __init__.py
  extractors.py
  processing.py
  generator.py
  exporters.py
requirements.txt
README.md
```

## How It Works

1. The app extracts text from each uploaded file.
2. The extraction is cleaned to remove obvious noise and repeated lines.
3. Long text is chunked into smaller segments.
4. Each chunk is summarized first.
5. Chunk summaries are combined into one final cheat sheet.
6. The user can edit the generated markdown and export it.

## OpenAI vs. Fallback Mode

The app supports two generation modes:

- `OpenAI mode`: if `OPENAI_API_KEY` is set, the app uses the OpenAI API for chunk summarization and final cheat sheet generation.
- `Heuristic mode`: if no API key is available, the app still works using a local rule-based prototype. This is useful for demos, but translation and final phrasing are stronger with OpenAI enabled.

Optional environment variables:

```bash
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_MODEL="gpt-4.1-mini"
```

## Run Locally

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the app

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Notes About Export

- Markdown export is the most robust option.
- PDF export uses a compact, print-friendly two-column layout.
- DOCX export keeps headings, bullets, and tables readable for later editing.
- Chinese or bilingual PDF export depends on available PDF font support, but the prototype includes a Unicode-friendly fallback.

## Suggested Next Improvements

- Add OCR for scanned PDFs
- Add image extraction for slide diagrams
- Add stronger formula detection
- Add source citations from uploaded files
- Add per-section regeneration
- Add richer table rendering in exports

## Prototype Goal

This first version is intentionally simple but functional. It is aimed at students who upload messy course materials and want a fast, dense, exam-oriented cheat sheet rather than a long summary.

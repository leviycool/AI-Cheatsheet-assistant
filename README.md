# Cheatsheet AI Assistant

Cheatsheet AI Assistant is a Streamlit app that turns lecture slides and course documents into compact, exam-oriented cheat sheets.

Instead of producing a page-by-page summary, the app is designed to extract the material students are most likely to revise from:

- core concepts
- measures and formulas
- important distinctions
- examples and findings that clarify the lecture

The current app is built for messy academic inputs such as PDFs, slide decks, lecture notes, and mixed OCR text.

## What It Does

- Upload `PDF`, `PPTX`, `DOCX`, or `TXT` files
- Clean noisy slide extraction and chunk long documents
- Generate an A4-style cheat sheet in English, Chinese, or bilingual output
- Prefer a concept-first structure over raw slide extraction
- Optionally use web search to clarify concepts that already appear in the uploaded slides
- Show token usage from OpenAI response metadata
- Show source transparency in a `Sources Used` panel
- Export results as `Markdown`, `PDF`, or `DOCX`

## How It Works

In OpenAI mode, the app follows a multi-step pipeline:

1. Parse and clean uploaded files
2. Extract candidate concepts from the slides
3. Clean and prioritize the concept inventory
4. Optionally clarify slide concepts with web search
5. Generate a compact cheat sheet from the cleaned concept list
6. Audit the final cheat sheet for duplication, OCR noise, and unsupported content

If no OpenAI API key is configured, the app falls back to a local heuristic mode. That mode is useful for offline testing, but the best results come from OpenAI mode.

## Quick Start

The repository includes a `Makefile` for the common setup flow:

```bash
make install
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
make run
```

The local app usually starts at:

```text
http://localhost:8501
```

## Manual Setup

If you prefer not to use `make`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## OpenAI Configuration

The app supports both environment variables and Streamlit secrets.

### Option 1: Environment Variables

```bash
export OPENAI_API_KEY="sk-your-key-here"
export OPENAI_MODEL="gpt-5.2"
```

### Option 2: Streamlit Secrets

Copy the example file:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then set:

```toml
OPENAI_API_KEY = "sk-your-key-here"
OPENAI_MODEL = "gpt-5.2"
# Optional for regional OpenAI projects:
# OPENAI_BASE_URL = "https://us.api.openai.com/v1"
```

The app also supports grouped secrets:

```toml
[openai]
api_key = "sk-your-key-here"
model = "gpt-5.2"
# Optional for regional OpenAI projects:
# base_url = "https://us.api.openai.com/v1"
```

## Streamlit Community Cloud

To deploy on Streamlit Community Cloud:

1. Connect the repository
2. Set the branch to `main`
3. Set the main file path to `app.py`
4. Add your secrets in `Advanced settings -> Secrets`

Example secrets:

```toml
OPENAI_API_KEY = "sk-your-key-here"
OPENAI_MODEL = "gpt-5.2"
# Optional for regional OpenAI projects:
# OPENAI_BASE_URL = "https://us.api.openai.com/v1"
```

## Token Usage

The app reads token usage directly from OpenAI API response metadata. It does not estimate usage by asking the model.

The `Token Usage` panel shows:

- model name
- input tokens
- output tokens
- total tokens
- per-step usage across extraction, cleaning, web clarification, generation, and audit
- estimated cost when pricing is configured
- raw debug metadata when debug mode is enabled

## Troubleshooting

### `incorrect regional hostname`

If your OpenAI project requires a regional hostname, set:

```toml
OPENAI_BASE_URL = "https://us.api.openai.com/v1"
```

This app also tries to recover automatically when OpenAI returns a hostname hint, but explicitly setting the correct base URL is more reliable.

### The app falls back to heuristic mode

If you see messages about heuristic generation, check:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_BASE_URL` if your project is regional
- the `Token Usage` panel and raw debug info

### Output quality is weak

Best results usually come from:

- lecture slides with recognizable headings
- OpenAI mode instead of heuristic mode
- enabling formulas when the lecture contains measures
- keeping web search enabled only when you want extra clarification for slide-supported concepts

## Limitations

- OCR-heavy PDFs can still produce noisy text
- Heuristic mode is intentionally conservative and may miss concepts
- Web search is only meant to clarify concepts already present in the uploaded lecture material
- The app is optimized for study-sheet generation, not for full lecture transcription or general summarization

## Development

Basic checks:

```bash
make check
```

## Project Structure

```text
app.py
cheatsheet_ai/
  __init__.py
  extractors.py
  processing.py
  generator.py
  exporters.py
.streamlit/
  config.toml
  secrets.toml.example
Makefile
requirements.txt
README.md
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

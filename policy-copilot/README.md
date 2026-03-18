# Policy & Claims Copilot

Simple local CLI RAG app for a single insurance policy PDF.

## What it does

- Answers policy questions using retrieved text from `data/policy.pdf`
- Runs a basic claim pre-check using the same policy context
- Shows grounded supporting sources with quote, section, and page
- Uses local embeddings plus Gemini 2.0 Flash for final answer generation

## Project structure

```text
.
|-- main.py
|-- requirements.txt
|-- README.md
|-- .env
|-- data/
|   `-- policy.pdf
|-- db/
`-- src/
    |-- __init__.py
    |-- chunker.py
    |-- pdf_loader.py
    |-- precheck.py
    |-- prompts.py
    |-- qa.py
    |-- retriever.py
    |-- utils.py
    `-- vector_store.py
```

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure `.env` contains:

```env
GOOGLE_API_KEY=your_key_here
```

4. Ensure the policy file exists at `data/policy.pdf`.

## Run

```bash
python main.py
```

On the first run, the app will:
- read the PDF
- create chunks with page and section metadata
- build a local Chroma index in `db/`

Later runs reuse the saved index unless the PDF changes.

## Notes

- This tool is advisory only and does not make final claim decisions.
- Final claim outcomes depend on insurer review, submitted documents, and full policy terms.

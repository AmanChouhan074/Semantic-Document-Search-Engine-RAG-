# Semantic Document Search Engine | (RAG)

A Streamlit-based semantic search application for documents using vector embeddings and FAISS.

## Features

- Upload PDF, DOCX, and TXT files
- Chunk documents for better semantic indexing
- Use HuggingFace sentence embeddings (`all-MiniLM-L6-v2`)
- Search documents by meaning, not just keywords
- Clean Streamlit UI with a cyberpunk theme

## Requirements

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

From the project folder:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal.

## Usage

1. Open the app in your browser.
2. Upload one or more documents under `UPLOAD`.
3. Configure chunk size and overlap if needed.
4. Click `INDEX DOCUMENTS`.
5. Go to `SEARCH` and enter your query.

## Notes

- The app stores the document index in session state, so reloading the page will reset the index.
- Supported document formats: `pdf`, `docx`, `txt`.
- Use the `DASHBOARD` tab to see loaded documents, chunks, and search counts.

## Files

- `app.py` - Streamlit application code
- `requirements.txt` - Python dependencies
- `README.md` - This file

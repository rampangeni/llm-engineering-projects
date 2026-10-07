# RAG Knowledge Assistant (Insurellm expert)

A retrieval-augmented "expert knowledge worker" for the fictional company **Insurellm**. One project, two selectable pipelines, an evaluation harness and two Gradio apps.

## Concepts covered

| Part | What it covers |
|---|---|
| `basic` pipeline | chunking, embeddings, Chroma retrieval, a single LLM call with chat history |
| `pro` pipeline | semantic chunking, query rewriting, reranking |
| Vector visualization | t-SNE / plotly |
| Evaluation | MRR, nDCG, keyword coverage, LLM-as-judge, Gradio dashboard |
| Chat UI | Gradio chat with pipeline selector |

## Features

- **basic**: LangChain `RecursiveCharacterTextSplitter` (500/200), HuggingFace `all-MiniLM-L6-v2` embeddings (OpenAI embeddings optional), Chroma, top-10 retrieval, one LLM call, conversational history (retrieval query = prior user turns + current question; chat history is passed to the LLM).
- **pro**: LLM-based semantic chunking (headline + summary + original text, ~25% overlap), OpenAI `text-embedding-3-large`, query rewriting using chat history, two retrievals (original + rewritten, 20 each) merged and de-duplicated, LLM reranking, top-10 context, answer prompt tuned for evaluation.
- **Evaluation** on 150 questions (`data/tests.jsonl`): MRR, nDCG, keyword coverage (retrieval) and LLM-as-judge accuracy/completeness/relevance (answers). CLI for a single test and a colour-coded Gradio dashboard with per-category charts.
- **Chat UI** with pipeline selector and a side panel showing retrieved context.
- Optional **t-SNE 2D/3D visualization** of the vector store.
- Sample corpus: `data/knowledge-base/` (markdown only, ~300 KB).

## Architecture

```
data/knowledge-base/*.md -> documents.load_documents
   basic: RecursiveCharacterTextSplitter ----\
   pro  : LLM semantic chunker (threads) -----+-> Embedder -> Chroma (data/vector_db/<pipeline>)
query -> basic: concat user turns -> top-k ----\
      -> pro  : rewrite -> 2x top-k -> merge -> LLM rerank -> top-k -> LLM answer
```
Full design: [docs/lld-design.md](docs/lld-design.md). Code lives in `src/rag_assistant/` (`pipelines/`, `evaluation/`, `store.py`, `embeddings.py`, `llm.py`, `app.py`, `viz.py`, `cli.py`).

## Prerequisites

- Python >= 3.11
- `OPENAI_API_KEY` (LLM calls for both pipelines; embeddings for `pro`). Copy `.env.example` to `.env`.
- `GROQ_API_KEY` only if you set a `groq/...` model (the `pro` pipeline can use `groq/openai/gpt-oss-120b`).
- The `basic` default embedder runs locally and downloads `all-MiniLM-L6-v2` (~90 MB) on first use; it needs the `local` extra (installs PyTorch). To avoid that, set `BASIC_EMBEDDING=openai:text-embedding-3-large`.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[local,viz]"      # drop "local" if using OpenAI embeddings for basic; drop "viz" if you don't need plots
# or with pip:  python -m venv .venv && source .venv/bin/activate && pip install -e ".[local,viz]"
cp .env.example .env                  # then add your keys
```

## Run

```bash
rag ingest --pipeline basic          # build data/vector_db/basic
rag ingest --pipeline pro            # LLM-chunks every document (~76 calls) + embeds; costs a little
rag chat --pipeline pro              # Gradio chat UI (pipeline can also be switched in the UI)
rag ask "Who founded Insurellm?" -p basic
rag eval --test 0 -p pro             # full report for one test row
rag eval --dashboard -p basic        # Gradio evaluation dashboard
rag viz --dims 3 --output vectors.html   # t-SNE plot (needs viz extra)
```
(`python -m rag_assistant.cli ...` works too.) Set `RAG_PIPELINE` to change the default.

## Example

```
$ rag ask "Who won the IIOTY award in 2023?" -p pro
Maxine Thompson won the Insurellm Innovator of the Year (IIOTY) award in 2023.
Sources: employees/Maxine Thompson.md
```
(Illustrative; actual output depends on the model.)

## Configuration

| Env var | Default |
|---|---|
| `RAG_PIPELINE` | `basic` |
| `RAG_DATA_DIR` | `./data` inside the project |
| `BASIC_LLM_MODEL` / `PRO_LLM_MODEL` / `JUDGE_LLM_MODEL` | `openai/gpt-4.1-nano` (any litellm model id) |
| `BASIC_EMBEDDING` | `hf:all-MiniLM-L6-v2` |
| `PRO_EMBEDDING` | `openai:text-embedding-3-large` |

Embedding spec is `hf:<model>` or `openai:<model>`. Each pipeline has its own vector DB; changing the embedding model requires re-ingesting.

## Tests

```bash
uv pip install -e ".[dev]" && pytest
```
Tests are fully offline: LLM and embedding clients are replaced by fakes (Chroma runs locally with the fake vectors). Retrieval metrics, ranking robustness, both pipelines end to end, chat handler and CLI parsing are covered.

## Troubleshooting

- **"Vector store ... is empty"**: run `rag ingest -p <pipeline>` first.
- **ImportError sentence-transformers**: install the `local` extra or switch `BASIC_EMBEDDING` to OpenAI.
- **Rate limits during pro ingest**: lower `pro_workers` in `config.py` (default 3); LLM calls retry with exponential backoff.
- **Wrong results after changing models**: delete `data/vector_db/<pipeline>` and re-ingest.
- **Pro ingest chunking failures** (malformed structured output): retried up to 5 times, then the error is raised; rerun ingest.
- **Gradio 6**: pinned to `<6` because the UI uses `Chatbot(type="messages")`.

## Cost notes

`pro` makes ~76 chunking calls at ingest, and per question: rewrite + embed x2 + rerank (20-40 chunks) + answer. Evaluating all 150 tests costs 150 x (pipeline calls + judge call) per pass.

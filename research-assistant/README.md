# Research Assistant

Ask questions about a folder of PDF research papers and get answers grounded in
those papers, with page-level citations.

Everything runs on your own machine: **no API keys, no paid services, no cloud
inference.**

| Layer | Choice |
| --- | --- |
| Agent | LangChain tool-calling agent (`AgentExecutor`) |
| Vector DB | Weaviate (client-supplied vectors, no server-side vectoriser) |
| Inference + embeddings | Ollama (`qwen2.5:3b`, `nomic-embed-text`) |
| PDF parsing / chunking | pypdf + `RecursiveCharacterTextSplitter` |
| CLI | Typer + Rich |

---

## Quick start

Put your PDFs in [data/](data/), then:

```bash
make up            # start Weaviate + Ollama
make pull-models   # one-off: download the models (~2.3 GB, a few minutes)
make build         # build the assistant image
make ingest        # chunk, embed and index every PDF in ./data
make ask Q="What do these papers conclude about retrieval depth?"
make chat          # interactive, multi-turn session
```

Without `make`, the same thing:

```bash
docker compose up -d weaviate ollama
docker compose run --rm model-puller
docker compose run --rm assistant ingest
docker compose run --rm assistant ask "How does the router balance load?"
docker compose run --rm assistant chat
```

> **Windows / WSL 2:** Docker Desktop must be *running* — not just its
> background `com.docker.service` — and WSL integration must be enabled for your
> distro under Settings → Resources → WSL integration. Otherwise `docker` on your
> `PATH` resolves to a shim that prints *"The command 'docker' could not be found
> in this WSL 2 distro"*.

### Hardware

The default model is **`qwen2.5:3b`** (~2 GB), chosen so the whole stack fits on
an 8 GB machine: Ollama needs ~2.5 GB with context, Weaviate ~300 MB, and Docker
Desktop's VM takes 1–2 GB on top.

With 16 GB or more, a bigger model gives noticeably better synthesis:

```bash
RA_LLM_MODEL=qwen2.5:7b make pull-models   # or llama3.1:8b
RA_LLM_MODEL=qwen2.5:7b make ingest
```

Put it in `.env` to avoid repeating it. CPU-only works — expect roughly 10–30 s
per question at 3B and 30–90 s at 8B. With an NVIDIA GPU, uncomment the
`deploy:` block under the `ollama` service in
[docker-compose.yml](docker-compose.yml) for a large speedup.

---

## Commands

| Command | What it does |
| --- | --- |
| `ingest` | Chunk, embed and index every PDF under the data directory |
| `ask "…"` | Answer one question and print the citations behind it |
| `chat` | Multi-turn session (`/papers`, `/reset`, `/exit`) |
| `status` | Health of Ollama and Weaviate, plus what is currently indexed |
| `reset` | Drop the Weaviate collection |

Useful flags: `ingest --force` (re-embed unchanged papers),
`ingest --data-dir PATH`, `ask --no-sources`, `--verbose` (show the agent's
tool calls and reasoning).

Re-running `ingest` is cheap and safe. Each paper's SHA-256 is stored with its
chunks, so unchanged papers are skipped, and an edited paper has its old chunks
deleted before the new ones are written — no duplicates either way.

---

## How it works

```
data/*.pdf
   │  pypdf ─ one Document per page, dehyphenated and whitespace-normalised
   ▼
chunks (1200 chars, 200 overlap, paragraph-aware)
   │  metadata: source, page, chunk_index, chunk_id, checksum
   ▼
Ollama  nomic-embed-text  ──►  768-dim vectors
   ▼
Weaviate collection "ResearchPaperChunk"   (HNSW, vectors supplied by client)
   ▲
   │  search_papers(query) ─ MMR over k=5 of 20 candidates
   │  list_papers()        ─ inventory of indexed papers
   ▼
LangChain tool-calling agent  ◄──►  Ollama qwen2.5:3b
   ▼
answer + citations (paper.pdf, p. N)
```

The agent is told to search before answering, to search again when results are
thin, to cite every claim, and to say so plainly when the papers do not cover
the question. Retrieved chunks are captured in a `SourceCollector` during each
run, so the citations printed by the CLI are the passages the model actually
saw — not something it composed. Each `Answer` also carries `tools_used`, so you
can tell whether the model retrieved or answered from memory.

**Retry on skipped retrieval.** Small models sometimes read "these papers" as
ambiguous and reply *"which papers do you mean?"* without searching at all. When
`tools_used` comes back empty, the question is put once more with that ambiguity
spelled out explicitly, and whichever attempt actually retrieved is the one
returned. On the 3B default this converts a class of non-answers into grounded,
cited ones; set `retry_without_retrieval=False` on `ResearchAssistant` to turn it
off.

### Layout

```
src/research_assistant/
  config.py        Settings (pydantic-settings, RA_* env vars)
  loader.py        PDF discovery, text cleaning, chunking      ← pure, no I/O beyond disk
  llm.py           Ollama chat + embedding models, preflight checks
  vectorstore.py   Weaviate connection, schema, stats, deletion
  ingest.py        Pipeline: load → chunk → embed → upsert, with idempotency
  agent.py         Tools, prompt, AgentExecutor, conversation state
  cli.py           Typer commands
tests/
  fakes.py         In-memory Weaviate / Ollama / agent doubles
  unit/            152 offline tests
  integration/     26 tests against live Weaviate + Ollama
```

---

## Configuration

Every setting has a working local default; copy [.env.example](.env.example) to
`.env` only if you want to change something. All variables use the `RA_` prefix.

| Variable | Default | Notes |
| --- | --- | --- |
| `RA_LLM_MODEL` | `qwen2.5:3b` | **Must support tool calling** (see below) |
| `RA_EMBEDDING_MODEL` | `nomic-embed-text` | Changing this needs `ingest --force` |
| `RA_OLLAMA_BASE_URL` | `http://localhost:11434` | `http://ollama:11434` in compose |
| `RA_WEAVIATE_HOST` | `localhost` | `weaviate` in compose |
| `RA_CHUNK_SIZE` / `RA_CHUNK_OVERLAP` | `1200` / `200` | Characters |
| `RA_RETRIEVAL_K` / `RA_RETRIEVAL_FETCH_K` | `5` / `20` | Returned / MMR candidates |
| `RA_MAX_AGENT_ITERATIONS` | `6` | Caps the tool-calling loop |
| `RA_REQUEST_TIMEOUT` | `300` | Seconds; raise it for slow CPU inference |

The agent needs a model that supports tool calling. `qwen2.5`, `qwen3`,
`llama3.1`, `llama3.2` and `mistral-nemo` do; plain `gemma2` and `phi3` do not.
A model without it answers from memory instead of searching, which shows up as
an empty `Sources` list.

**A note on the 3B default.** It reliably searches, grounds its answer in what it
retrieves, and the `Sources` list is always accurate — those citations are
collected from the retriever, not from the model's text. But at 3B the model
often *ignores the instruction to cite inline* as `(paper.pdf, p. N)`, giving
correct prose with the citations only in the `Sources` footer. A 7B/8B model
follows the inline-citation instruction much more consistently. If inline
attribution matters to you and you have the RAM, use `qwen2.5:7b`.
Changing the embedding model changes the vector space, so re-index with
`ingest --force` (or `reset` then `ingest`) afterwards.

---

## Tests

```bash
make test              # unit + integration, in a container against live services
make venv test-unit    # 152 unit tests in a local venv, no services, ~9 s
make test-integration  # integration tests against localhost services
make coverage          # unit coverage report
```

**Unit tests** (`tests/unit/`, 152 tests, no marker) run fully offline. Weaviate, Ollama
and the agent executor are replaced by the doubles in [tests/fakes.py](tests/fakes.py)
— including a `FakeToolCallingChatModel` that scripts `AIMessage`s so the *real*
`AgentExecutor` loop (tool call → observation → final answer) is exercised
without any inference. They cover settings validation, PDF cleaning and chunk
metadata, collection lifecycle and filters, ingestion idempotency and
per-file error isolation, tool output and citation formatting, conversation
history windowing, and every CLI command's output and exit code.

**Integration tests** (`tests/integration/`, marker: `integration`) talk to real
Weaviate and real Ollama: schema creation, dense vectors round-tripping through
Weaviate, retrieval actually returning the right paper, re-ingestion not
duplicating chunks, the agent answering with correct citations, and the CLI
end-to-end from `ingest` through `ask`. They use a separate collection that is
dropped on teardown, and **skip themselves with an actionable message** when the
services are not up — so `pytest` passes offline:

```
SKIPPED - Weaviate not reachable at http://localhost:8080
          (start it with: docker compose up -d weaviate)
```

Assertions about generated text stay behavioural (a search was performed, the
citation names the right paper, the retrieved context contains the fact) rather
than matching exact wording, so they do not flake on model sampling.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Could not reach Ollama at …` | `make up`, then `make status` |
| `Missing Ollama model(s): …` | `make pull-models` (the error prints the exact `ollama pull` commands) |
| `Could not connect to Weaviate at …` | `make up`; check `make logs` |
| `The index is empty` | Put PDFs in `data/` and run `make ingest` |
| A paper ingests as `empty` | It is a scanned image with no text layer — OCR it first (e.g. `ocrmypdf in.pdf out.pdf`) |
| Answers ignore the papers | Your model probably lacks tool calling — switch to `qwen2.5:3b` or `llama3.1:8b` |
| Answers are right but have no inline `(paper, p. N)` citations | Expected at 3B; the `Sources` footer is still accurate. Use `qwen2.5:7b` for inline citations |
| Answers are very slow | Enable the GPU block in `docker-compose.yml`, or stay on the 3B default |
| Ollama container is OOM-killed | The model is too big for your RAM — drop to `qwen2.5:3b`, or raise Docker Desktop's memory limit |
| `docker: command not found` in WSL | Enable WSL integration in Docker Desktop settings |

## Costs and network use

Nothing in the runtime path leaves your machine, and no account or key is needed
anywhere. The only outbound traffic is one-off downloads: container images, the
Python wheels, and the Ollama models.

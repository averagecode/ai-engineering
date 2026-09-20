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
evals/
  dataset.yaml     Ground truth: 34 cases over all 12 papers, verified
  metrics.py       Deterministic scorers as pure functions       ← no MLflow, no network
  scorers.py       MLflow @scorer wrappers + optional LLM judges
  verify_dataset.py  Re-checks the ground truth against the PDFs
  run_eval.py      Harness: mlflow.genai.evaluate over the live stack
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

## Evals

MLflow-based evaluation lives in [evals/](evals/). Tracking is local — MLflow
writes to `./mlruns` and the judge can be Ollama — so evals stay free and
key-less like the rest of the stack.

```bash
make eval-verify   # check the ground truth still matches the PDFs (offline, instant)
make eval-smoke    # 3 cases, to check the harness end to end
make eval          # the full deterministic suite against the live stack
make eval-judge    # additionally run LLM-as-judge scorers (slow)
make eval-ui       # browse runs at http://localhost:5000
```

### Deterministic first, judges second

Most regressions in a RAG system are *retrieval* regressions — an embedding-model
swap, a chunk-size change, a broken filter — and those are measurable exactly.
So the default suite uses no LLM judge at all. Every scorer is a pure function
over the agent's `Answer` ([evals/metrics.py](evals/metrics.py)), which makes it
fast, reproducible, and unit-tested in [tests/unit/test_eval_metrics.py](tests/unit/test_eval_metrics.py).

| Scorer | Measures | Catches |
| --- | --- | --- |
| `retrieved_anything` | `tools_used` is non-empty | the model answering from memory instead of searching |
| `retrieved_expected_source` | expected paper among retrieved chunks | retrieval pointing at the wrong paper |
| `retrieved_expected_page` | expected paper on a page that *carries* the fact | right paper, wrong chunk |
| `reciprocal_rank` | 1/rank of the first correct chunk | ranking degradation inside top-k |
| `citations_are_valid` | every cited file exists in the corpus | fabricated sources |
| `citations_are_grounded` | every citation was actually retrieved | citing a page it never read |
| `cited_expected_source` | the answer credits the right paper | mis-attribution |
| `contains_expected_fact` | answer states an expected figure | answer-quality drift |
| `fact_coverage` | fraction of expected figures present | partial answers to multi-fact questions |
| `cites_inline` | prose carries `(paper.pdf, p. N)` | tracks the 3B inline-citation gap explicitly |
| `refused_out_of_corpus` | declined *and* cited nothing | hallucinating on unanswerable questions |

Out-of-corpus cases are scored as a separate group: they should produce a refusal,
so grading them on retrieval metrics would penalise correct behaviour.

### Ground truth

[evals/dataset.yaml](evals/dataset.yaml) holds 34 cases covering **all 12 papers**,
and evals run against the **full corpus** — not a subset. Retrieval difficulty
scales with corpus size and homogeneity, so scoring against fewer papers would
describe an easier system than the one you have.

That creates a problem this corpus makes acute: a dozen papers all about missing
baryons and the SZ effect overlap heavily (`IllustrisTNG` appears in 7, `EAGLE`
in 6). Asserting "only paper X can answer this" is usually *wrong* — several
papers legitimately report a filament gas temperature, and retrieving any of them
is correct. So cases are split by what can honestly be asserted:

| kind | n | Ground truth | Scored on |
| --- | --- | --- | --- |
| `bibliographic` | 12 | A title pins exactly one paper — "who is the lead author of …" | source + answer + integrity |
| `scoped_fact` | 12 | Question names the paper by title, asks for one of its figures | source + answer + integrity |
| `open_fact` | 6 | Information the corpus contains, no paper named | answer + integrity only |
| `inventory` | 1 | The corpus itself | integrity |
| `out_of_corpus` | 3 | Unanswerable — must decline | refusal + integrity |

`scoped_fact` works because **paper titles sit in the page-1 text that gets
embedded**, so retrieval can be steered to a named paper without needing a
metadata filter. `open_fact` cases never assert a source, so a correct answer
drawn from an equally valid paper is not punished.

`make eval-verify` re-checks every claim against the actual PDF text and warns
about papers no case exercises. Run it whenever the corpus changes — **a stale
expectation turns a working system into a failing score**, and nothing else in the
suite can catch that. It has already earned its keep twice: it caught a pattern
broken by a PDF gluing an affiliation marker to an author surname
(`Jianzhuo Lia`), and an inventory case that was checking page text when
filenames only exist in metadata.

### On LLM-as-judge

`make eval-judge` adds MLflow's `Correctness` and `Guidelines` scorers, pointed at
Ollama (`ollama:/<model>` — MLflow has a first-class Ollama provider that needs no
API key). Be clear-eyed about the limit: with one local model, **the judge is the
model under test**, and a 3B judge is weak. Treat those scores as a smoke signal
and rely on the deterministic scorers for regressions. Pointing `--judge-model` at
a larger model is what makes them meaningful.

MLflow's built-in retrieval scorers — `RetrievalGroundedness`, `RetrievalRelevance`,
`RetrievalSufficiency`, `ToolCallCorrectness` — all require a `trace` column,
because they read MLflow retriever spans rather than input/output text. Using them
means instrumenting the agent with `mlflow.tracing` first; that is not wired up
here, and the custom scorers cover the same ground deterministically.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Could not reach Ollama at …` | `make up`, then `make status` |
| `Missing Ollama model(s): …` | `make pull-models` (the error prints the exact `ollama pull` commands) |
| `Could not connect to Weaviate at …` | `make up`; check `make logs` |
| `dependency failed to start: container …-weaviate-1 is unhealthy` | Weaviate is running but `/v1/.well-known/ready` returns 503, and the log says `raft … not part of a stable configuration`. Its RAFT node name must stay stable across container recreation — see `CLUSTER_HOSTNAME` in [docker-compose.yml](docker-compose.yml). If you changed that value, the persisted RAFT log no longer matches: either change it back, or `docker compose down -v` and re-ingest |
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

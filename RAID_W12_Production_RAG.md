### Overview

The goal of this project is to build, evaluate and deploy a **production-grade Retrieval-Augmented Generation (RAG) system** — a question-answering service that grounds every answer in a real corpus of documents, exposes it as a documented HTTP API, monitors itself, and can be honestly defended in a technical interview.

You will pick a real corpus, design and measure the retrieval pipeline, wrap it in a FastAPI service with proper request validation, add one caching layer and one reliability pattern (documented in the README), and demonstrate the whole thing to a coding mentor in a 30-minute defense.

This is a 2-day intensive RAID. Time budget: ~10 hours per day per person, matching the AI Piscine RAID format.

### Role Play

You are the founding ML engineer at a small Kazakh fintech, legal-tech, or ed-tech startup. Business has asked you to build an internal "ask the docs" assistant that lets employees query the company's knowledge base in natural language and get grounded, source-cited answers. The founding CEO can only fund one engineer for the prototype (that's you), so the system must be end-to-end: from ingestion to serving to a documented failure story. On the day-2 demo call, the CEO will ask three things: (a) how do you know it works? (b) what happens when it breaks? (c) how do we know how much it costs? You must answer all three with numbers, not adjectives.

### Group Size

**2–3 students** per RAID. Solo work is not allowed. Two students is the recommended size — one focused on retrieval and evaluation, one on API + deployment + monitoring, both integrating and defending together. Three-person groups must show clear per-person ownership: at defense, each member is asked to walk through code they personally wrote.

### Learning Objectives

By the end of this project, you will be able to:

1. Choose a document loader and chunking strategy appropriate to a real, messy corpus (PDF, Markdown, HTML), and justify chunk size and overlap with a small ablation study.
2. Compute embeddings using a pre-trained model (Sentence-BERT family), store them in a vector database (FAISS or ChromaDB), and defend the choice.
3. Build a retrieval pipeline end-to-end: load → clean → chunk → embed → index → query → retrieve, and quantify its quality with a labeled ground-truth set using `recall@k` and one additional metric (`MRR` or `nDCG`).
4. Wrap an LLM call in a Pydantic-validated FastAPI endpoint that returns structured JSON with sources and confidence.
5. Add operational endpoints — `/health` and `/ready` — that distinguish "process alive" from "model loaded".
6. Design a system prompt that constrains the model to grounded JSON output, and demonstrate that it survives a small regression suite of test questions.
7. Add at least one caching layer (embedding cache, retrieval cache, or response cache) and measure its effect on p95 latency.
8. Add at least one reliability pattern: timeout, retry with exponential backoff, or graceful degradation with a documented fallback path.
9. Instrument the service with structured logs tied to a `request_id` and expose at least five metrics: p50/p95 latency, error rate, token usage per request, cache hit rate, and retrieval quality.
10. Run the service locally with a single command (`uvicorn app.main:app --port 8000`) from a fresh clone with a documented setup.
11. Write at least one Architecture Decision Record (ADR) explaining a non-obvious choice.
12. Defend every design choice verbally in a 30-minute oral defense, without any LLM assistance.

### Instructions

#### Data — pick one corpus

Groups pick exactly one of the following at the start of the RAID:

**Option A — SEC 10-K filings (English, ~500 pages total).**
Pick 3–5 companies from EDGAR (Apple, Tesla, Amazon, NVIDIA, Microsoft are canonical). Download the most recent 10-K annual filing for each in HTML or PDF. Ground-truth questions are numeric-heavy ("What was Apple's revenue in Q3 2023?"), which forces you to think about numeric extraction and citation.

**Option B — Kazakh Labor Code + Tax Code (Russian, structured legal text).**
Publicly available on `adilet.zan.kz`. One well-structured corpus with numbered articles. Ground-truth questions are article-lookup style ("Каков минимальный размер отпускных согласно ТК РК?"). Rewards precise chunking by article boundary.

You may propose a third corpus of your own (scientific papers, a company's public documentation, etc.) — approval by a mentor is required. The corpus must be legally usable and contain at least ~200 pages of real text.

#### 1. Corpus preparation and chunking

Create `notebooks/01_ingestion.ipynb`:

- Load raw documents using an appropriate loader (`PyPDFLoader`, `UnstructuredMarkdownLoader`, `WebBaseLoader`, or a custom parser).
- Clean and normalize the text (strip page headers/footers, fix hyphenation across line breaks, decide what to do with tables).
- Attach metadata to every chunk: at minimum `source_file`, `page` (if applicable), `section_title` (if applicable), `chunk_id`, `char_start`, `char_end`, and `document_version`.
- Implement **at least two** chunking strategies. `RecursiveCharacterTextSplitter` is required. A second strategy (semantic chunking with `SemanticChunker`, section-based, or a custom rule for your corpus) is required. Run both and record the number of chunks produced.
- Persist chunks + metadata to disk (JSONL or Parquet). The pipeline from raw docs to persisted chunks must be idempotent and reproducible via `scripts/build_index.py`.

#### 2. Embedding and indexing

Create `scripts/build_index.py`:

- Choose one embedding model. `sentence-transformers/all-MiniLM-L6-v2` (English) or `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (mixed / Russian) are safe defaults. If you choose a different model, defend the choice in your ADR.
- Compute embeddings for all chunks. Include an on-disk embedding cache keyed by `hash(text + model_name + preprocessing_version)` — this is a required piece of engineering, not optional, because you will re-index at least twice during ablation.
- Index into a vector database. `FAISS` (in-memory, `IndexFlatIP` or `IndexHNSWFlat`) or `ChromaDB` (persistent) both acceptable. If you use FAISS, you must persist the index and the chunk store to disk and demonstrate that the service loads them at startup without re-embedding.
- The index build must complete on a laptop CPU in under 15 minutes for the chosen corpus. If it doesn't, your chunks are wrong or your corpus is too big — cut it before defense.

#### 3. Retrieval evaluation

Create `evaluation/ground_truth.jsonl` and `evaluation/evaluate_retrieval.py`.

- Build a ground-truth set of **at least 15 questions** for your corpus. For each question, list the chunk IDs (or the source_file + section) that contain the correct answer. **All 15 questions must be authored by you** — no LLM-generated questions in the ground-truth set. Aim for a spread: some numeric-fact questions, some definition questions, some multi-chunk questions where the answer requires combining two sections.
- Implement `evaluate_retrieval.py` that measures at minimum:
  - `recall@5`
  - one of: `MRR`, `nDCG@10`, or `precision@5`
- Report a numeric table in `README.md`. Target: `recall@5 ≥ 0.75` on the ground truth set to pass the defense. Above 0.85 is where an interviewer starts nodding.

**Reference point:** the naive baseline (fixed 500-char chunks, no overlap, `all-MiniLM-L6-v2`, exact cosine top-5) on an SEC 10-K corpus with 15 hand-labeled questions hits `recall@5 = 0.73` (see `baseline_recall.md` provided with this RAID). Clearing 0.75 requires at least one deliberate improvement over that baseline: better chunking (overlap, sentence-aware, or section-aware), a stronger embedding model, or hybrid retrieval (BM25 + vector). Anything above 0.85 typically requires two of these three.

**Note for Kazakh Labor Code (Option B) groups:** the structured legal text will likely hit `recall@5 ≥ 0.85` even with the naive baseline setup, because articles are self-contained. If your baseline hits above 0.85, you must run **at least two ablations** (chunking + one other lever) to demonstrate that you understand what is driving the number. The defense score is not driven by absolute recall alone — it is driven by whether you can explain the number.

#### 4. Ablation study (required)

Run at least **one** ablation and report a plot or table:

- **Chunk size sweep:** fix everything else, vary chunk size across {200, 400, 800, 1600} characters, measure `recall@5` on the same 15-question set.
- **OR embedding model comparison:** fix chunk size, compare 2 embedding models, report both `recall@5` and encoding latency.
- **OR retrieval strategy comparison:** pure vector vs hybrid (BM25 + vector with Reciprocal Rank Fusion), same corpus, same ground truth.

The ablation is graded on the presence of numbers and the presence of one honest sentence explaining what the numbers actually mean for your use case. "500 tokens was best" is not enough — "500 tokens hit recall@5=0.82 while 200 tokens hit only 0.61 because our corpus has multi-sentence definitions" is what we want.

#### 5. Generation and prompt engineering

Create `app/generation.py`:

- Wrap an LLM in a function `answer(question: str, retrieved_chunks: list[Chunk]) -> AnswerResponse`. Any inference backend is acceptable: OpenAI API, Anthropic API, a local Ollama model, or a hosted Yandex/inDrive endpoint. Default recommendation: **Ollama with `llama3.2:3b` or `qwen2.5:7b`** so you have no API costs.
- Design a system prompt that:
  - Instructs the model to answer ONLY from the retrieved chunks.
  - Returns strict JSON conforming to a Pydantic schema with fields `answer` (str), `sources` (list of source references), `confidence` (float in [0, 1]), and `used_context` (bool).
  - Refuses to answer with `"answer": "I don't know from the provided context"` when the retrieved chunks are irrelevant.
- Include a **prompt version** in code (`PROMPT_VERSION = "rag_v1"`) and log it with every request.
- Build a **prompt regression test** in `tests/test_prompt.py`: at least 5 questions where you know exactly what the model must return (or must refuse). This test is run in CI and must pass.

#### 6. API service (FastAPI)

Create `app/main.py`. The service must expose:

- `GET /health` — liveness. Always returns `{"status": "ok"}` if the process is up.
- `GET /ready` — readiness. Returns `{"ready": true}` only if the model, index, and cache are loaded and accessible.
- `POST /ask` — the main endpoint. Accepts a Pydantic `AskRequest` with fields `question: str` (required, min length 3, max length 500) and optional `top_k: int` and `filters: dict`. Returns the `AnswerResponse` from generation.
- `GET /metrics` — Prometheus-format metrics, or a simple JSON snapshot of the counters if Prometheus is too heavy.

Requirements:

- Every request has a `request_id` (UUID4, generated if not passed in headers). Every log line for that request carries the ID.
- Pydantic validates every field. Invalid input returns HTTP 422 with a helpful error, never a 500 with a stack trace.
- The service must survive a 10-minute smoke test where you send 100 random-length questions from the ground-truth set with `wrk`, `hey`, or a Python loop, without leaking memory or crashing.

#### 7. Caching

Add at least one cache layer. Options in increasing order of value:

- **Embedding cache** (required if not already in §2) — keyed by `hash(text + model + preprocessing_version)`.
- **Retrieval cache** — keyed by `hash(query + collection_id + index_version + top_k)`. Trivial in-memory dict is acceptable; log `cache_hit` metric.
- **Response cache** — keyed by `hash(question)`. **You must document a cache invalidation strategy** (TTL, index_version-bump, or manual invalidate endpoint) or omit this layer with an ADR explaining why.

Measure the cache's effect: run 50 questions twice, report p95 latency with and without cache. Both numbers go in the README.

#### 8. Reliability

Add at least **one** of the following, and unit-test it (send a mocked failure through your handler and assert the correct response):

- **Timeout** — bound the LLM call to N seconds; return HTTP 504 with a clean error message on timeout.
- **Retry with exponential backoff** — up to 3 attempts on transient failures, wait 1s → 2s → 4s.
- **Graceful degradation** — if the vector DB is unreachable, return an honest "I couldn't search the documents right now" with a `degraded: true` flag rather than crashing.

You do **not** need to demonstrate this live with a real fault injection at defense. Instead, your README must include a "What happens when it breaks" section (see §11) that lists at least three failure modes and how the service responds, and your `tests/` folder must contain the unit test that proves your chosen pattern works.

#### 9. Observability

At minimum, log per request:

- `request_id`, `question`, `retrieved_chunk_ids`, `prompt_version`, `model_name`, `latency_ms_by_stage` (embedding, retrieval, generation, total), `token_usage`, `answer_length`, `cache_hit_bool`, `error_bool`.

Logs go to stdout in JSON format. In the README, show one example log line for a successful request and one for a failed request.

Metrics endpoint exposes: p50/p95 latency, error rate over last 100 requests, cache hit rate, total tokens consumed, and one retrieval-quality metric (e.g., mean retrieval score of top-1 chunk over last 100 requests).

#### 10. Packaging and local run

The service must run from a fresh clone with these three commands:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

- Provide a `requirements.txt` (pinned versions) and a `Makefile` (or a `scripts/run.sh`) with `make install`, `make index`, and `make serve` targets.
- Provide a `.env.example` with any config the service reads (LLM endpoint, model name, top_k). `.env` is gitignored.
- Startup time from `uvicorn` invocation to `/ready` returning 200 must be under 30 seconds on a modern laptop (the index is loaded from disk, embeddings are not recomputed).
- **Docker is optional and does not affect the grade.** If your group already knows Docker Compose and wants to include a `docker-compose.yml`, feel free — but do not spend 2-day RAID time on it.

#### 11. Documentation

`README.md` must contain, in this order:

1. **What this project does** — one paragraph.
2. **Corpus and licensing** — where you got the documents, and confirmation you're allowed to use them.
3. **Quickstart** — the three-command local run from §10, plus a working `curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"..."}'` example with a sample response.
4. **Architecture diagram** — components, arrows, and where each one lives (a hand-drawn image is fine; must be in `docs/architecture.png` or embedded in the README).
5. **Retrieval quality** — the `recall@5` and one other metric on your 15-question set, in a small table.
6. **Ablation results** — the plot or table from §4.
7. **Latency and cost budget** — p50, p95, and estimated cost per 1,000 questions (compute or API).
8. **What happens when it breaks** — a short section listing at least 3 failure modes and how the system responds. This replaces the live fault-injection demo. Be specific: name the component, name the failure, name the observable behavior (e.g., "if the LLM API times out after 30s, we return HTTP 504 with body `{'error': 'llm_timeout', 'request_id': '...'}`; retry logic already attempted 3 times before this point").

`docs/adr/` must contain **at least 1 ADR** following the format: **Context → Options considered → Decision → Consequences**. Pick one of:

- Chunking strategy: what you picked and what you rejected.
- Vector DB choice.
- Embedding model choice.
- Cache invalidation approach.

### Rules and Thresholds (must be hit to pass defense)

- `uvicorn app.main:app --port 8000` in a fresh clone reaches a healthy state (both `/health` and `/ready` return 200) within 30 seconds.
- `POST /ask` on a valid question returns a JSON response with `answer` and non-empty `sources` in under 15 seconds p95 on a modern laptop CPU (no GPU required).
- `POST /ask` on `{"question": ""}` returns HTTP 422, never a 500.
- `recall@5` on the 15-question ground truth set is **≥ 0.75**. Above 0.85 is target.
- The prompt regression test (`pytest tests/test_prompt.py`) passes on the current `main`.
- At least one cache layer is present and its effect on p95 latency is documented with numbers.
- At least one reliability pattern is present, has a unit test that proves it works, and is named in the README failure section.
- README includes the numeric table from §11 point 5 and the ablation plot from §11 point 6.
- At least 1 ADR is present in `docs/adr/`.
- No hardcoded credentials in the repository. `.env.example` is present; `.env` is gitignored.

### Repository structure

```
project/
│  README.md
│  requirements.txt
│  Makefile                  (install / index / serve / test targets)
│  .env.example
│  .gitignore
│
├── app/
│   │  main.py               (FastAPI app, endpoints)
│   │  schemas.py            (Pydantic models)
│   │  retrieval.py          (retriever + cache)
│   │  generation.py         (LLM call + prompt)
│   │  observability.py      (logging, metrics, request_id)
│   │  reliability.py        (timeout / retry / fallback)
│
├── scripts/
│   │  build_index.py        (idempotent index build)
│
├── notebooks/
│   │  01_ingestion.ipynb
│   │  02_ablation.ipynb
│
├── evaluation/
│   │  ground_truth.jsonl
│   │  evaluate_retrieval.py
│   │  results/
│
├── tests/
│   │  test_api.py           (endpoint tests, TestClient)
│   │  test_prompt.py        (5+ prompt regression tests)
│   │  test_reliability.py   (unit-test for chosen reliability pattern)
│   │  test_retrieval.py     (recall@5 must be ≥ 0.75)
│
├── docs/
│   │  architecture.png
│   │  adr/
│   │  │   001-chunking.md   (or your chosen topic)
│
└── data/
    │  raw/                  (source corpus, gitignored if large)
    │  chunks.jsonl          (persisted chunks + metadata)
    │  index/                (persisted vector index)
```

### Defense (30 minutes, offline, slot-booked)

- **5 minutes** — demo. You start the service with `uvicorn app.main:app` and take three live questions. One is from the ground-truth set (should succeed), one is deliberately off-topic (should refuse honestly), and one is the mentor's choice.
- **15 minutes** — retrieval and evaluation. Mentor will ask you to walk through the ablation numbers, defend chunk size, defend embedding model, and open the ground-truth set to spot-check whether the labels are honest. Expect one question that requires you to modify `evaluate_retrieval.py` live.
- **5 minutes** — observability + failure story. Mentor asks: "read me the JSON log line for one recent request", "which reliability pattern did you pick and where is the test that proves it works", and "which failure mode from your README section 8 is most likely to happen first in production and why?"
- **5 minutes** — architecture. One whiteboard question: "How would this scale to 1M documents? What breaks first?" Expected: a coherent answer covering embedding cost, index memory, and where you would introduce async processing.

You may **not** use an LLM during the defense. You may use your notes, your code, your ADRs, and your README. If you use an LLM you fail the RAID. Each group member is asked to walk through code they personally wrote — decide who owns what before defense day.

### Tips

1. Get the whole pipeline working end-to-end with tiny corpus and dumb defaults before you optimize anything. First green light, then tuning.
2. Persist your index. Rebuilding embeddings on every service start is a self-inflicted wound.
3. Your ground truth set is your most important artifact. Spend an afternoon on it. Sloppy labels make everything else meaningless.
4. If `recall@5` is stuck below 0.6, look at your chunks before you blame the embedding model. Ninety percent of the time the chunks are the problem.
5. The system prompt matters as much as the retrieval. A great retriever plus a sloppy prompt still hallucinates. Test the prompt with adversarial inputs (irrelevant questions, empty context, contradictory chunks).
6. Log with structure, not with `print`. JSON logs cost you 15 minutes to set up and save you hours in the defense.
7. Do the ablation early. If you leave it to the last day, you will not have a plot on defense day.
8. The 30-minute defense is a real interview simulation. Practice with your group at least once against each other before the real defense — including who owns which files.
9. Day 1 ends with a working end-to-end pipeline (even if bad numbers). Day 2 is when you improve. If Day 1 does not end with a running `POST /ask`, you have already lost the RAID.

### Resources

- **LangChain** loaders and splitters: https://python.langchain.com/docs/how_to/#document-loaders
- **FAISS** wiki (index types): https://github.com/facebookresearch/faiss/wiki
- **ChromaDB** getting started: https://docs.trychroma.com/getting-started
- **Sentence-Transformers** model directory: https://www.sbert.net/docs/pretrained_models.html
- **FastAPI** tutorials: https://fastapi.tiangolo.com/tutorial/
- **Pydantic** validation examples: https://docs.pydantic.dev/latest/concepts/validators/
- **Ollama** for local LLMs: https://ollama.com/library
- **Prometheus** Python client (if you go beyond a JSON /metrics): https://github.com/prometheus/client_python
- **Chip Huyen, *Designing Machine-Learning Systems*, Chapters 8–10** — monitoring, deployment, incident response.
- **Vercel AI SDK / Weights & Biases writeups on RAG evaluation** (search "RAG evaluation" — many honest field reports exist).

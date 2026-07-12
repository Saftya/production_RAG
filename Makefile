.PHONY: install index serve test eval diagnose clean

install:
	python3 -m venv venv && . venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

index:            ## ingest data/raw -> chunks -> embeddings -> FAISS index
	python3 scripts/build_index.py --strategy section

serve:
	uvicorn app.main:app --port 8000

test:             ## hermetic: offline backends, fixture corpus
	EMBEDDING_BACKEND=hash LLM_BACKEND=stub pytest -q

eval:             ## the recall number for the README / defense
	python3 evaluation/evaluate_retrieval.py --mode hybrid

diagnose:         ## per-question OK/FAIL, to check ground-truth labels
	python3 evaluation/evaluate_retrieval.py --diagnose

clean:
	rm -rf data/index data/.embed_cache data/chunks.jsonl .pytest_cache

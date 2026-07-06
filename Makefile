.PHONY: install ingest index serve test eval clean

install:
	python3 -m venv venv && . venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

ingest:
	python3 scripts/ingest.py --raw data/raw --strategy section

index:
	python3 scripts/build_index.py

serve:
	uvicorn app.main:app --port 8000

test:
	EMBEDDING_BACKEND=hash LLM_BACKEND=stub pytest -q

eval:
	python3 evaluation/evaluate_retrieval.py --mode hybrid

clean:
	rm -rf data/index data/.embed_cache data/chunks.jsonl .pytest_cache

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import settings
from app.embeddings import build_embedder
from app.vectorstore import FaissStore
from app.retrieval import Retriever
from evaluation.evaluate_retrieval import is_relevant, load_gt

gt = load_gt()
r = Retriever(build_embedder(), FaissStore.load(settings.index_dir), mode="hybrid")
ok = 0
for it in gt:
    got, _ = r.retrieve(it["question"], top_k=5)
    hit = any(is_relevant(rc, it) for rc in got)
    ok += hit
    mark = "OK  " if hit else "FAIL"
    print(f'{mark} [{it["id"]}] {it["question"][:55]}')
    if not hit:
        print(f'      метка: {it.get("relevant_sections")}')
        print('      топ-3:', [ (rc.chunk.source_file, rc.chunk.section_title[:35]) for rc in got[:3] ])
print(f'\nrecall@5 = {ok}/{len(gt)} = {ok/len(gt):.3f}')

"""System prompt, context formatting, and prompt version.

The prompt forces grounded, source-cited, JSON-only output and an explicit
refusal path. Bumping the prompt bumps PROMPT_VERSION so it is logged and can be
correlated with a change in eval numbers.
"""
from __future__ import annotations

from app.schemas import RetrievedChunk

PROMPT_VERSION = "rag_v1"

SYSTEM_PROMPT = """You are a grounded question-answering assistant for an internal document base.

Rules:
1. Answer ONLY using facts found in the CONTEXT below. Do not use outside knowledge.
2. If the context does not contain the answer, you MUST reply with exactly:
   {"answer": "I don't know from the provided context", "sources": [], "confidence": 0.0, "used_context": false}
3. Cite the chunk_id of every source you used in "sources".
4. "confidence" is your calibrated certainty in [0,1] that the answer is supported by the context.
5. Reply with a SINGLE JSON object and nothing else. No markdown, no prose outside JSON.

Response schema:
{"answer": <string>, "sources": [<chunk_id string>, ...], "confidence": <float 0..1>, "used_context": <bool>}
"""


def format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no context retrieved)"
    blocks = []
    for rc in chunks:
        c = rc.chunk
        header = f"[chunk_id={c.chunk_id}] source={c.source_file}"
        if c.section_title:
            header += f" section={c.section_title}"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    user = f"CONTEXT:\n{format_context(chunks)}\n\nQUESTION: {question}\n\nJSON answer:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

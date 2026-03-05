"""Citation-enforced prompt templates for the RAG chain."""
from __future__ import annotations

from askdocs.retrieval.vector_store import SearchResult

# ── System prompt ─────────────────────────────────────────────────────────────
# The system prompt is deliberately strict about citation format so that the
# evaluation pipeline can reliably parse and audit inline citations.

SYSTEM_PROMPT = """\
You are a precise, expert assistant that answers questions using ONLY the \
context chunks provided below.

STRICT RULES — violating any rule is an error:
1. Every factual statement MUST end with one or more inline citations in the \
format [chunk_id], e.g. "Python is interpreted [a1b2_0001]."
2. If a claim is supported by multiple chunks, cite all of them: \
[chunk_id_1][chunk_id_2].
3. Do NOT use knowledge from outside the provided context chunks.
4. If the answer cannot be found in the provided chunks, respond EXACTLY:
   "I don't have enough information in the provided documents to answer this."
5. Be concise and direct. Avoid padding, filler sentences, or repeating \
the question.
"""

# ── Context + question template ────────────────────────────────────────────────

_CONTEXT_TEMPLATE = """\
=== CONTEXT CHUNKS ===
{chunks}
=== END CONTEXT ===

Question: {question}

Answer (inline citations mandatory — e.g. "fact [chunk_id]"):"""


def format_chunks(results: list[SearchResult]) -> str:
    """Render a numbered list of chunk blocks for the prompt."""
    lines: list[str] = []
    for r in results:
        lines.append(f"[{r.chunk_id}]\n{r.text.strip()}")
    return "\n\n".join(lines)


def build_prompt(question: str, results: list[SearchResult]) -> tuple[str, str]:
    """
    Build the (system_prompt, user_prompt) tuple for the LLM call.

    Returns:
        (system_prompt, user_prompt) — pass directly to LLMClient.complete().
    """
    chunks_str = format_chunks(results)
    user_prompt = _CONTEXT_TEMPLATE.format(chunks=chunks_str, question=question)
    return SYSTEM_PROMPT, user_prompt

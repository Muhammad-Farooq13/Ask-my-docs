"""Streamlit UI for AskMyDocs."""
from __future__ import annotations

import json
import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000/api/v1")

st.set_page_config(page_title="AskMyDocs", page_icon="📚", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 AskMyDocs")
    st.caption("Hybrid RAG · Cross-encoder reranking · Citation enforcement")
    st.divider()

    st.header("⚙️ Ingest Documents")
    source_path = st.text_input("Source path (file or directory)", value="data/raw")
    reset_index = st.checkbox("Reset existing index", value=False)

    if st.button("▶  Ingest", type="primary", use_container_width=True):
        with st.spinner("Ingesting documents…"):
            try:
                resp = requests.post(
                    f"{API_BASE}/ingest",
                    json={"source_path": source_path, "reset": reset_index},
                    timeout=300,
                )
                if resp.status_code == 200:
                    d = resp.json()
                    st.success(f"✓ Indexed {d['chunks_indexed']} chunks")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API. Is `make serve` running?")

    st.divider()

    st.header("📊 Index Status")
    if st.button("Refresh", use_container_width=True):
        try:
            resp = requests.get(f"{API_BASE}/health", timeout=5)
            if resp.status_code == 200:
                d = resp.json()
                col1, col2 = st.columns(2)
                col1.metric("Status", d["status"].upper())
                col2.metric("Chunks", d["vector_store_size"])
            else:
                st.warning("API returned an error.")
        except requests.exceptions.ConnectionError:
            st.error("API unreachable.")

# ── Main area ─────────────────────────────────────────────────────────────────
st.header("💬 Ask a Question")

question = st.text_area(
    "Your question",
    height=100,
    placeholder="What does the documentation say about…?",
    label_visibility="collapsed",
)

col_ask, col_clear = st.columns([1, 6])
with col_ask:
    ask = st.button("Ask", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=True):
        st.rerun()

if ask and question.strip():
    with st.spinner("Retrieving and generating answer…"):
        try:
            resp = requests.post(
                f"{API_BASE}/query",
                json={"question": question},
                timeout=120,
            )
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach API.")
            st.stop()

    if resp.status_code == 200:
        data = resp.json()

        st.subheader("Answer")
        st.markdown(data["answer"])

        if data["missing_citations"]:
            st.warning(
                f"⚠️ Citation gap: chunks not cited — "
                + ", ".join(f"`{c}`" for c in data["missing_citations"])
            )

        st.divider()
        st.subheader(f"📎 Retrieved Sources  ({len(data['sources'])} chunks)")
        for src in data["sources"]:
            label = f"[{src['chunk_id']}]  {src['filename']}  —  score: {src['score']:.4f}"
            with st.expander(label):
                st.markdown(f"```\n{src['text']}\n```")
                st.caption(f"Full path: {src['source']}")

    elif resp.status_code == 503:
        st.warning("Index not loaded yet. Use the Ingest panel on the left.")
    else:
        st.error(f"API error {resp.status_code}: {resp.text}")

elif ask:
    st.warning("Please enter a question first.")

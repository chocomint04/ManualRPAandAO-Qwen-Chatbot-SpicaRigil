"""
RPA & AO Manual RAG Chatbot - Streamlit Version
RAG system using Qwen2.5-1.5B-Instruct + FAISS + BGE embeddings
"""

import os
import time
import warnings
import numpy as np
import torch
import streamlit as st

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

warnings.filterwarnings("ignore")
os.environ["PYTHONUNBUFFERED"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load all models (cached so they only load once) ──────────────
@st.cache_resource(show_spinner="Loading models, please wait (this may take a few minutes)...")
def load_all():
    langchain_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = FAISS.load_local(
        "./faiss_index_phi",
        langchain_embeddings,
        allow_dangerous_deserialization=True,
    )

    LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

    return langchain_embeddings, vector_store, tokenizer, model, reranker


# ── RAG Pipeline ─────────────────────────────────────────────────

def is_query_relevant(query, langchain_embeddings, threshold=0.25):
    domain_probe = "real property appraisal assessment tax Manila Philippines LGU"
    q_emb = langchain_embeddings.embed_query(query)
    d_emb = langchain_embeddings.embed_query(domain_probe)
    similarity = np.dot(q_emb, d_emb) / (
        np.linalg.norm(q_emb) * np.linalg.norm(d_emb)
    )
    return float(similarity) > threshold


def retrieve_context(query, vector_store, reranker, k=6, fetch_k=20):
    docs = vector_store.similarity_search(query, k=fetch_k)
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), reverse=True)

    top_score = ranked[0][0] if ranked else -999
    if top_score < -5.0:
        return None, []

    top_docs = [doc for _, doc in ranked[:k]]
    context = "\n\n---\n\n".join([doc.page_content for doc in top_docs])
    return context, top_docs


def build_prompt(question, context, tokenizer):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant specializing in Manila real estate property tax regulations. "
                "Answer ONLY using the provided context. "
                "Do NOT add any information not explicitly stated in the context. "
                "If the answer is not in the context, say exactly: 'The information is not available in the document.' "
                "If the question is unrelated to real property appraisal, assessment, or the BLGF manual, "
                "respond with: 'This question is outside the scope of the property tax manual.' "
                "Never attempt to answer from general knowledge. "
                "Answer DIRECTLY and only what the question asks. Do not repeat the question. Do not add preamble."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def ask_llm(question, langchain_embeddings, vector_store, tokenizer, model, reranker, k=6, max_new_tokens=512):
    if not is_query_relevant(question, langchain_embeddings):
        return (
            "I can only answer questions about the Manila Real Property Appraisal "
            "and Assessment Operations manual. Your question appears to be outside this scope."
        )

    context, source_docs = retrieve_context(question, vector_store, reranker, k=k)
    if context is None:
        return "The information is not available in the document."

    prompt = build_prompt(question, context, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_len:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer


# ── Streamlit UI ─────────────────────────────────────────────────

st.set_page_config(
    page_title="RPA & AO Manual Chatbot",
    page_icon="📋",
    layout="centered",
)

st.title("📋 Manual on Real Property Appraisal and Assessment Operations")
st.caption("RAG-powered Chatbot — answers are grounded exclusively in the BLGF manual.")
st.warning(
    "⚠️ This chatbot is for **informational purposes only** and is not a substitute for professional legal or financial advice.",
    icon="⚠️",
)

langchain_embeddings, vector_store, tokenizer, model, reranker = load_all()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

with st.sidebar:
    st.header("💡 Example Questions")
    examples = [
        "What is the purpose of the Manual on Real Property Appraisal and Assessment Operations?",
        "Who issued the BLGF manual and when?",
        "What are the different classifications of real property?",
        "How is the market value of a property determined?",
        "What is the assessment level for residential properties?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.prefill = ex

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

prefill = st.session_state.pop("prefill", None)
user_input = st.chat_input("Ask a question about the RPA & AO Manual...") or prefill

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching manual and generating answer..."):
            response = ask_llm(
                user_input,
                langchain_embeddings,
                vector_store,
                tokenizer,
                model,
                reranker,
            )
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

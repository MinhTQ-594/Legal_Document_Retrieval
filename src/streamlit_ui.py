import streamlit as st
st.set_page_config(page_title="ZaloAI Legal Chatbot", layout="wide")

# --- Your chatbot backend functions ---
from chatbot import (
    load_llm_model,
    load_original_sbert_model,
    initialize_all_retrievers,
    retrieve_documents_unified,
    generate_answer,
    RETRIEVERS
)

# --- Initialization ---
@st.cache_resource
def initialize():
    tokenizer, model, device = load_llm_model()
    sbert_model = load_original_sbert_model()
    initialize_all_retrievers(sbert_model)
    return tokenizer, model, device

st.title("🧑‍⚖️ ZaloAI Legal Chatbot (Chat UI)")

tokenizer, model, device = initialize()

if not RETRIEVERS:
    st.error("❌ No retrievers initialized. Please check your backend configuration.")
    st.stop()

import os
print("working directory before:", os.getcwd())
os.chdir('../src')
print("working directory after:", os.getcwd())

# --- Sidebar configuration ---
with st.sidebar:
    st.header("⚙️ Retrieval Options")
    retriever_keys = list(RETRIEVERS.keys())
    selected_retriever = st.selectbox(
        "Select Retriever",
        retriever_keys,
        format_func=lambda k: RETRIEVERS[k]['description']
    )
    if st.button("🔄 Reset Conversation"):
        st.session_state["chat_history"] = []

# --- Conversation state ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# --- Display past messages ---
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

# --- Chat input ---
user_input = st.chat_input("Ask your legal question...")
if user_input:
    # Store user message
    st.session_state["chat_history"].append({"role": "user", "text": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate bot response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving information and generating response..."):
            docs = retrieve_documents_unified(user_input, selected_retriever, top_n=3)
            answer = generate_answer(user_input, docs, model, tokenizer, device)

            if docs:
                doc_summary = "\n\n".join(
                    f"**{i+1}. {doc['title']} (Law: {doc['law_id']})**\n{doc['text']}"
                    for i, doc in enumerate(docs)
                )
                full_response = f"**Answer (by {selected_retriever}):**\n\n{answer}\n\n---\n**Relevant Documents:**\n\n{doc_summary}"
            else:
                full_response = "🤖 I couldn't find any relevant legal information for that question."

            st.markdown(full_response)
            st.session_state["chat_history"].append({"role": "assistant", "text": full_response})

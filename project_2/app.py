# app.py – minimal Streamlit UI for your RAG chatbot
import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# 1. Load the vector store you created earlier
embeddings = SentenceTransformerEmbeddings(model_name="thenlper/gte-small")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# 2. Build the chain (same as in the notebook)
SYSTEM_TEMPLATE = """
You are a **Customer Support Chatbot**. Use only the information in CONTEXT to answer.
If the answer is not in CONTEXT, respond with “I'm not sure from the docs.”

Rules:
1) Use ONLY the provided <context> to answer.
2) If the answer is not in the context, say: "I don't know based on the retrieved documents."
3) Be concise and accurate. Prefer quoting key phrases from the context.
4) When possible, cite sources as [source: source] using the metadata.

CONTEXT:
{context}

USER:
{question}
"""

prompt = PromptTemplate(
    template=SYSTEM_TEMPLATE,
    input_variables=["context", "question"]
)
llm = Ollama(model="gemma3:1b", temperature=0.1)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True
)

# 3. Streamlit UI
st.set_page_config(page_title="RAG Support Bot", page_icon="🤖")
st.title("Customer Support Bot")

if "chat_history" not in st.session_state:  # persist history across reruns
    st.session_state.chat_history = []

# Display existing conversation
for user_msg, bot_msg in st.session_state.chat_history:
    st.chat_message("user").write(user_msg)
    st.chat_message("assistant").write(bot_msg)

# Accept new question
if user_input := st.chat_input("Ask a question:"):
    st.chat_message("user").write(user_input)

    # Run chain
    response = chain({"question": user_input, "chat_history": st.session_state.chat_history})
    answer = response["answer"]

    st.session_state.chat_history.append((user_input, answer))
    st.chat_message("assistant").write(answer)

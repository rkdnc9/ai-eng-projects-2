# Quick Reference – RAG Chatbot Exercise

## 1. Document Loading

* **Local PDF** → `PyPDFLoader(file_path)`
* **Live web page** → `UnstructuredURLLoader([url])`\
  Fallback pattern: wrap URL call in `try/except` and load local PDF on failure.

## 2. Text Splitting

* `RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)`\
  Produces `docs` list ready for embedding.

## 3. Embeddings

* `SentenceTransformerEmbeddings(model_name="thenlper/gte-small")`\
  384-dim vectors, downloaded/cached automatically.

## 4. Vector Store

* `FAISS.from_documents(docs, embeddings)`\
  Save locally with `.save_local("faiss_index")`; load with `FAISS.load_local(..., allow_dangerous_deserialization=True)`.

## 5. Retriever

* `vectorstore.as_retriever(search_kwargs={"k": 8})`\
  `k` = # of chunks passed to LLM.

## 6. Prompt Template

* Build one `PromptTemplate` with `{context}` & `{question}` placeholders.
* Rules inside template steer the bot (cite sources, stay concise, etc.).

## 7. LLM

* `Ollama(model="gemma3:1b", temperature=0.1)`\
  Low temp → more factual; runs locally via Ollama.

## 8. RAG Chain

______

Python

```
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True
)
```

## 9. Chat Loop

* Pass `(question, chat_history)` dict; append `(q, a)` tuples to history list.

## 10. Streamlit UI (app.py)

* Load FAISS → build chain → `st.chat_input()`/`st.chat_message()` for web interface.

***

| **Step** | **Code / Value** | **Best-Practice Notes from Notebook** |
|---|---|---|
| **Loaders** | `PyPDFLoader` or `UnstructuredURLLoader` | Always wrap web-loader in `try/except`; fallback to local PDF so the notebook never crashes on a 404. |
| **Split** | `RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)` | 300/30 is the *sweet spot* for gte-small; bigger chunks > 512 tokens hurt retrieval quality. |
| **Embed** | `SentenceTransformerEmbeddings(model_name="thenlper/gte-small")` | 384-dim output; fully local after first download. |
| **Store** | `FAISS.save_local("faiss_index")` | Keep the folder name short and consistent; path is hard-coded in `app.py`. |
| **Retriever** | `as_retriever(search_kwargs={"k": 8})` | 8 chunks balance speed vs. context; raise only if you see missing answers. |
| **Prompt** | `SYSTEM_TEMPLATE` with rules 1-4 | Rules are intentionally *strict*; forces bot to say “I don’t know” instead of hallucinating. |
| **LLM** | `Ollama(model="gemma3:1b", temperature=0.1)` | Low temp + small model = fast, factual answers on CPU. |
| **Chain** | `ConversationalRetrievalChain.from_llm(...)` | Use `combine_docs_chain_kwargs` dict so future LangChain updates don’t break signature. |
| **Chat history** | list of `(q, a)` tuples | Always pass `chat_history` key (not `context`) when calling the chain. |
| **UI** | Streamlit `app.py` | Re-loads the *same* FAISS index; no re-building needed. |
| **Kernel tip** | `conda install ipykernel` while env is active, then register once. | Install `ipykernel` in *compute* envs only; keep a separate lightweight Jupyter env to avoid bloat. |

### Python One-Liners Worth Remembering
- `**kwargs` is just a dict; use it when you want to forward extra options without changing a function’s signature.  
- `chat_history` inside LangChain is a list of *string tuples* `(user_msg, assistant_msg)`—not raw documents.


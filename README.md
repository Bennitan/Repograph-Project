# 🧠 RepoGraph AI
**The Foundation for Autonomous Software Architecture.**

RepoGraph AI transforms raw source code into a searchable, optimized, and failure-predictive "Knowledge Graph." 

## 🚀 Core Platform Features
* **Semantic Brain (Vector RAG):** Search code by "intent" (e.g., "how is money handled?") rather than exact keywords.
* **Optimization Agent (God Mode):** Automatically refactor inefficient $O(n^3)$ algorithms into high-performance $O(n)$ code.
* **Blast Radius Simulator:** Visualize system-wide impact of a single function failure before it happens.
* **Knowledge Graph Infrastructure:** Powered by Neo4j to map complex dependencies and security risks.

## 🛠️ Tech Stack
* **AI Engine:** Google Gemini (Generative AI & Embeddings)
* **Graph Database:** Neo4j
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Frontend:** Streamlit

## ⚙️ Setup
1. Clone this repo.
2. Install requirements: `pip install -r requirements.txt`
3. Add your `GOOGLE_API_KEY` to `.streamlit/secrets.toml`.
4. Run: `streamlit run app.py`
# ⚽ Football Knowledge Base

A lightweight semantic search engine over football knowledge — built with Streamlit, TF-IDF, and scikit-learn. Deployable on Render's free tier with zero model downloads.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.56.0-red) ![Render](https://img.shields.io/badge/Deployed-Render-purple)

---

## What It Does

Type any football question and the app retrieves the most relevant passages from a curated knowledge base covering history, rules, players, competitions, tactics, and more.

- **Search**: Ask anything — "Who won the 2022 World Cup?", "How does the 4-3-3 work?", "What injuries are common in football?"
- **Chunking controls**: Adjust chunk size and overlap in real time to see how retrieval changes
- **Stats & Info**: Browse all source documents and understand the search methodology

---

## Topics Covered

| # | Topic |
|---|-------|
| 1 | History of Football |
| 2 | Rules & Gameplay |
| 3 | FIFA World Cup |
| 4 | Premier League |
| 5 | Lionel Messi |
| 6 | Cristiano Ronaldo |
| 7 | Tactics & Formations |
| 8 | Women's Football |
| 9 | Injuries & Safety |
| 10 | Champions League |
| 11 | Brazil National Team |
| 12 | Transfer Windows |

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| Vectorisation | TF-IDF via scikit-learn |
| Similarity search | Cosine similarity via scikit-learn |
| Hosting | Render (free tier) |

---

## Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/your-username/football-knowledge-base.git
cd football-knowledge-base
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

## Project Structure

```
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── render.yaml             # Render deployment config
└── .streamlit/
    └── config.toml         # Streamlit server config
```

---

## Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Render will auto-detect `render.yaml` and configure the service
5. Deploy — no environment variables needed

The `render.yaml` is already configured:
```yaml
services:
  - type: web
    name: football-knowledge-base
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
    envVars:
      - key: PYTHON_VERSION
        value: "3.11"
```

---

## Why TF-IDF Instead of a Vector Database

The original design used ChromaDB with a `sentence-transformers` embedding model. This caused Render free tier deployments to crash due to memory limits.

| | Original | Current |
|---|---|---|
| **Vector store** | ChromaDB | In-memory numpy array |
| **Embeddings** | `all-MiniLM-L6-v2` (downloaded at runtime) | TF-IDF via scikit-learn |
| **RAM usage** | 400–600MB+ | ~80MB |
| **Model download on startup** | Yes (~100MB) | None |
| **Render free tier compatible** | ❌ | ✅ |

**Trade-off:** TF-IDF is keyword-based rather than semantic. It matches on exact and related words rather than meaning. For this domain with consistent football terminology, it performs well in practice.

---

## 📄 License

MIT

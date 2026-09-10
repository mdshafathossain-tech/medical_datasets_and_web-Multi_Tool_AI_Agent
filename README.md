# 🩺 Medical Multi-Tool AI Agent

A **100% free**, multi-tool AI agent that answers natural-language questions about **Heart Disease, Cancer, and Diabetes** patient datasets using text-to-SQL — and falls back to a live web search for general medical knowledge questions. Built with LangChain, Groq's free LLM API, and a modern Streamlit chat interface.

> Ask "What's the average cholesterol for heart disease patients?" and get a real answer computed from actual SQL over your dataset — or ask "What causes high blood pressure?" and get a web-search-grounded answer. The agent decides which tool to use automatically.

---

## ✨ Key Features

- **🆓 100% Free Architecture** — No paid API keys required anywhere in the stack:
  - **Groq API** (free tier) powers the LLM for both routing decisions and SQL generation.
  - **DuckDuckGo** (no key required) powers general medical web search.
  - Runs entirely free both locally and when deployed to Streamlit Cloud.
- **🧠 Multi-Tool SQL Routing** — A single conversational agent automatically routes each question to the right tool:
  - `HeartDiseaseDBTool`, `CancerDBTool`, `DiabetesDBTool` — text-to-SQL over 3 separate SQLite databases.
  - `MedicalWebSearchTool` — for definitions, symptoms, causes, and treatments not covered by the datasets.
- **💬 Interactive Streamlit UI** — Modern chat interface with persistent history, clickable sample questions, live system status indicators, and an expandable **"Thinking & Tool Details"** panel showing exactly which tool was called and what it returned.
- **🛡️ Safety Guardrails** — Generated SQL is validated to allow only read-only `SELECT` statements before execution.
- **⚙️ Modular, Production-Style Code** — Cleanly separated into `db_setup.py`, `tools.py`, `agent.py`, and `app.py`.

---

## 🏗️ System Architecture / Workflow

```
                         ┌─────────────────────┐
                         │      User Query       │
                         │  (Streamlit Chat UI)  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Routing Agent        │
                         │ (LangChain create_agent│
                         │  + Groq LLM)           │
                         └──────────┬───────────┘
                                    │
              ┌─────────────┬──────┴──────┬─────────────┐
              ▼             ▼             ▼             ▼
      ┌───────────────┐┌───────────┐┌────────────┐┌──────────────────┐
      │HeartDiseaseDB  ││CancerDB   ││DiabetesDB  ││MedicalWebSearch  │
      │Tool            ││Tool       ││Tool        ││Tool              │
      │(text-to-SQL)   ││(text-to-  ││(text-to-   ││(DuckDuckGo)      │
      │                ││SQL)       ││SQL)        ││                  │
      └───────┬────────┘└─────┬─────┘└─────┬──────┘└────────┬─────────┘
              ▼                ▼             ▼                ▼
        heart.db          cancer.db     diabetes.db      Live Web Results
      (SQLite)           (SQLite)      (SQLite)
              │                │             │                │
              └────────────────┴─────────────┴────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Natural Language      │
                         │  Answer (+ tool trace) │
                         └─────────────────────┘
```

**Flow summary:**
1. The user asks a question in the Streamlit chat UI.
2. The routing agent (a Groq-powered LLM using LangChain's `create_agent`) reads the question and decides which tool(s) to call, guided by an explicit system prompt.
3. Dataset questions trigger a text-to-SQL chain that converts the question into a `SELECT` query, validates it, runs it against the relevant SQLite database, and returns the result.
4. General medical questions trigger a free DuckDuckGo web search instead.
5. The agent phrases the final answer in plain English and the UI displays it, along with an optional expandable trace of exactly which tool(s) were called.

---

## 📂 Folder Structure

```
medical-multi-tool-agent/
├── data/                     # Source CSV files (not committed if large/private)
│   ├── heart.csv
│   ├── cancer.csv
│   └── diabetes.csv
├── databases/                 # Auto-generated SQLite databases (git-ignored)
│   ├── heart.db
│   ├── cancer.db
│   └── diabetes.db
├── db_setup.py                # CSV cleaning + SQLite database creation
├── tools.py                   # 4 LangChain tools (3 SQL + 1 web search)
├── agent.py                   # Main routing agent (LangChain create_agent + Groq)
├── app.py                     # Streamlit chat UI
├── requirements.txt           # Pinned Python dependencies
├── .gitignore
├── .env                        # Local secrets (GROQ_API_KEY) — never committed
└── README.md
```

---

## 🚀 Local Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your CSV datasets
Place `heart.csv`, `cancer.csv`, and `diabetes.csv` inside a `data/` folder in the project root.

### 5. Get a free Groq API key
1. Go to [console.groq.com](https://console.groq.com) and sign up (free).
2. Navigate to **API Keys → Create Key**.
3. Copy the key.

### 6. Create your `.env` file
In the project root, create a file named `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 7. Build the databases (optional — the app does this automatically too)
```bash
python db_setup.py
```

### 8. Run the app
```bash
streamlit run app.py
```
The app will open automatically at `http://localhost:8501`.

---

## ☁️ Streamlit Cloud Deployment Guide

### 1. Push your project to GitHub
Make sure `.env` is **not** committed (it's excluded via `.gitignore`) — only `requirements.txt`, your source files, and your `data/*.csv` files (if you're comfortable making them public; otherwise see the note below) should be pushed. See the [Git commands](#-git-commands-to-publish-this-project) section below.

> **Note on data privacy:** If your CSVs contain sensitive or licensed data, do not commit them to a public repo. Instead, either use a private repository, or add a step to download the CSVs from a private source at app startup.

### 2. Create a new Streamlit Cloud app
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **"New app"**.
3. Select your repository, branch (`main`), and set **Main file path** to `app.py`.

### 3. Configure `GROQ_API_KEY` in Streamlit Secrets
1. In your new app's dashboard, go to **⚙️ Settings → Secrets**.
2. Add the following (in TOML format):
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
3. Click **Save**.

Streamlit Cloud injects these secrets as environment variables at runtime, so `os.getenv("GROQ_API_KEY")` in `tools.py`/`agent.py` will pick it up automatically — no code changes needed.

### 4. Deploy
Click **"Deploy"**. Streamlit Cloud will:
- Install everything from `requirements.txt`.
- Run `app.py`, which automatically calls `build_all_databases()` on startup to generate the SQLite databases from your `data/*.csv` files if they don't already exist.

### 5. Verify
Once deployed, check the sidebar status indicators:
- 🟢 Databases Ready
- ⚡ LLM Engine: Groq API

If either shows an error, check the **"Manage app" → logs** panel in Streamlit Cloud for the exact error message.

---

## 🧭 Git Commands to Publish This Project

Run these from your project's root folder (where `app.py` lives):

```bash
# 1. Initialize a new Git repository
git init

# 2. Stage all files (respecting .gitignore)
git add .

# 3. Create your first commit
git commit -m "Initial commit: Multi-Tool Medical AI Agent"

# 4. Rename the default branch to main (if not already)
git branch -M main

# 5. Connect to your GitHub repository
#    (create an empty repo on GitHub first, then copy its URL)
git remote add origin https://github.com/<your-username>/<your-repo-name>.git

# 6. Push to GitHub
git push -u origin main
```

For subsequent updates:
```bash
git add .
git commit -m "Describe your changes here"
git push
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Agent Framework | LangChain (`create_agent`) |
| LLM | Groq API (`openai/gpt-oss-20b`, free tier) |
| Database | SQLite (via SQLAlchemy) |
| Web Search | DuckDuckGo (`ddgs`) |
| Data Processing | Pandas |

---

## ⚠️ Known Limitations

- Free-tier LLMs (like the Groq model used here) are less reliable at complex multi-join SQL generation than paid GPT-4-class models — simple aggregate/filter questions work best.
- The SQL safety guard blocks non-`SELECT` statements but is not a full SQL parser; treat this as a demo/learning project rather than a production-hardened system handling untrusted multi-user input.
- DuckDuckGo search results can occasionally be rate-limited under heavy use.

---

## 📄 License

This project is provided as-is for educational purposes. Add your preferred license (e.g., MIT) here.

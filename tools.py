"""
tools.py
--------
Defines the 4 tools used by the Multi-Tool Medical AI Agent, using
100% FREE components so this project can run locally and on
Streamlit Cloud without any paid API keys:

    1. HeartDiseaseDBTool  - text-to-SQL over heart.db
    2. CancerDBTool        - text-to-SQL over cancer.db
    3. DiabetesDBTool      - text-to-SQL over diabetes.db
    4. MedicalWebSearchTool - DuckDuckGo search (no key required)

LLM backend:
    - Primary:  Groq (free tier) via `langchain-groq`, model
                "openai/gpt-oss-20b" -- fast, free, and works on
                Streamlit Cloud (cloud-hosted, no local install needed).
    - Fallback: Local Ollama via `langchain-ollama`, used automatically
                if no GROQ_API_KEY is found in the environment. Useful
                for fully offline development, but note Ollama requires
                a local model server, so it will NOT work as-is on
                Streamlit Cloud.

Import ALL_TOOLS from this module in agent.py to wire everything
into the AgentExecutor.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file in the project root (e.g. GROQ_API_KEY=...)
# into the environment. Without this call, os.getenv("GROQ_API_KEY") below
# will return None even if the key is correctly set in .env, since nothing
# else in this script reads that file.
load_dotenv()

from langchain_community.utilities import SQLDatabase
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool
from langchain_core.language_models import BaseChatModel

# As of LangChain v1.0, `create_sql_query_chain` moved out of the core
# `langchain` package into the separate `langchain-classic` package
# (`pip install langchain-classic`). This try/except keeps tools.py working
# whether you're on LangChain 1.0+ (langchain_classic) or an older
# pre-1.0 install (langchain.chains) without you needing to know which.
try:
    from langchain_classic.chains import create_sql_query_chain
except ImportError:
    from langchain.chains import create_sql_query_chain

from db_setup import DATASET_CONFIG  # reuse the same config as db_setup.py


# --------------------------------------------------------------------------
# LLM factory: free Groq (cloud) by default, local Ollama as fallback
# --------------------------------------------------------------------------

def get_llm() -> BaseChatModel:
    """
    Return a chat LLM instance for SQL generation, using only free options.

    Selection logic:
        1. If GROQ_API_KEY is set in the environment -> use ChatGroq.
           Model defaults to "openai/gpt-oss-20b" (free tier, fast,
           tool-calling capable). Override with the GROQ_MODEL env var,
           e.g. GROQ_MODEL="llama-3.3-70b-versatile".
        2. Otherwise -> use ChatOllama against a local Ollama server.
           Model defaults to "llama3.1". Override with OLLAMA_MODEL.
           Requires Ollama installed and running locally
           (https://ollama.com), and the model pulled beforehand:
               ollama pull llama3.1

    temperature=0 keeps SQL generation deterministic rather than
    "creative", which matters when generating queries against real data.

    Note on model choice: Groq deprecated "llama-3.1-8b-instant" and
    "llama-3.3-70b-versatile" for free/developer-tier usage in mid-2026.
    "openai/gpt-oss-20b" is the currently supported free-tier replacement
    at time of writing -- check https://console.groq.com/docs/deprecations
    if you hit a model-not-found error later, and update GROQ_MODEL.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")

    if groq_api_key:
        from langchain_groq import ChatGroq

        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        return ChatGroq(model=model_name, temperature=0, api_key=groq_api_key)

    # --- Fallback: local Ollama (fully free, fully offline) ---
    print(
        "[info] GROQ_API_KEY not found -- attempting local Ollama fallback. "
        "Note: this fallback will NOT work on Streamlit Cloud."
    )
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise RuntimeError(
            "No usable LLM backend found. GROQ_API_KEY is not set in the "
            "environment, and the optional local-Ollama fallback package "
            "(langchain-ollama) is not installed.\n\n"
            "If you are running on Streamlit Cloud: go to your app's "
            "Settings -> Secrets and add:\n"
            '    GROQ_API_KEY = "your_groq_api_key_here"\n'
            "then reboot the app (saving secrets alone does not always "
            "restart the running process).\n\n"
            "If you are running locally: make sure a `.env` file exists in "
            "the project root containing GROQ_API_KEY=your_key, or install "
            "`langchain-ollama` and run a local Ollama server instead."
        ) from exc

    model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
    return ChatOllama(model=model_name, temperature=0)


# Shared LLM instance used for SQL generation across all 3 DB tools.
SQL_LLM = get_llm()


# --------------------------------------------------------------------------
# Safety guard: only allow read-only (SELECT) queries
# --------------------------------------------------------------------------

def _is_safe_select_query(sql_query: str) -> bool:
    """
    Basic guardrail to ensure the LLM-generated SQL is a read-only SELECT.

    This is not a full SQL parser/sanitizer -- it's a first line of defense
    appropriate for a single-user local demo. For production, consider
    running queries against a read-only DB connection/user as well.
    """
    normalized = sql_query.strip().lower()
    forbidden_keywords = (
        "insert", "update", "delete", "drop", "alter",
        "create", "replace", "truncate", "attach", "pragma",
    )
    starts_with_select = normalized.startswith("select")
    contains_forbidden = any(kw in normalized for kw in forbidden_keywords)
    return starts_with_select and not contains_forbidden


# --------------------------------------------------------------------------
# Factory: build a text-to-SQL tool for a given dataset
# --------------------------------------------------------------------------

def make_sql_tool(dataset_name: str, tool_name: str, description: str) -> Tool:
    """
    Build a LangChain Tool that converts a natural-language question into
    a SQL query against the given dataset's SQLite database, executes it,
    and returns the raw result rows as a string.

    Args:
        dataset_name: Key into DATASET_CONFIG ('heart', 'cancer', 'diabetes').
        tool_name: Name shown to the agent (used in routing decisions).
        description: Description shown to the agent -- this is what the LLM
            reads to decide *when* to call this tool, so be specific about
            what kind of questions it should handle.

    Returns:
        A configured langchain_core.tools.Tool instance.
    """
    config = DATASET_CONFIG[dataset_name]
    db_path: Path = config["db"]
    table_name: str = config["table"]

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at '{db_path}'. Run `python db_setup.py` first."
        )

    # SQLDatabase wraps the SQLite file via SQLAlchemy and exposes schema
    # info that create_sql_query_chain uses to build accurate prompts.
    db = SQLDatabase.from_uri(
        f"sqlite:///{db_path}",
        include_tables=[table_name],
        sample_rows_in_table_info=3,  # gives the LLM a few example rows
    )

    sql_generation_chain = create_sql_query_chain(SQL_LLM, db)

    def _run_query(question: str) -> str:
        """Convert `question` to SQL, validate it, execute it, return results."""
        try:
            sql_query = sql_generation_chain.invoke({"question": question})
        except Exception as exc:  # noqa: BLE001 - surface LLM/chain errors to the agent
            return f"Error generating SQL query: {exc}"

        # create_sql_query_chain (and smaller open-weight models especially)
        # sometimes wraps the query in markdown code fences, adds a
        # trailing "SQLQuery:" prefix, or -- as seen with some free Groq
        # models -- echoes part of the input question before the SQL
        # (e.g. "Question: How many patients...\nSELECT COUNT(*) ..."). We
        # defensively strip all of that by locating the actual SELECT
        # statement rather than trusting the whole output is clean SQL.
        raw_output = (
            sql_query.replace("```sql", "")
            .replace("```", "")
            .replace("SQLQuery:", "")
            .strip()
        )

        # Find where the real SQL starts (case-insensitive) and discard
        # any echoed question/preamble text before it.
        lowered = raw_output.lower()
        select_index = lowered.find("select")
        if select_index == -1:
            return (
                "Refused to execute: no SELECT statement found in the "
                f"model's output. Raw output was: {raw_output}"
            )
        cleaned_query = raw_output[select_index:].strip()

        # Some models also add trailing commentary after the query --
        # keep only up to the first semicolon if one is present.
        if ";" in cleaned_query:
            cleaned_query = cleaned_query.split(";")[0].strip() + ";"

        if not _is_safe_select_query(cleaned_query):
            return (
                "Refused to execute a potentially unsafe or non-SELECT query. "
                f"Generated query was: {cleaned_query}"
            )

        try:
            result = db.run(cleaned_query)
        except Exception as exc:  # noqa: BLE001 - surface DB errors to the agent
            return f"Error executing SQL query '{cleaned_query}': {exc}"

        return (
            f"SQL query used: {cleaned_query}\n"
            f"Result: {result if result else 'No rows returned.'}"
        )

    return Tool(
        name=tool_name,
        description=description,
        func=_run_query,
    )


# --------------------------------------------------------------------------
# Build the 3 dataset-specific SQL tools
# --------------------------------------------------------------------------

HeartDiseaseDBTool = make_sql_tool(
    dataset_name="heart",
    tool_name="HeartDiseaseDBTool",
    description=(
        "Use this tool to answer questions about the Heart Disease dataset "
        "(patient records with fields like age, sex, cholesterol, blood "
        "pressure, chest pain type, and a 'target' indicating presence of "
        "heart disease). Input should be a natural language question about "
        "this data, e.g. 'What is the average cholesterol for patients over 50?'"
    ),
)

CancerDBTool = make_sql_tool(
    dataset_name="cancer",
    tool_name="CancerDBTool",
    description=(
        "Use this tool to answer questions about the Cancer dataset "
        "(patient/tumor records with measurements like radius, texture, "
        "smoothness, and a diagnosis label, e.g. malignant/benign). Input "
        "should be a natural language question about this data, e.g. "
        "'How many malignant cases are in the dataset?'"
    ),
)

DiabetesDBTool = make_sql_tool(
    dataset_name="diabetes",
    tool_name="DiabetesDBTool",
    description=(
        "Use this tool to answer questions about the Diabetes dataset "
        "(patient records with fields like glucose level, BMI, insulin, "
        "age, and an outcome indicating diabetes diagnosis). Input should "
        "be a natural language question about this data, e.g. 'What is "
        "the average BMI for patients with diabetes?'"
    ),
)


# --------------------------------------------------------------------------
# Web search tool: DuckDuckGo only (free, no API key required)
# --------------------------------------------------------------------------

def _build_web_search_tool() -> Tool:
    """
    Build the general-purpose medical web search tool using DuckDuckGo.

    No API key required, which keeps the whole project free to run
    both locally and on Streamlit Cloud.
    """
    duckduckgo_search = DuckDuckGoSearchRun()

    description = (
        "Use this tool to answer general medical questions that are NOT "
        "answerable from the Heart Disease, Cancer, or Diabetes datasets -- "
        "for example, questions about symptoms, treatments, drug "
        "interactions, or general medical knowledge. Input should be a "
        "natural language search query."
    )

    def _run_duckduckgo(query: str) -> str:
        try:
            return duckduckgo_search.invoke(query)
        except Exception as exc:  # noqa: BLE001
            return f"DuckDuckGo search failed: {exc}"

    return Tool(
        name="MedicalWebSearchTool",
        description=description,
        func=_run_duckduckgo,
    )


MedicalWebSearchTool = _build_web_search_tool()


# --------------------------------------------------------------------------
# Export all tools for use by the agent
# --------------------------------------------------------------------------

ALL_TOOLS = [
    HeartDiseaseDBTool,
    CancerDBTool,
    DiabetesDBTool,
    MedicalWebSearchTool,
]


if __name__ == "__main__":
    # Quick manual smoke test -- run `python tools.py` to sanity-check
    # that all 4 tools load correctly and can be invoked.
    print("Loaded tools:")
    for tool in ALL_TOOLS:
        print(f"  - {tool.name}: {tool.description[:70]}...")

    print("\n--- Test: HeartDiseaseDBTool ---")
    print(HeartDiseaseDBTool.func("How many patients are in the dataset?"))
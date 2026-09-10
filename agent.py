"""
agent.py
--------
Builds the main Multi-Tool Medical AI Agent that routes natural-language
questions to the correct tool:

    - HeartDiseaseDBTool / CancerDBTool / DiabetesDBTool for numerical,
      statistical, or row-level questions about the 3 medical datasets.
    - MedicalWebSearchTool for general medical knowledge (definitions,
      symptoms, causes, treatments) not answerable from the datasets.

Uses LangChain 1.0's `create_agent` (LangGraph-based agent loop) with a
free ChatGroq model -- no paid APIs required.

Import `query_agent` from this module in app.py (Streamlit) to answer
user questions.
"""

import os

from dotenv import load_dotenv

# Load GROQ_API_KEY (and anything else) from a .env file in the project root.
load_dotenv()

from langchain.agents import create_agent
from langchain_groq import ChatGroq

from tools import ALL_TOOLS


# --------------------------------------------------------------------------
# Agent LLM
# --------------------------------------------------------------------------
# Kept separate from tools.py's SQL_LLM: this model handles routing and
# final-answer phrasing, while SQL_LLM (in tools.py) only handles
# text-to-SQL generation. Separating them means you can tune/swap one
# without touching the other -- e.g. try a larger Groq model here for
# better routing decisions while keeping SQL generation on a smaller one.
AGENT_MODEL_NAME = os.getenv("GROQ_AGENT_MODEL", "openai/gpt-oss-20b")

AGENT_LLM = ChatGroq(
    model=AGENT_MODEL_NAME,
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)


# --------------------------------------------------------------------------
# System prompt: explicit routing instructions
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a Medical Multi-Tool AI Assistant. You have access to 4 tools and \
must choose the correct one(s) to answer the user's question. Never answer \
from your own general knowledge if a tool can answer the question instead \
-- always prefer calling a tool.

ROUTING RULES:

1. HeartDiseaseDBTool
   Use for numerical, statistical, or row-level questions about the Heart \
   Disease patient dataset -- e.g. averages, counts, comparisons, or \
   filters involving fields like age, sex, cholesterol, blood pressure, \
   chest pain type, or heart disease diagnosis ('target').

2. CancerDBTool
   Use for numerical, statistical, or row-level questions about the \
   Cancer patient dataset -- e.g. averages, counts, comparisons, or \
   filters involving tumor measurements (radius, texture, smoothness, \
   etc.) or diagnosis labels (malignant/benign).

3. DiabetesDBTool
   Use for numerical, statistical, or row-level questions about the \
   Diabetes patient dataset -- e.g. averages, counts, comparisons, or \
   filters involving fields like glucose level, BMI, insulin, age, or \
   diabetes outcome.

4. MedicalWebSearchTool
   Use for general medical knowledge questions that are NOT about the \
   specific numbers/rows in the 3 datasets above -- e.g. definitions, \
   symptoms, causes, risk factors, treatments, prevention, or "what is \
   X" / "how does X work" style questions.

DECISION GUIDE:
- If the question asks "how many", "average", "what percentage", \
  "compare", "list patients where...", or references specific dataset \
  columns/values -> use the matching DB tool (Heart/Cancer/Diabetes).
- If the question asks "what is", "what causes", "how is X treated", \
  "what are the symptoms of", or is about medical knowledge in general \
  -> use MedicalWebSearchTool.
- If a question could touch more than one dataset, call each relevant \
  DB tool separately and combine the results in your answer.
- If you are unsure which dataset a statistical question refers to, ask \
  the user for clarification instead of guessing.

RESPONSE STYLE:
- After calling the appropriate tool(s), answer the user in clear, plain \
  English. Do not just paste raw SQL or tool output -- summarize it into \
  a direct, natural-language answer.
- If a tool returns an error or no data, tell the user plainly rather \
  than making up an answer.
"""


# --------------------------------------------------------------------------
# Build the agent
# --------------------------------------------------------------------------
# create_agent (LangChain 1.0+) builds a LangGraph agent loop: the model
# is called, it may choose to call one or more tools, the tool results are
# fed back in, and the loop repeats until the model responds without
# calling a tool.
agent = create_agent(
    model=AGENT_LLM,
    tools=ALL_TOOLS,
    system_prompt=SYSTEM_PROMPT,
)


# --------------------------------------------------------------------------
# Clean wrapper function for use by app.py
# --------------------------------------------------------------------------

def query_agent(question: str) -> str:
    """
    Run a natural-language question through the routing agent and return
    a plain-text answer.

    Args:
        question: The user's natural language question.

    Returns:
        The agent's final natural-language response as a string. If
        something goes wrong, returns a readable error message instead
        of raising, so callers (e.g. a Streamlit UI) can display it
        directly without needing their own try/except.
    """
    if not question or not question.strip():
        return "Please enter a question."

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
    except Exception as exc:  # noqa: BLE001 - surface agent errors to the caller
        return f"Sorry, something went wrong while processing your question: {exc}"

    messages = result.get("messages", [])
    if not messages:
        return "Sorry, the agent did not return a response."

    # The final message in the list is the agent's last AI response
    # (after any tool calls have been resolved).
    final_message = messages[-1]

    # AIMessage objects expose `.content`; some result shapes may already
    # be plain dicts, so handle both defensively.
    content = getattr(final_message, "content", None)
    if content is None and isinstance(final_message, dict):
        content = final_message.get("content")

    if not content:
        return "Sorry, the agent did not return a readable response."

    return content


# --------------------------------------------------------------------------
# Verbose wrapper: returns the answer AND a trace of tool calls made
# --------------------------------------------------------------------------
# Used by app.py to power the "Thinking & Tool Details" expander, so users
# can see which tool(s) the agent chose and what they returned.

def query_agent_with_trace(question: str) -> tuple[str, list[dict]]:
    """
    Run a natural-language question through the routing agent and return
    both the final answer and a step-by-step trace of tool calls made.

    Args:
        question: The user's natural language question.

    Returns:
        A tuple of (answer, trace) where:
            answer: The agent's final natural-language response as a string.
            trace: A list of dicts describing each tool call/result, e.g.
                [{"type": "tool_call", "tool": "HeartDiseaseDBTool",
                  "input": {"question": "..."}},
                 {"type": "tool_result", "tool": "HeartDiseaseDBTool",
                  "output": "SQL query used: ... Result: ..."}]
                Empty if the agent answered without calling any tool, or
                if something went wrong before a trace could be built.
    """
    if not question or not question.strip():
        return "Please enter a question.", []

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
    except Exception as exc:  # noqa: BLE001 - surface agent errors to the caller
        return (
            f"Sorry, something went wrong while processing your question: {exc}",
            [],
        )

    messages = result.get("messages", [])
    if not messages:
        return "Sorry, the agent did not return a response.", []

    trace: list[dict] = []
    for msg in messages:
        msg_type = type(msg).__name__

        # AIMessage with tool_calls -> the agent decided to call tool(s)
        if msg_type == "AIMessage" and getattr(msg, "tool_calls", None):
            for tool_call in msg.tool_calls:
                trace.append(
                    {
                        "type": "tool_call",
                        "tool": tool_call.get("name", "unknown_tool"),
                        "input": tool_call.get("args", {}),
                    }
                )

        # ToolMessage -> the result returned by a tool
        elif msg_type == "ToolMessage":
            tool_name = getattr(msg, "name", "unknown_tool")
            output = str(getattr(msg, "content", ""))
            trace.append(
                {
                    "type": "tool_result",
                    "tool": tool_name,
                    # Truncate long outputs so the UI stays readable.
                    "output": output[:800] + ("..." if len(output) > 800 else ""),
                }
            )

    final_message = messages[-1]
    content = getattr(final_message, "content", None)
    if content is None and isinstance(final_message, dict):
        content = final_message.get("content")

    if not content:
        content = "Sorry, the agent did not return a readable response."

    return content, trace


# --------------------------------------------------------------------------
# Manual test block
# --------------------------------------------------------------------------

if __name__ == "__main__":
    test_questions = [
        # Should route to HeartDiseaseDBTool
        "What is the average cholesterol level in the heart disease dataset?",
        # Should route to CancerDBTool
        "How many malignant cases are in the cancer dataset?",
        # Should route to DiabetesDBTool
        "What is the average BMI for patients with diabetes in the dataset?",
        # Should route to MedicalWebSearchTool
        "What are the common symptoms of type 2 diabetes?",
    ]

    for i, question in enumerate(test_questions, start=1):
        print(f"\n{'=' * 70}")
        print(f"Test {i}: {question}")
        print("=" * 70)
        answer = query_agent(question)
        print(f"Answer: {answer}")

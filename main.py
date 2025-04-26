# main.py

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict

from agents.researcher import research_graph
from agents.drafter import drafter_graph

# Load environment variables
load_dotenv()

# 1. Define the State schema
class QAState(TypedDict):
    question: str
    research_data: str
    final_answer: str

# 2. Node: Run Research Agent
def run_research_graph(state: QAState) -> QAState:
    output = research_graph.invoke({"question": state["question"]})
    return {
        "question": state["question"],
        "research_data": output["response"],
        "final_answer": ""
    }

# 3. Node: Run Drafter Agent
def run_drafter_graph(state: QAState) -> QAState:
    output = drafter_graph.invoke({"research_data": state["research_data"]})
    return {
        "question": state["question"],
        "research_data": state["research_data"],
        "final_answer": output["final_draft"]
    }

# 4. Initialize Graph
builder = StateGraph(QAState)

# 5. Add nodes
builder.add_node("research", run_research_graph)
builder.add_node("draft", run_drafter_graph)

# 6. Set entry point and edges
builder.set_entry_point("research")
builder.add_edge("research", "draft")
builder.add_edge("draft", END)

# 7. Compile Graph
graph = builder.compile()

# 8. Run Example
if __name__ == "__main__":
    query = "What are the recent advancements in quantum computing as of April 2025?"
    initial_state = {
        "question": query,
        "research_data": "",
        "final_answer": ""
    }
    
    result = graph.invoke(initial_state)
    
    print("\n✅ Final Answer:\n")
    print(result["final_answer"])

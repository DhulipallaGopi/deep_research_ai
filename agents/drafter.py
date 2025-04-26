# agents/drafter.py

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.graph import StateGraph
from typing import TypedDict

# 1. Drafter State Schema
class DraftState(TypedDict):
    research_data: str
    final_draft: str

# 2. Groq LLM Setup
llm = ChatGroq(
    model="Llama3-70b-8192", 
    temperature=0.5,
    groq_api_key="gsk_ekXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXvEYcp7y9"  #
)

# 3. Drafter Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant summarizing and drafting clear responses."),
    ("user", "Here is the research data:\n{research_data}\n\nNow write a concise and well-structured answer.")
])

# 4. Drafter Agent
def get_drafter_agent() -> Runnable:
    return prompt | llm

# 5. Node Function
def invoke_drafter_agent(state: DraftState) -> DraftState:
    drafter = get_drafter_agent()
    response = drafter.invoke({"research_data": state["research_data"]})
    return {
        "research_data": state["research_data"],
        "final_draft": response.content if hasattr(response, "content") else str(response)
    }

# 6. Build Drafter Graph
def build_drafter_graph() -> Runnable:
    builder = StateGraph(DraftState)
    builder.add_node("drafter", RunnableLambda(invoke_drafter_agent))
    builder.set_entry_point("drafter")
    return builder.compile()

# 7. Expose Graph
drafter_graph = build_drafter_graph()

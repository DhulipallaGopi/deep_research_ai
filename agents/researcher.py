

from langchain_groq import ChatGroq  
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent, Tool
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.graph import StateGraph
from typing import TypedDict

#1. Define Researcher State Schema
class ResearchState(TypedDict):
    question: str
    response: str

# 2. Groq LLM Setup
llm = ChatGroq(
    model="Llama3-70b-8192",  # or "Mixtral-8x7b-32768" if you want
    temperature=0.3,
    groq_api_key="gsk_ekxuxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxp7y9") #use the your api key for the getting puts

# 3. Tavily Web Search Tool Setup
tavily_tool = TavilySearchResults(tavily_api_key="tvly-deXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXFryEiCcXQ")  
tools = [
    Tool(
        name="web-search",
        func=tavily_tool.run,
        description="Useful for web search queries."
    )
]

# 4. Research Agent Function
def get_research_agent() -> Runnable:
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type="chat-zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True,
    )
    return agent

# 5. Node Function to Invoke Research Agent
def invoke_research_agent(state: ResearchState) -> ResearchState:
    agent = get_research_agent()
    result = agent.invoke(state["question"])
    return {
        "question": state["question"],
        "response": result
    }

# 6. Build Research Graph
def build_research_graph() -> Runnable:
    builder = StateGraph(ResearchState)
    builder.add_node("research", RunnableLambda(invoke_research_agent))
    builder.set_entry_point("research")
    return builder.compile()

# 7. Expose Graph
research_graph = build_research_graph()

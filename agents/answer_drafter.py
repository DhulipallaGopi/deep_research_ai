# agents/answer_drafter.py

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

# 1. Groq LLM Setup
llm = ChatGroq(
    model="Llama3-70b-8192",
    temperature=0.3,
    groq_api_key="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"  # 🔥 Replace with your actual key
)

# 2. Answer Drafter Prompt
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant summarizing and drafting answers from research."),
    ("user", "Here is the research data:\n{research_data}\n\nPlease provide a detailed response.")
])

# 3. Answer Drafter Agent
def get_answer_drafter_agent() -> Runnable:
    return answer_prompt | llm

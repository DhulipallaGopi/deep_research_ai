Overview
Deep Research AI leverages the power of language models and stateful agents to help summarize, draft, and conduct web searches to assist in research projects. It comprises two core agents:

Drafter Agent: This agent summarizes research data and drafts clear responses.

Researcher Agent: This agent queries the web using the Tavily search tool to gather additional research data and assists in drafting responses.

Key Components
1. Drafter Agent
The Drafter Agent works with the ChatGroq language model and a custom-built prompt. It processes the research data and generates a concise and well-structured answer.

Drafter State Schema
The DraftState schema contains the following fields:

research_data: Raw research data to be summarized.

final_draft: The drafted summary generated by the Drafter Agent.

Setup
The Drafter Agent is powered by the Llama3-70b-8192 model from Groq and uses the following components:

Prompt Template: A structured prompt asking the agent to summarize and provide a response.

LLM Setup: Configuration of the Groq language model with temperature and API key.

Graph Definition
The Drafter Graph is constructed using StateGraph and processes the DraftState. It is triggered by invoking the agent with the research data and returns the final draft.

Example Usage:
python
Copy
Edit
drafter_graph = build_drafter_graph()
2. Researcher Agent
The Researcher Agent integrates the Tavily Search Tool with the Groq LLM to help gather information from the web. It processes queries, conducts web searches, and provides research insights.

Research State Schema
The ResearchState schema includes:

question: The query or research question.

response: The result or information fetched from the web.

Setup
Tavily Web Search Tool: Used to fetch research results from the web.

LLM Setup: Configured using the Groq API and the Llama3-70b-8192 model.

Agent Initialization: The Researcher Agent uses the chat-zero-shot-react-description agent type for flexible queries.

Graph Definition
The Researcher Graph is also built using StateGraph. It processes the ResearchState and returns responses to research questions.

Example Usage:
python
Copy
Edit
research_graph = build_research_graph()
Setup Instructions
Install Dependencies: To use Deep Research AI, install the required Python libraries:

bash
Copy
Edit
pip install langchain langchain-groq langgraph
Groq API Key: Replace the placeholder groq_api_key with your actual API key for the Groq language model.

Tavily API Key: Similarly, replace the tavily_api_key with your Tavily API key for web search functionality.

Running the Agents: You can now run the Drafter and Researcher agents as per your research needs:

python
Copy
Edit
drafter_result = drafter_graph.invoke(state)
researcher_result = research_graph.invoke(state)
Important Notes
Ensure that your Groq API keys and Tavily API keys are properly secured.

Follow GitHub’s push protection guidelines to avoid pushing sensitive data (e.g., API keys).

Consider version control practices such as .gitignore to exclude sensitive files like __pycache__ or any other files containing keys.

Contributing
Feel free to contribute to the repository by forking it and submitting a pull request. Make sure to follow good coding practices and keep sensitive data out of commits.
#output
## Output Images
## Output Images

<p align="center">
  <img src="assests/output1.jpg" width="200" height="250">
  <img src="assests/output2.jpg" width="200" height="250">
  <img src="assests/output3.jpg" width="200" height="250">
  <img src="assests/output4.jpg" width="200" height="250">
  <img src="assests/output5.jpg" width="200" height="250">
</p>





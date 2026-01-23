from google.adk.agents import Agent

from wikipedia import search, page

def wikipedia_search(query: str) -> str:
    """Searches Wikipedia for a given query."""
    pages = search(query, results=1)
    if pages:
        return page(pages[0], auto_suggest=False).content
    else:
        return ""

MODEL = "gemini-3-flash-preview"

# --- Researcher Agent ---
researcher = Agent(
    name="researcher",
    model=MODEL,
    description="Gathers information on a topic using Wikipedia.",
    instruction="""
    You are an expert researcher. Your goal is to find comprehensive and accurate information on the user's topic.
    Use the `wikipedia_search` tool to find relevant information.
    Return detailed information about the topic provided by the tool. Format your response in markdown.
    If you receive feedback that your research is insufficient, use the feedback to refine your next search.
    """,
    tools=[wikipedia_search],
)

root_agent = researcher


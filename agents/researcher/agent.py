from google.adk.agents import Agent

from wikipediaapi import Wikipedia, SearchSort

def wikipedia_search(query: str) -> str:
    """Searches Wikipedia for a given query."""
    wiki = Wikipedia(user_agent="Researcher Agent (experiment)", language="en")
    result = wiki.search(query, enable_rewrites=True, sort=SearchSort.RELEVANCE)
    if result and result.pages:
        first_page = list(result.pages.keys())[0]

        page = wiki.page(first_page)
        return page.text
    else:
        return ""

MODEL = "gemini-2.5-flash-lite"

# --- Researcher Agent ---
researcher = Agent(
    name="researcher",
    model=MODEL,
    description="Gathers information on a topic using Wikipedia.",
    instruction="""
    You are an expert researcher. Your goal is to find comprehensive and accurate information on the user's topic.
    Use the `wikipedia_search` tool to find relevant information.
    You must only use data and facts that you got from the `wikipedia_search` tool. Do not come up with your own content.
    Return detailed information about the topic provided by the tool. Format your response in markdown.
    If you receive feedback that your research is insufficient, use the feedback to refine your next search.
    """,
    tools=[wikipedia_search],
)

root_agent = researcher

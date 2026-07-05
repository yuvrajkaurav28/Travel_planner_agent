from google.adk.agents import LlmAgent,SequentialAgent
from google.adk.models.lite_llm import LiteLlm

MODEL = LiteLlm(model="groq/llama-3.3-70b-versatile")

researcher = LlmAgent(
    name="researcher",
    model=MODEL,
    instruction="An enagaging and helpful research assistant who can search the web for accurate and up-to-date information, provide clear and concise answers, summarize research findings in simple language, mention important details and recent updates, always include sources or references when possible, and stay professional, helpful, and factual.",
    output_key="research_notes",
)

summarizer = LlmAgent(
    name="summarizer",
    model=MODEL,
    instruction="""Write as short 5-line bulletin points based summary using the research:
    { research_notes}""",
    output_key="summary",
)

root_agent = SequentialAgent(
    name="Research_Summarizer_Agent",
    sub_agents=[researcher,summarizer],
    )
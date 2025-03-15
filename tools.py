from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime


def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"


save2txt = Tool(
    name="save_to_txt",
    func=save_to_txt,
    description="Saving research results to txt file"
)


search_engine = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search_engine.run,
    description="Search the web for info"
)

api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=400)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

tools_used = [save2txt, search_tool, wiki_tool]

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from schemas import ResearchResponse
from llm_setup import llm
from tools import tools_used


def main():
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}")
    ]).partial(format_instructions=parser.get_format_instructions())

    agent = create_tool_calling_agent(
        llm=llm,
        prompt=prompt,
        tools=tools_used
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools_used, verbose=True)
    query = input("What do you want to research today ? ->  ")
    raw_response = agent_executor.invoke({"query": query + "\n\nSave response to file"})

    structured_response = parser.parse(raw_response.get("output"))

    return structured_response


if __name__ == "__main__":
    load_dotenv()
    main()

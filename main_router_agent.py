
import os
from dotenv import load_dotenv
load_dotenv()
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain import hub

from backend.core import run_llm
from backend.tools.tools import cars_tool, general_tool, medical_tool
from langchain.agents import AgentExecutor, create_react_agent
from typing import Any, Dict, List

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")



def main_agent(query:str, chat_history: List[Dict[str, Any]] = []):
    # OpenAI's Chatmodel for main router agen
    llm = ChatOpenAI(verbose=True,temperature=0,openai_api_key=OPENAI_API_KEY)

    template = """
        You are a helpful assistant.
        Your name is bob..
        You should answer questions with correct answer.
        Do not guess or estimate  answers. you must rely only on the answer that you get from the tools.
        Do not answer to question with anything else but the tools provided to you.

        Begin!

        Question: {question}
        """
    tools=[
        cars_tool(), medical_tool(), general_tool()
    ]

    print(tools)

    react_prompt = hub.pull("hwchase17/react")

    prompt_template = PromptTemplate(
        template=template, input_variables=["question"]
    )
   
    agent=create_react_agent(
        llm,
        tools,
        prompt=react_prompt
    )

   
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    answer=agent_executor.invoke(
        input={
            "input":prompt_template.format_prompt(question=query),
            "chat_history":chat_history

        }
    )
    
    return answer

if __name__=='__main__':
    print(main_agent(query="Do you have bandages?"))
    

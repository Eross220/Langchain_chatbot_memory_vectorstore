
import os
from dotenv import load_dotenv
load_dotenv()
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain import hub
from backend.tools.tools import cars_tool, general_tool, medical_tool
from langchain.agents import AgentExecutor, create_react_agent
from typing import Any, Dict, List
from langchain.agents import create_openai_functions_agent
from backend.output_parsers import product_parser
from backend.chains.custom_chains import get_products_chain



OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")



def main_agent(query:str, chat_history: List[Dict[str, Any]] = []):
    # OpenAI's Chatmodel for main router agen
    llm = ChatOpenAI(verbose=True,temperature=0,openai_api_key=OPENAI_API_KEY)

    template = """
        You are a helpful assistant.
        Do not guess or estimate  answers. you must rely only on the answer that you get from the tools.
        Do not answer to question with anything else but the tools provided to you.

        Begin!

        Question: {question}
        """
    
    prompt_template= PromptTemplate(
        template=template,
        input_variables=["question"]
    )

    tools=[
       medical_tool(), cars_tool()
    ]

    prompt = hub.pull("hwchase17/openai-functions-agent")

    agent=create_openai_functions_agent(
        llm,
        tools,
        prompt
    )
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


    answer=agent_executor.invoke(
        input={
            "input":prompt_template.format_prompt(question=query),
            "chat_history":chat_history

        }
    )


    products_chains=get_products_chain()
    products= products_chains.run(answer=answer["output"])
   
    products= product_parser.parse(products)
    product_list= products.to_dict()

    print("products:", products)

    # print("question:\n", query)
    
    # print("answer: ",answer["output"])

    # print("product list:\n", product_list)
    # print("product:\n",product_list["products"][0].to_json())

    return (
        {
            "result":answer["output"],
            "product_list":products.to_dict()
        }
    ) 

    # return answer["output"]

if __name__=='__main__':
    
    # print(main_agent(query=" I have toothache. what should I do?"))
    # print(main_agent(query="Do you have bandages?"))

    

    print(main_agent(query="Truck"))

    print(main_agent(query="Petrol"))

    print(main_agent(query="I am looking for red color truck."))


    
   
    

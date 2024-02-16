
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


    print(tools)



    prompt = hub.pull("hwchase17/openai-functions-agent")

    agent=create_openai_functions_agent(
        llm,
        tools,
        prompt
    )
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # answer=agent_executor.invoke(
    #     {
    #         "input":query,
    #         "chat_history": chat_history  
    #     }
    # )

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

    # print("question:\n", query)
    
    # print("answer: ",answer["output"])

    # print("product list:\n", product_list)
    # print("product:\n",product_list["products"][0].to_json())

    return (
        {
            "result":answer["output"],
            "product_list":product_list
        }
    ) 

if __name__=='__main__':
    
    # main_agent(query=" I have toothache. what should I do?")
    answer = main_agent(query="Do you have bandages")

    print(answer)

    # product_list =answer["product_list"]["products"]

    
    # def create_product_string(porduct_list:List[any]):
    #     product_json_array= []

    #     product_list_str = " " 
    #     for index, product in enumerate(product_list):
    #         product_json_array.append(product.to_json())
    #         product_json= product.to_json()
    #         product_name= product_json['product_name']
    #         product_link= product_json['product_link']
    #         if len(product_json["product_link"])!=0:
    #              product_list_str += f"{index}. {product_name} \n {product_link}"
           

    #     return product_list_str


    # print("json array:",create_product_string(product_list))
    #answer= main_agent(query="what is your name?")
    # products_chains=get_products_chain()

    # products= products_chains.run(answer=answer["output"])
    # products= product_parser.parse(products)

    # print(products.to_dict())
    

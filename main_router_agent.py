
import os
from dotenv import load_dotenv
load_dotenv()
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain import hub
from backend.tools.tools import cars_tool, general_tool, medical_tool
from langchain.agents import AgentExecutor
from typing import Any, Dict, List
from langchain.agents import create_openai_functions_agent
from backend.output_parsers import product_parser
from backend.chains.custom_chains import get_products_chain
from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import json

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

car_template = """
        You are a  car related helpful assistant.
        Your purpose is to answer question about cars 
        Do not guess or estimate  answers. you must rely only on the answer that you get from the tools.
        Do not answer to question with anything else but the tools provided to you.
        Begin!

        Question: {question}
        """

medical_template = """
        You are a  medical helpful assistant.
        Your purpose is to answer question about Medical & Health. 
        Do not guess or estimate  answers. you must rely only on the answer that you get from the tools.
        If you don't find the tools, you must use to answer general_tool().
        Do not answer to question with anything else but the tools provided to you.
        Begin!

        Question: {question}
        """

general_template = """
        You are a helpful assistant.
        Your purpose is to answer about common question excluding car and medical.
        Do not guess or estimate  answers. you must rely only on the answer that you get from the tools.
        Do not answer to question with anything else but the tools provided to you.
        Begin!

        Question: {question}
        """

embeddings = OpenAIEmbeddings()
prompt_templates = [car_template, medical_template, general_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)

def prompt_router(input):
    query_embedding = embeddings.embed_query(input["query"])
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    print("Using Car" if most_similar == car_template else "Using Medical")
    return most_similar

def main_agent(query:str, chat_history: List[Dict[str, Any]] = []):
    # OpenAI's Chatmodel for main router agen
    llm = ChatOpenAI(verbose=True,temperature=0,openai_api_key=OPENAI_API_KEY)

    template = """
        You are a  Bob helpful assistant.
        Do not guess or estimate  answers. you must rely only on the answer that you get from the tools.
        Do not answer to question with anything else but the tools provided to you.
        Begin!

        Question: {question}
        """
    # template=prompt_router({
    #     "query":query
    # })
    prompt_template= PromptTemplate(
        template=template,
        input_variables=["question"]
    )



    #############certain tools for answering with certain domain##############

    tools=[
       medical_tool(), cars_tool(), general_tool()
    ]


    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent=create_openai_functions_agent(
        llm,
        tools,
        prompt
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, tags=[query] ,handle_parsing_errors=True)
    answer=agent_executor.invoke(
        input={
            "input":prompt_template.format_prompt(question=query),
            "chat_history":chat_history,
            "tags":query
        }
    )

    ###########################parsing answer as answer and product list for frontend###########

    products_chains=get_products_chain()
    products= products_chains.invoke({"answer":answer["output"]})
    products= products_chains.run(answer=answer["output"])

    #check validation for parsing
    def is_valid_json(products):
        try:
            json.loads(products)
            return True
        except ValueError:
            return False
    
    print("json validation:",is_valid_json(products))
    
    if(is_valid_json(products)):
        products= product_parser.parse(products)
        product_list= products.to_dict()
    else :
        product_list={ "products":[]}
    

    return (
        {
            "result":answer["output"],
            "product_list":product_list
        }
    ) 

if __name__=='__main__':
    
    # print(main_agent(query=" I have toothache. what should I do?"))
    # print(main_agent(query="Do you have bandages?"))

    

    
    print(main_agent(query="Hello"))

    print(main_agent(query="red"))

    # print(main_agent("what is petrol?"))

    # print(main_agent(query="Truck"))

    # print(main_agent(query="Petrol"))

    # print(main_agent(query="I am looking for red color truck."))


    
   
    

import os
from dotenv import load_dotenv
load_dotenv()
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from typing import Any, Dict, List
from langchain import hub
# import backend.tools.tools as car_agent_tool
#from backend.tools.tools import cars_parameter_confirm_tool
from langchain.agents import AgentExecutor, create_react_agent, create_openai_functions_agent
from backend.slots.slot_filling_conversation import SlotFilling
from backend.slots.slot_memory import SlotMemory
from backend.core import run_llm

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

llm_slot = ChatOpenAI(temperature=0.7)
memory = SlotMemory(llm=llm_slot)

slot_filling = SlotFilling(memory=memory, llm=llm_slot)
chain = slot_filling.create()


def car_answer_agent(query:str,  chat_history: List[Dict[str, Any]] = []):
    # LLM for car answer agent.. this process the questions 
    llm = ChatOpenAI(verbose=True,temperature=0,openai_api_key=OPENAI_API_KEY)

    print("current slot", slot_filling.memory.current_slots)
    if(slot_filling.memory.inform_check==False):
       answer=chain.predict(input=query)
    
    else : 
        answer=run_llm(query=query)
    # template = """
    #     You are a helpful assistant having a conversation with a human.
    #     Your purpose is to answer the question about cars.
    #     You should answer questions with correct answer.
    #     For example, to answer any question about cars, you should require the type of car: SUV, Truck, Sedan, Other.
    #     You should require the fuel type: Petrol, hybrid, electric.
    #     You should require the color: blue, green, grey, red, white, silver, black. If you don't get this information then prompt a follow up question to complete the context.
    #     Begin!

    #     Question: {question}
    #     """
    # tools=[
    #     car_agent_tool.cars_parameter_confirm_tool()
    # ]

    # prompt = hub.pull("hwchase17/openai-functions-agent")


    # prompt_template= PromptTemplate(
    #     template=template,
    #     input_variables=["question"]
    # )

    # agent= create_openai_functions_agent(
    #     llm,
    #     tools,
    #     prompt
    # )

    # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # answer=agent_executor.invoke(
    #     input={
    #         "input":prompt_template.format_prompt(question=query),
    #         "chat_history":chat_history

    #     }
    # )

    return answer







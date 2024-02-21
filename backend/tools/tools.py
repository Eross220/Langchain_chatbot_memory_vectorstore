import os
from langchain.tools import BaseTool, StructuredTool, tool
from dotenv import load_dotenv
load_dotenv()
from typing import Any, Dict, List
from backend.agents.car_answer_agent import car_answer_agent

from backend.tools.tools_function import run_llm_rag_with_media_link, run_llm, general_answer
from langchain.tools import  StructuredTool
from dotenv import load_dotenv
load_dotenv()

from backend.agents.car_answer_agent import car_answer_agent




def cars_tool():
    
    structured_tool = StructuredTool.from_function(
        name="car_answer",
        func=car_answer_agent,
        description="Search for information about Cars. For any questions related Cars, you must use this tool!",
        return_direct=True
    )

    return structured_tool

def medical_tool():
    
    structured_tool = StructuredTool.from_function(
        name="medical_answer",
        func=run_llm_rag_with_media_link,
        description="Search for information about Health & Medicine. For any questions about Health & Medicine, you must use this tool!",
        return_direct=True
    )

    return structured_tool

def general_tool():
    structured_tool = StructuredTool.from_function(
        name="general_answer",
        func=general_answer,
        description="Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects.If you can't find certain tools to answer question, you must use this tool. ",
        return_direct=True
    )

    return structured_tool

def cars_parameter_confirm_tool():
    
    structured_tool = StructuredTool.from_function(
        name="car_parameter_confirmed",
        func=run_llm,
        description="Use this tool when you confirmed all parameter of cars. If you didn't confirmed the all parameters about cars, you  don't must use this tool."
    )


    return structured_tool

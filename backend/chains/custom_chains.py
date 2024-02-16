from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from backend.output_parsers import product_parser

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
llm_creative = ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo")


def get_products_chain()-> LLMChain:
    product_template="""
       given the answer {answer} from I want you to extract:
       products list you mentioned..
       You must  only extract products which you explained in answer.
       \n
       {format_instructions}
    """
    product_prompt_template= PromptTemplate(
        input_variables=["answer"],
        template=product_template,
        partial_variables={
            "format_instructions": product_parser.get_format_instructions()
        },
    )

    return LLMChain(llm=llm, prompt=product_prompt_template)
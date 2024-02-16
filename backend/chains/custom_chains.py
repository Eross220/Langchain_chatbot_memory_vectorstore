from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from backend.output_parsers import product_parser

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
llm_creative = ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo")


def get_products_chain()-> LLMChain:
    product_template="""
       given the text ** {answer}  **, I want you to create:\n
       products list you mentioned.\n
       Please follow the following rules:\n
       1.You must  only extract products which is included in context.
       2.If you don't have links of products or product name, you shoud give empty value. Don't try to make.
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
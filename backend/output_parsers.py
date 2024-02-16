from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


class Products(BaseModel):
    products_list: List[str] = Field(description="product name list which you explained")
    

    def to_dict(self):
        return {"products": self.products_list}


class IceBreaker(BaseModel):
    ice_breakers: List[str] = Field(description="ice breaker list")

    def to_dict(self):
        return {"ice_breakers": self.ice_breakers}


class TopicOfInterest(BaseModel):
    topics_of_interest: List[str] = Field(
        description="topic that might interest the person"
    )

    def to_dict(self):
        return {"topics_of_interest": self.topics_of_interest}


product_parser = PydanticOutputParser(pydantic_object=Products)
ice_breaker_parser = PydanticOutputParser(pydantic_object=IceBreaker)
topics_of_interest_parser = PydanticOutputParser(pydantic_object=TopicOfInterest)
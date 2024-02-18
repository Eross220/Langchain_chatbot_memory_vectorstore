from langchain.chains import ConversationChain
from langchain.chat_models.base import BaseChatModel
from pydantic import BaseModel

from backend.slots.prompt import CHAT_PROMPT
from backend.slots.slot_memory import SlotMemory


class SlotFilling():
    memory: SlotMemory
    llm: BaseChatModel

    def __init__(self, memory: SlotMemory, llm: BaseChatModel):
        self.memory = memory
        self.llm = llm

    class Config:
        arbitrary_types_allowed = True

    def create(self) -> ConversationChain:
        return ConversationChain(llm=self.llm, memory=self.memory, prompt=CHAT_PROMPT)

    def log(self):
        print(f"【Slot】: {self.memory.current_slots}")
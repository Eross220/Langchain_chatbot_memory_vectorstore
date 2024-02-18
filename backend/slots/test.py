from langchain_openai import ChatOpenAI
from slot_filling_conversation import SlotFilling
from slot_memory import SlotMemory


llm = ChatOpenAI(temperature=0.7)
memory = SlotMemory(llm=llm)



slot_filling = SlotFilling(memory=memory, llm=llm)
chain = slot_filling.create()



print(chain.predict(input="Hi"))

print(chain.predict(input="I am looking for truck of red color"))

print(chain.predict(input="Petrol"))

print(slot_filling.memory.inform_check)

print(slot_filling.memory.current_slots)
slot_filling.log()
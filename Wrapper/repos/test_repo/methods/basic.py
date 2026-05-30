"""Class method that directly contains a .invoke call."""
from langchain.dummy import Anything


class AgentClient:
    def chat(self, message):
        obj = self.get_model()
        return obj.invoke({"input": message})

    def get_model(self):
        return None

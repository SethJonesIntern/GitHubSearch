"""LLM wrapper: the actual model call lives here. The prompt it sends is a
parameter, so a backward slice of the .invoke argument should leave the
function via that parameter and continue into the callers."""
from langchain.chat_models import ChatOpenAI


def call_model(prompt, temperature=0.0):
    model = ChatOpenAI(temperature=temperature)
    return model.invoke(prompt)

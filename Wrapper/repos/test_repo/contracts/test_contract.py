"""A pytest test with contract-like structure around an LLM call:

  * preconditions on the prompt that gets sent (the asserts/guards reachable
    by a backward slice of the .invoke argument), and
  * postconditions on the model's reply (asserts on the returned value).

build_prompt holds a precondition of its own, so the interprocedural slice has
to step into it to find the full set of constraints on the prompt.
"""
from test_repo.contracts.wrapper import call_model

MAX_LEN = 200


def build_prompt(topic):
    prompt = f"Write a haiku about {topic}"
    assert len(prompt) < MAX_LEN          # precondition: prompt length bounded
    return prompt


def test_haiku():
    topic = "spring"
    assert topic                          # precondition: topic is non-empty
    prompt = build_prompt(topic)
    reply = call_model(prompt, temperature=0.0)
    assert "haiku" not in reply           # postcondition on the reply
    assert len(reply) > 0                 # postcondition on the reply

from promptflow import tool
from typing import List


@tool
def format_pf_inputs_as_chat_completion(chat_history: List, chat_input: str) -> List:
    """Formats the chat history and the current chat input as an OpenAI style chat conversation."""
    # refactor the whole chat_history thing
    conversation = [
        {
            "role": "user" if "inputs" in message else "assistant",
            "content": (
                message["inputs"]["chat_input"]
                if "inputs" in message
                else message["outputs"]["chat_output"]
            ),
        }
        for message in chat_history
    ]

    # add the user input as last message in the conversation
    conversation.append({"role": "user", "content": chat_input})

    return conversation

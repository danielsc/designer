import json
import ast
import os
import inspect
import requests
from openai import AsyncAzureOpenAI
from urllib.parse import quote
from chainlit.playground.providers.openai import stringify_function_call
import chainlit as cl
import base64
from promptflow.client import PFClient
from promptflow.tracing import start_trace
from dotenv import load_dotenv
load_dotenv()

start_trace()


@cl.on_chat_start
def start_chat():
    print("starting chat")

    cl.user_session.set(
        "chat_history",
        [],
    )

    promptflows = [
        os.path.join(os.path.dirname(__file__), 'functions_flow'),
        os.path.join(os.path.dirname(__file__), 'autogen_flow')
    ]

    config = dict(
        active_promptflow = promptflows[0],
        promptflows = promptflows,
    )
    cl.user_session.set("config", config)
    

def show_images(image):
    elements = [
        cl.Image(
            content=image,
            name="generated image",
            display="inline",
        )
    ]
    return elements

@cl.step(type="llm")
async def call_promptflow(chat_history, message):
    prompt_flow = cl.user_session.get("config")["active_promptflow"]
    client = PFClient()
 
    response = await cl.make_async(client.test)(prompt_flow, 
                                                  inputs={"chat_history": chat_history,
                                                          "question": message.content})
    return response

async def activate_promptflow(command: str, command_id: str):
    config = cl.user_session.get("config")
    if len(command.split(" ")) < 2:
        await cl.Message(content=f"#### Promptflow is currently set to `{config['active_promptflow']}`").send()
        return
    
    promptflow_number = int(command.split(" ")[1])
    if promptflow_number < 0 or promptflow_number >= len(config["promptflows"]):
        await cl.Message(content=f"#### Invalid promptflow number `{promptflow_number}` -- needs to be between 0 and {len(config['promptflows'])}").send()
        return

    config["active_promptflow"] = config["promptflows"][promptflow_number]
    await cl.Message(content=f"#### Set promptflow to `{config['active_promptflow']}`").send()

    cl.user_session.set(
        "chat_history",
        [],
    )

@cl.on_message
async def run_conversation(message: cl.Message):
    question = message.content 
    chat_history = cl.user_session.get("chat_history")
 
    if question.startswith("/activate"):
        await activate_promptflow(message.content, message.id)
    else:
        response = await call_promptflow(chat_history, message)

        if "messages" in response:
            for thing in response["messages"]:
                async with cl.Step(name=thing["role"]) as child_step:
                    child_step.output = thing["content"]

        if response["image"]:
            elements = show_images(response["image"])
        else:
            elements = []  

        await cl.Message(content=response["answer"], 
                            author="Answer",
                            elements=elements).send()
        
        chat_history.append({"inputs": {"question": message.content}, 
                            "outputs": {"answer": response["answer"]}})

if __name__ == "__main__":

    print("using the follwoing chat_model", os.getenv("OPENAI_CHAT_MODEL"))

    from chainlit.cli import run_chainlit
    run_chainlit(__file__)

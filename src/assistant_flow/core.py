# enable type annotation syntax on Python versions earlier than 3.9
from __future__ import annotations

import time
import os
import logging
import json
import inspect

from openai import AzureOpenAI
import base64

from promptflow.tracing import trace
from opentelemetry import trace as otel_trace
from opentelemetry import context as otel_context
from promptflow.contracts.multimedia import Image
from threading import Thread

tracer = otel_trace.get_tracer(__name__)

class AssistantsAPIGlue:
    def __init__(
        self,
        client: AzureOpenAI,
        question: str,
        session_state: dict[str, any] = {},
        context: dict[str, any] = {},
        tools: dict[str, callable] = {},
    ):
        # Provision an AzureOpenAI client for the assistants
        logging.info("Creating AzureOpenaI client")
        self.client = client
        self.tools = tools

        if "max_waiting_time" in context:
            logging.info(
                f"Using max_waiting_time from context: {context['max_waiting_time']}"
            )
            self.max_waiting_time = context["max_waiting_time"]
        else:
            self.max_waiting_time = 120

        if "thread_id" in session_state:
            logging.info(f"Using thread_id from session_stat: {session_state['thread_id']}")
            otel_trace.get_current_span().set_attribute("AssistantsAPIGlue_thread_id",  session_state['thread_id'])
            self.thread_id = self.client.beta.threads.retrieve(session_state['thread_id']).id
        else:
            logging.info(f"Creating a new thread")
            self.thread_id = self.client.beta.threads.create().id
            otel_trace.get_current_span().set_attribute("AssistantsAPIGlue_thread_id", self.thread_id)

        # Add last message in the thread
        logging.info("Adding message in the thread")
        self.add_message(dict(role="user", content=question))

        if "assistant_id" in context:
            logging.info(f"Using assistant_id from context: {context['assistant_id']}")
            self.assistant_id = context["assistant_id"]
        elif "OPENAI_ASSISTANT_ID" in os.environ:
            logging.info(
                f"Using assistant_id from environment variables: {os.getenv('OPENAI_ASSISTANT_ID')}"
            )
            self.assistant_id = os.getenv("OPENAI_ASSISTANT_ID")
        else:
            raise Exception(
                "You need to provide OPENAI_ASSISTANT_ID in the environment variables (or pass assistant_id in the context)"
            )
        # get current span
        otel_trace.get_current_span().set_attribute("AssistantsAPIGlue_assistant_id", self.assistant_id)

    def add_message(self, message):
        _ = self.client.beta.threads.messages.create(
            thread_id=self.thread_id,
            role=message["role"],
            content=message["content"],
        )

    @trace
    def run(self, messages=None):
        # run handler in a separate thread
        # Capture the current context
        current_context = otel_context.get_current()
        self.queue = QueuedIteratorStream()

        def run_with_context():
            # Reactivate the captured context in the new thread
            token = otel_context.attach(current_context)
            try:
                self.run_inner(messages)
            finally:
                otel_context.detach(token)

        # run handler in a separate thread with context
        thread = Thread(target=run_with_context)
        thread.start()
        
        return dict(
            chat_output=self.queue.iter(),
            session_state={ "thread_id": self.thread_id },
            planner_raw_output=None
        )
        


    def run_inner(self, messages=None):
        if messages:
            logging.info("Adding last message in the thread")
            _ = self.client.beta.threads.messages.create(
                thread_id=self.thread_id,
                role=messages[-1]["role"],
                content=messages[-1]["content"],
            )

        # Run the thread
        logging.info("Running the thread")
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=self.assistant_id,
        )
        logging.info(f"Run status: {run.status}")
        self.queue.send(f"Running message on Thread: {self.thread_id}\n")

        start_time = time.time()
        step_logging_cursor = None

        # keep track of messages happening during the loop
        internal_memory = []

        # loop until max_waiting_time is reached
        while (time.time() - start_time) < self.max_waiting_time:
            # checks the run regularly
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread_id, run_id=run.id
            )
            logging.info(
                f"Run status: {run.status} (time={int(time.time() - start_time)}s, max_waiting_time={self.max_waiting_time})"
            )

            # # check run steps
            # run_steps = self.client.beta.threads.runs.steps.list(
            #     thread_id=self.thread_id, run_id=run.id, after=step_logging_cursor
            # )

            # for step in run_steps:
            #     trace_step(step.model_dump())
            #     internal_memory.append(step.step_details.model_dump())
            #     step_logging_cursor = step.id

            if run.status == "completed":
                # check run steps
                run_steps = self.client.beta.threads.runs.steps.list(
                    thread_id=self.thread_id, run_id=run.id #, after=step_logging_cursor
                )

                for step in reversed(list(run_steps)):
                    trace_step(step.model_dump())
                    internal_memory.append(step.step_details.model_dump())

                messages = []
                for message in self.client.beta.threads.messages.list(
                    thread_id=self.thread_id
                ):
                    message = self.client.beta.threads.messages.retrieve(
                        thread_id=self.thread_id, message_id=message.id
                    )
                    messages.append(message)
                logging.info(f"Run completed with {len(messages)} messages.")
                self.queue.send(f"Run completed with {len(messages)} messages.\n")

                final_message = messages[0]

                mixed_response = []

                for message in final_message.content:
                    if message.type == "text":
                        mixed_response.append(message.text.value)
                    elif message.type == "image_file":
                        file_id = message.image_file.file_id
                        mixed_response.append(
                            Image(self.client.files.content(file_id).read())
                        )
                    else:
                        logging.critical("Unknown content type: {}".format(message.type))

                for response in mixed_response:
                    self.queue.send(response)
                
                self.queue.end()

                return
            
            elif run.status == "requires_action":
                # if the run requires us to run a tool
                tool_call_outputs = []

                for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                    trace_tool(tool_call.model_dump())
                    internal_memory.append(tool_call.model_dump())
                    self.queue.send(f"Tool call: {tool_call.function.name} with arguments: {tool_call.function.arguments}\n")

                    if tool_call.type == "function":
                        tool_func = self.tools[tool_call.function.name]
                        tool_call_output = tool_func(
                            **json.loads(tool_call.function.arguments)
                        )

                        # tool_call_output = call_tool(
                        #     tools_map[tool_call.function.name], **json.loads(tool_call.function.arguments)
                        # )

                        tool_call_outputs.append(
                            {
                                "tool_call_id": tool_call.id,
                                "output": json.dumps(tool_call_output),
                            }
                        )
                        internal_memory.append(tool_call_outputs[-1])
                    else:
                        raise ValueError(f"Unsupported tool call type: {tool_call.type}")

                if tool_call_outputs:
                    _ = self.client.beta.threads.runs.submit_tool_outputs(
                        thread_id=self.thread_id,
                        run_id=run.id,
                        tool_outputs=tool_call_outputs,
                    )
            elif run.status in ["cancelled", "expired", "failed"]:
                raise ValueError(f"Run failed with status: {run.status}")

            elif run.status in ["in_progress", "queued"]:
                time.sleep(1)

            else:
                raise ValueError(f"Unknown run status: {run.status}")

@trace
def trace_step(step):
    logging.info(
            "The assistant has moved forward to step {}".format(step["id"])
    )
@trace
def trace_tool(tool_call):
    logging.info(
            "The assistant has asks for tool execution of {}".format(tool_call["function"]["name"])
    )



import queue
import json
from typing import Any, Dict, List
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry import trace

class QueuedIteratorStream:
    terminate: str = "<--terminate-->"
    queue: queue.Queue[str]
    output: List[str]
    context_carrier: Dict[str, Any]

    def __init__(self) -> None:
        self.queue = queue.Queue()
        self.output = []
        self.context_carrier = {}
        # Write the current context into the carrier.
        TraceContextTextMapPropagator().inject(self.context_carrier)

    def send(self, event: str) -> None:
        if event is not None and event != "":
            if isinstance(event, Image):
                self.output.append(event.to_base64(with_type=True))
                self.queue.put_nowait(f"![]({event.to_base64(with_type=True)})")
            else:
                self.output.append(event)
                self.queue.put_nowait(event)

    def end(self) -> None:
        tracer = trace.get_tracer(__name__)
        ctx = TraceContextTextMapPropagator().extract(carrier=self.context_carrier)

        with tracer.start_as_current_span("stream", context=ctx) as span:
            span.set_attribute("framework", "promptflow")
            span.set_attribute("span_type", "Function")
            span.set_attribute("function", "stream")
            span.set_attribute("output", json.dumps(self.output))

        self.queue.put_nowait(self.terminate)

    def iter(self) -> Any:
        while True:
            token = self.queue.get()

            if token == self.terminate:
                break

            yield token


from openai import AssistantEventHandler
from typing_extensions import override
from openai.types.beta import AssistantStreamEvent
from openai.types.beta.threads import (
    Run,
    Text,
    Message,
    ImageFile,
    TextDelta,
    MessageDelta,
    MessageContent,
    MessageContentDelta,
)
from openai.types.beta.threads.runs import RunStep, ToolCall, RunStepDelta, ToolCallDelta


class EventHandler(AssistantEventHandler):
    def __init__(self, thread_id, assistant_id):
        super().__init__()
        self.output = None
        self.tool_id = None
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.run_id = None
        self.run_step = None
        self.function_name = ""
        self.arguments = ""
      
    @override
    def on_text_created(self, text) -> None:
        print(f"\nassistant on_text_created > ", end="", flush=True)

    @override
    def on_text_delta(self, delta, snapshot):
        # print(f"\nassistant on_text_delta > {delta.value}", end="", flush=True)
        print(f"{delta.value}")

    @override
    def on_end(self, ):
        print(f"\n end assistant > ",self.current_run_step_snapshot, end="", flush=True)

    @override
    def on_exception(self, exception: Exception) -> None:
        """Fired whenever an exception happens during streaming"""
        print(f"\nassistant > {exception}\n", end="", flush=True)

    @override
    def on_message_created(self, message: Message) -> None:
        print(f"\nassistant on_message_created > {message}\n", end="", flush=True)

    @override
    def on_message_done(self, message: Message) -> None:
        print(f"\nassistant on_message_done > {message}\n", end="", flush=True)

    @override
    def on_message_delta(self, delta: MessageDelta, snapshot: Message) -> None:
        # print(f"\nassistant on_message_delta > {delta}\n", end="", flush=True)
        pass

    @override
    def on_tool_call_created(self, tool_call: ToolCall):
        print(f"\nassistant on_tool_call_created > {tool_call}")
        
    @override
    def on_tool_call_done(self, tool_call: ToolCall) -> None:       
        print(f"\nassistant on_tool_call_done > {tool_call}")
        
        
    @override
    def on_run_step_created(self, run_step: RunStep) -> None:
        # 2       
        print(f"on_run_step_created")
        self.run_id = run_step.run_id
        self.run_step = run_step
        print("The type of run_step run step is ", type(run_step), flush=True)
        print(f"\n run step created assistant > {run_step}\n", flush=True)

    @override
    def on_run_step_done(self, run_step: RunStep) -> None:
        print(f"\n run step done assistant > {run_step}\n", flush=True)

    @override
    def on_tool_call_delta(self, delta, snapshot): 
        if delta.type == 'function':
            # the arguments stream thorugh here and then you get the requires action event
            print(delta.function.arguments, end="", flush=True)
            self.arguments += delta.function.arguments
        elif delta.type == 'code_interpreter':
            print(f"on_tool_call_delta > code_interpreter")
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print(f"\n\noutput >", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)
        else:
            print("ELSE")
            print(delta, end="", flush=True)

    @override
    def on_event(self, event: AssistantStreamEvent) -> None:
        # print("In on_event of event is ", event.event, flush=True)

        if event.event == "thread.run.requires_action":
            print("\nthread.run.requires_action > submit tool call")
            print(f"ARGS: {self.arguments}")

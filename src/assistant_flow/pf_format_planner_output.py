from promptflow import tool
from typing import Any


@tool
def format_planner_output(planner_output: Any) -> dict:
    """This extracts the user-friendly message from the orchestrator output."""
    assert planner_output is not None, "Planner returned None"

    return dict(
        chat_output=planner_output["choices"][0]["message"]["content"],
        session_state=planner_output["choices"][0]["session_state"]
      )

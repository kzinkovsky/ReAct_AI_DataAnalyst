from model import RetrieveTextInput, SummarizeTextInput, CountStructuredInput, FunctionType, FunctionInput
from config import client, tools
from tools import retrieve_text, summarize_text, count_structured
import json

# The function to route inputs to an appropriate tool
def execute_function(function_call: FunctionType) -> dict:

    if isinstance(function_call, RetrieveTextInput):
        return retrieve_text(function_call)
    elif isinstance(function_call, SummarizeTextInput):
        return summarize_text(function_call)
    elif isinstance(function_call, CountStructuredInput):
        return count_structured(function_call)

# The ReAct Agent function
def process_user_query_react(messages: list) -> tuple[str, list]:
    """ReAct agent to process user queries interactively."""

    for step in range(7):  # think-observe-react step limit
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0
        )

        assistant_message = response.choices[0].message
        
        # Processing a tool call
        if assistant_message.tool_calls:
            
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [tc.model_dump() for tc in assistant_message.tool_calls]
            })
            for function in assistant_message.tool_calls:
                function_call = function.function
                function_name = function_call.name
                arguments = json.loads(function_call.arguments)
                arguments["function_type"] = function_name

                try:
                    function_input = FunctionInput(function_call=arguments)
                    result = execute_function(function_input.function_call)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": function.id,
                        "content": json.dumps(result)
                    })
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": function.id,
                        "content": error_msg
                    })

        else:
            # Final response without function call
            messages.append({
                "role": "assistant",
                "content": assistant_message.content
            })
            return assistant_message.content, messages

    return "Too many steps. Reached iteration limit.", messages
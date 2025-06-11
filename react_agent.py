from config import client, model_name, tools
import json
from model import (
                   DatasetOverview, 
                   SelectSemanticIntent, 
                   SelectSemanticCategory, 
                   CountIntent, CountCategory, 
                   SumValues, 
                   MultiplicationFloat, 
                   DivisionFloat, 
                   ShowExamples, 
                   SummarizeText, 
                   Finish,
                   FunctionType, 
                   FunctionInput 
                   )
from tools import (
                   get_dataset_overview,
                   select_semantic_intent,
                   select_semantic_category,
                   count_intent,
                   count_category,
                   sum_values,
                   multiplication_float,
                   division_float,
                   show_examples,
                   summarize_text,
                   finish
                   )

# The function to route inputs to an appropriate tool

def execute_function(function_call: FunctionType) -> dict:

    if isinstance(function_call, DatasetOverview):
        return get_dataset_overview(function_call)
    elif isinstance(function_call, SelectSemanticIntent):
        return select_semantic_intent(function_call)
    elif isinstance(function_call, SelectSemanticCategory):
        return select_semantic_category(function_call)
    elif isinstance(function_call, CountIntent):
        return count_intent(function_call)
    elif isinstance(function_call, CountCategory):
        return count_category(function_call)
    elif isinstance(function_call, SumValues):
        return sum_values(function_call)
    elif isinstance(function_call, MultiplicationFloat):
        return multiplication_float(function_call)
    elif isinstance(function_call, DivisionFloat):
        return division_float(function_call)
    elif isinstance(function_call, ShowExamples):
        return show_examples(function_call)
    elif isinstance(function_call, SummarizeText):
        return summarize_text(function_call)
    elif isinstance(function_call, Finish):
        return finish(function_call)

# The ReAct Agent function
def process_user_query_react(messages: list) -> tuple[str, list]:
    """ReAct agent to process user queries interactively."""

    for step in range(20):  # think-observe-react step limit
        response = client.chat.completions.create(
            model=model_name,
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
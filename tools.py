from config import df_csdataset, client, model_name
import pandas as pd
from model import (
                   IntentClass,
                   CategoryClass,
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

# Tools implementations

def get_dataset_overview(_: DatasetOverview) -> dict:
    """Get dataset overview."""
    return {
        "dataset_name": "Bitext Customer Support Dataset",
        "description": "Customer Service Tagged Training Dataset for LLM-based Virtual Assistants.",
        "working fields": {
            "instruction": "User query from the Customer Service domain.",
            "intent": "The classified intent of the query (e.g., place_order, cancel_order).",
            "category": "High-level semantic category for intent (e.g., ORDER, DELIVERY).",
            "responce": "Example expected assistant response."
        },
        "number of rows in the dataset": len(df_csdataset),
        "number of unique intents": len(df_csdataset["intent"].unique()),
        "number of unique categories": len(df_csdataset["category"].unique())
    }


def select_semantic_intent(input_data: SelectSemanticIntent) -> dict:
    """Uses LLM to select the most appropriate intent based on a text description."""
    global df_csdataset

    query = input_data.query
    possible_intents = [intent.value for intent in IntentClass]

    system_prompt = (
        "You classify user queries into one of the following intents:\n\n"
        + "\n".join(f"- {intent}" for intent in possible_intents) +
        "\n\nChoose one most suitable intent based on the user's request. "
        "Answer only with the meaning of the intent, without explanation."
    )

    user_query = f"User query: {query}"

    messages = [
        {"role": "system",
         "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    
    response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0
        )
    
    selected_intent = response.choices[0].message.content.strip()

    if selected_intent not in possible_intents:
        return {
            "query": query,
            "error": "LLM was unable to select a valid intent",
            "llm_response": selected_intent
        }

    return {
        "query": query,
        "selected_intent": selected_intent
    }

def select_semantic_category(input_data: SelectSemanticCategory) -> dict:
    """Uses LLM to select the most appropriate category based on a text description."""
    global df_csdataset
    query = input_data.query
    possible_category = [category.value for category in CategoryClass]

    system_prompt = (
        "You classify user queries into one of the following category:\n\n"
        + "\n".join(f"- {category}" for category in possible_category) +
        "\n\nChoose one most suitable category based on the user's request. "
        "Answer only with the meaning of the category, without explanation."
    )

    user_query = f"User query: {query}"

    messages = [
        {"role": "system",
         "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    
    response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0
        )
    
    selected_category = response.choices[0].message.content.strip()

    if selected_category not in possible_category:
        return {
            "query": query,
            "error": "LLM was unable to select a valid category",
            "llm_response": selected_category
        }

    return {
        "query": query,
        "selected_intent": selected_category
    }


def count_intent(input_data: CountIntent) -> dict:
    """Count of rows for particular intent."""
    global df_csdataset
    intent_value = input_data.intent_class.value

    filtered_df = df_csdataset[df_csdataset["intent"] == intent_value]

    return {"selected intent": intent_value, "number of rows": len(filtered_df)}


def count_category(input_data: CountCategory) -> dict:
    """Count of rows for particular category."""
    global df_csdataset
    category_value = input_data.category_class.value

    filtered_df = df_csdataset[df_csdataset["category"] == category_value]

    return {"selected intent": category_value, "number of rows": len(filtered_df)}


def sum_values(input_data: SumValues) -> dict:
    """Sum of numbers."""
    
    return {"sum": sum(input_data.values)}


def multiplication_float(input_data: MultiplicationFloat) -> float:
    """Multiplication of two numbers."""

    return input_data.a * input_data.b

def division_float(input_data: DivisionFloat) -> float:
    """Division of two numbers."""

    return input_data.numerator / input_data.denominator


def show_examples(input_data: ShowExamples) -> dict:
    """Show n rows of dataset with or without conditions on selected Inent and Order."""
    global df_csdataset
    n = input_data.number_rows
    target_field = input_data.target_field.value if input_data.target_field else None
    condition_field = input_data.condition_field.value if input_data.condition_field else None
    condition_field_value = input_data.condition_field_value.value if input_data.condition_field_value else None
    
    if isinstance(input_data.condition_field_value, IntentClass):
        df_source = df_csdataset[df_csdataset["intent"] == condition_field_value]
        condition_status = f"{condition_field} = {condition_field_value}"

    elif isinstance(input_data.condition_field_value, CategoryClass):
        df_source = df_csdataset[df_csdataset["category"] == condition_field_value]
        condition_status = f"{condition_field} = {condition_field_value}"

    else:
        df_source = df_csdataset
        condition_status = "no condition (full dataset)"

    if target_field:
        df_sample = df_source[[target_field]].sample(n=min(n, len(df_source)), random_state=42)
    else:
        df_sample = df_source.sample(n=min(n, len(df_source)), random_state=42)

    return {
        "by condition": condition_status,
        "number of examples shown": len(df_sample),
        "examples": df_sample.to_dict(orient="records")
    }


def summarize_text(input_data: SummarizeText) -> dict:
    """Summarize text from unstructured field (instruction or response) using AI summarization."""
    global df_csdataset
    text_field = input_data.text_field.value if input_data.text_field else "instruction"
    number_rows = min(input_data.number_rows, 100)
    condition_field = input_data.condition_field.value if input_data.condition_field else None
    condition_field_value = input_data.condition_field_value.value if input_data.condition_field_value else None 

    if condition_field == "intent" and not isinstance(input_data.condition_field_value, IntentClass):
        return {"error": f"Expected IntentClass for field 'intent', got {type(input_data.condition_field_value).__name__}"}
    if condition_field == "category" and not isinstance(input_data.condition_field_value, CategoryClass):
        return {"error": f"Expected CategoryClass for field 'category', got {type(input_data.condition_field_value).__name__}"}

    if condition_field and condition_field_value:
        df_filtered = df_csdataset[df_csdataset[condition_field] == condition_field_value]
        condition_status = f"{condition_field} = {condition_field_value}"
    else:
        return {"error": "User query should be more specified in terms of Category or Intent"}

    available_rows = len(df_filtered)
    if available_rows == 0:
        return {"error": f"No rows match condition: {condition_status}"}
    
    nmin = min(number_rows, available_rows)

    df_sample = df_filtered.sample(n=nmin, random_state=42)
    text = "\n".join(f"- {line}" for line in df_sample[text_field])

    prompt = f"""Here is a list of messages:\n{text}\n\nProvide a brief summary of the main themes they raise:"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    return {
        "summarized_field": text_field,
        "conditions": condition_status,
        "number of samples": nmin,
        "summary": response.choices[0].message.content.strip()
    }


def finish(input_data: Finish) -> str:
    """Finish: answer ready."""
    return "Answer ready."

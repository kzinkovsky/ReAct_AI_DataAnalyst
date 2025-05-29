from config import df_csdataset, client
from model import RetrieveTextInput, SummarizeTextInput, CountStructuredInput, FunctionType, FunctionInput
import pandas as pd

# Tools implementations

def retrieve_text(input_data: RetrieveTextInput) -> list:
    """Retrieve text from dataframe."""
    field = input_data.target_field.value
    condition_field = input_data.condition_field.value if input_data.condition_field else None
    condition_class = input_data.condition_class.value if input_data.condition_class else None
    number_rows = input_data.number_rows
    
    if field and condition_field and condition_class:
        #filtered_df = df_csdataset[df_csdataset[condition_field] == condition_class][field].head(number_rows)
        filtered_df = df_csdataset[df_csdataset[condition_field] == condition_class][field].sample(n=number_rows)

    elif field:
        #filtered_df = df_csdataset[field].head(number_rows)
        filtered_df = df_csdataset[field].sample(n=number_rows)

    else:
        #filtered_df = df_csdataset.head(number_rows)
        filtered_df = df_csdataset.sample(n=number_rows)

    return filtered_df.tolist()


def summarize_text(input_data: SummarizeTextInput) -> str:
    """Summarize a list of texts using AI summarization."""

    text = "\n".join(f"- {line}" for line in input_data.text_list)

    prompt = f"""Here is a list of messages:\n{text}\n\nProvide a brief summary of the main themes they raise:"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    return response.choices[0].message.content.strip()


def count_structured(input_data: CountStructuredInput) -> dict:
    """Count statistics for a structured column."""

    target_field = input_data.target_field.value
    count_operation = input_data.count_operation.value
    target_class = input_data.target_class.value if input_data.target_class else None

    if count_operation == 'list_unique':
        result = df_csdataset[target_field].unique().tolist()

    elif count_operation == 'unique_count':
        result = df_csdataset[target_field].nunique()

    elif count_operation == 'distribution':
        value_counts = df_csdataset[target_field].value_counts()
        result = value_counts.to_dict()

    elif count_operation == 'class_count':
        if target_class is None:
            raise ValueError("target_class must be provided for class_count operation.")
        count = df_csdataset[df_csdataset[target_field] == target_class].shape[0]
        result = {target_class: count}

    elif count_operation == 'most_common':
        value_counts = df_csdataset[target_field].value_counts()

        if not value_counts.empty:
           max_count = value_counts.max()
           most_common_values = value_counts[value_counts == max_count].to_dict()
           result = [{"value": k, "count": v} for k, v in most_common_values.items()]
        else:
           result = []

    elif count_operation == 'least_frequent':
        value_counts = df_csdataset[target_field].value_counts()
        
        if not value_counts.empty:
           min_count = value_counts.min()
           least_common_values = value_counts[value_counts == min_count].to_dict()
           result = [{"value": k, "count": v} for k, v in least_common_values.items()]
        else:
           result = []   

    else:
        raise ValueError(f"Unsupported count_operation: {count_operation}")

    return {
        "operation": count_operation,
        "target_field": target_field,
        "result": result
    }
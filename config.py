import pandas as pd
import streamlit as st
from openai import OpenAI
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

# Initialize the variable with the data set as a data frame
@st.cache_data
def load_csv_dataset():
    url = "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    df = pd.read_csv(url)
    return df

# Load dataset once and reuse
df_csdataset = load_csv_dataset()

# Initilize OpenAI object
api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# Define name of LLM model
model_name = "gpt-4o-mini"

# Initialize the variable with tools schema
tools = [
     {
        "type": "function",
        "function": {
            "name": "get_dataset_overview",
            "description": "Get name of dataset, total number of rows and column names",
            "parameters": DatasetOverview.model_json_schema()
        }
    },
     {
        "type": "function",
        "function": {
            "name": "select_semantic_intent",
            "description": "Uses LLM to select the most appropriate intent based on a text description",
            "parameters": SelectSemanticIntent.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "select_semantic_category",
            "description": "Uses LLM to select the most appropriate category based on a text description",
            "parameters": SelectSemanticCategory.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_intent",
            "description": "Count how many records match the selected Intent class",
            "parameters": CountIntent.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_category",
            "description": "Count how many records match the selected Category class",
            "parameters": CountCategory.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sum_values",
            "description": "Calculate the sum of a list of numeric values",
            "parameters": SumValues.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multiplication_float",
            "description": "Calculate the multiplication of two floats",
            "parameters": MultiplicationFloat.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "division_float",
            "description": "Calculate the division of two floats",
            "parameters": DivisionFloat.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "show_examples",
            "description": "Display N random examples from the dataset with optionally filtered field",
            "parameters": ShowExamples.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_text",
            "description": "Generate a concise summary based on N randomly selected messages from a specified field",
            "parameters": SummarizeText.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Indicate that the task is complete and no further steps are needed",
            "parameters": Finish.model_json_schema()
        }
    }
]


# Initilize the global variable with system prompt
system_prompt = """
You are a helpful assistant that answers questions based on an internal dataset.

Use the following reasoning process:
- Thought: Think about what needs to be done.
- Action: Call a tool if necessary.
- Observation: See what the tool returned.
- Repeat Thought → Action → Observation if needed.
- Final Answer: Only after enough observations, answer the user.

Important rules:
- You have access to a fixed set of tools. Always select the most relevant one based on the user query.
- If a tool returns no relevant data or cannot answer the query, try a different tool—but do not repeat calls with the same parameters.
– If the query is unrelated to the dataset or no relevant information exists in the data, say so clearly and politely.
– Do not attempt to answer based on general knowledge or assumptions.
"""
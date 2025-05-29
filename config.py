import pandas as pd
import streamlit as st
from openai import OpenAI
from model import RetrieveTextInput, SummarizeTextInput, CountStructuredInput

# Initialize the global variable with the data set as a data frame
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

# Initialize the global variable with tools schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_text",
            "description": "Retrieve text from dataframe",
            "parameters": RetrieveTextInput.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_text",
            "description": "Summarize a list of texts using AI summarization",
            "parameters": SummarizeTextInput.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_structured",
            "description": "Count statistics for a structured field",
            "parameters": CountStructuredInput.model_json_schema()
        }
    }
]

# Initilize the global variable with system prompt
system_prompt = """
You are a helpful assistant that answers questions strictly based on an internal customer support dataset.

Use the following process:
- Thought: Think about what needs to be done.
- Action: Call a tool if necessary.
- Observation: See what the tool returned.
- Repeat Thought → Action → Observation if needed.
- Final Answer: Only after enough observations, answer the user.

If no relevant data exists or the query is out-of-scope, politely inform the user.

If summarizing, always retrieve multiple entries before summarizing.
"""

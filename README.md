# AI Data Analyst Chat with ReAct Agent

This project implements an interactive chat powered by a **ReAct agent** using the OpenAI API and custom analytics tools.  
It supports follow-up questions, multilingual interaction, and provides a simple user interface built with Streamlit.

---

### Dataset and Capabilities

The AI Data Analyst agent works with the publicly available **Bitext dataset** — Customer Service Tagged Training Dataset for LLM-based Virtual Assistants  
[🔗 Dataset on HuggingFace](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset#bitext---customer-service-tagged-training-dataset-for-llm-based-virtual-assistants)

It analyzes these dataset fields:  
- **instruction**: user requests from the Customer Service domain  
- **category**: high-level semantic category for intent  
- **intent**: specific intent behind the instruction  
- **response**: example expected assistant response  

Typical questions you can ask:  
- What categories exist?  
- What are the most frequent categories?  
- Show 3 examples of responses for Category CANCEL  
- What is Intent distribution?  
- Summarize how the agent responds to Intent CANCEL  

A session lasts for up to 10 interaction steps or until restarted.

---

### Deploying and Running on Streamlit Cloud

To deploy and run this app on [Streamlit Cloud](https://share.streamlit.io):

1. **Push this repository to GitHub.**

2. **Create a new app** on [Streamlit Cloud](https://share.streamlit.io), connecting it to your GitHub repository.

3. **Add your OpenAI API key** in the app's **Secrets** (under “Settings > Secrets”) as:  
   `OPENAI_API_KEY = your_openai_api_key_here`

4. **Deploy the app**. The AI Data Analyst chat will be accessible via your Streamlit Cloud app URL.
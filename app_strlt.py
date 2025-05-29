import streamlit as st
from config import system_prompt
from react_agent import process_user_query_react

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]
if "step" not in st.session_state:
    st.session_state.step = 0
if "session_active" not in st.session_state:
    st.session_state.session_active = True

def reset_session():
    st.session_state.messages = [{"role": "system", "content": system_prompt}]
    st.session_state.step = 0
    st.session_state.session_active = True

st.title("AI Data Analyst Chat")

st.markdown("""
### About this Chat

In this chat you can explore the publicly available **Bitext dataset** — Customer Service Tagged Training Dataset for LLM-based Virtual Assistants  
[🔗 Dataset on HuggingFace](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset#bitext---customer-service-tagged-training-dataset-for-llm-based-virtual-assistants)

The AI Data Analyst works with the following dataset fields:  
- **instruction**: a user request from the Customer Service domain  
- **category**: the high-level semantic category for the intent  
- **intent**: the intent corresponding to the user instruction  
- **response**: an example expected response from the virtual assistant  

**Example questions you can ask the AI Data Analyst:**  
- What categories exist?  
- What are the most frequent categories?  
- Show 3 examples of responses for Category CANCEL  
- What is Intent distribution?  
- Summarize how the agent responds to Intent CANCEL  

A session lasts for a maximum of 10 steps or until you restart it.
""")

if st.button("Restart Session"):
    reset_session()
    st.experimental_rerun()

if st.session_state.session_active:
    with st.form(key="input_form", clear_on_submit=True):
        user_input = st.text_input("Enter your question:")
        submit = st.form_submit_button("Send")

    if submit and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        answer, updated_messages = process_user_query_react(st.session_state.messages)

        st.session_state.messages = updated_messages
        st.session_state.step += 1

        st.markdown(f"**You:** {user_input}")
        st.markdown(f"**AI_Agent:** {answer}")

        if st.session_state.step >= 10:
            st.warning("Reached the maximum of 10 interaction steps. Please restart the session.")
            st.session_state.session_active = False
else:
    st.info("Session ended. Please press 'Restart Session' to start a new conversation.")

st.write("---")

with st.expander("Show conversation history"):
    for i, msg in enumerate(st.session_state.messages):
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        st.write(f"{i+1}. **{role}:**")
        st.text_area("", value=content, height=80, key=f"history_{i}")

import streamlit as st
from openai import OpenAI
import os
import json
#from langchain_openai.chat_models import ChatOpenAI

st.set_page_config(layout="wide", page_title="OpenRouter chatbot app")
st.title("OpenRouter chatbot app")

# api_key, base_url = os.environ["API_KEY"], os.environ["BASE_URL"]
api_key, base_url = st.secrets["API_KEY"], st.secrets["BASE_URL"]
selected_model = "google/gemma-3-1b-it:free"

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"You are a helpful assistant."},
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not api_key:
        st.info("Invalid API key.")
        st.stop()
    client = OpenAI(api_key=api_key, base_url=base_url)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(
        model=selected_model,
        messages=st.session_state.messages
    )
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

# def generate_response(input_text):
#    chat = ChatOpenAI(
#        model=selected_model,
#        openai_api_key=os.environ["API_KEY"],
#        openai_api_base=os.environ["BASE_URL"]
#    )
#    st.info(chat.invoke(input_text))

# with st.form("Prompt"):
#     text = st.text_area(
#         "Enter text:", "How can I help you?"
#     )
#     submitted = st.form_submit_button("Submit")
#     if submitted:
#        generate_response(text)
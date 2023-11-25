import streamlit as st
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from PIL import Image

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0)
    chain = ConversationChain(llm=llm)
    return chain

chain = load_chain()

image = Image.open('frodo.jpg')
st.set_page_config(page_title="Lord of the Rings Demo", page_icon=image, layout="wide")
with st.sidebar:
    choose_character = st.sidebar.selectbox("Select a character", ("Frodo", "Gandalf"),index=None, placeholder="Select a character")
st.header("Lord of the Rings Demo")


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def get_text():
    input_text = st.chat_input(placeholder= "Hello, how are you?", key="input")
    return input_text

user_input = get_text()

if user_input:
    if not choose_character:
        st.error("Please select a character")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant",avatar=image):
        output = chain.run(input=user_input)
        st.markdown(output)
    st.session_state.messages.append({"role": "assistant", "content": output})
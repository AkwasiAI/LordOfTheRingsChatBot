import streamlit as st
from langchain.chains import ConversationalRetrievalChain

from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
import openai
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from PIL import Image
import os 

def setup_qa_chain(user_input):
    # Load documents
    loader = PyPDFLoader("books.pdf")
    data = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(data)

    # Create embeddings and store in vectordb
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(
        search_type='mmr',
        search_kwargs={'k':4, 'fetch_k':4}
    )

    # Setup memory for contextual conversation        
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    #Prompt template 

    general_system_template = r""" 
        Imagine you are Frodo Baggins from Lord of the Rings. Answer the question as best as you can. If you do not know say. Answer in the tone of Frodo Baggins. 
        ----
        {context}
        ----
        """
    general_user_template = "Question:```{question}```"
    messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages( messages )

    combine_docs_chain_kwargs = {'prompt' : qa_prompt}

    # Setup LLM and QA chain
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, streaming=False)
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True, combine_docs_chain_kwargs= combine_docs_chain_kwargs)
    return qa_chain


st.set_page_config(page_title="Lord of the Rings Demo", layout="wide")
with st.sidebar:
    choose_character = st.sidebar.selectbox("Select a character", ("Frodo", "Gandalf"),index=None, placeholder="Select a character")
    openai_api_key = st.sidebar.text_input(
        label="OpenAI API Key",
        type="password",
        value=st.session_state['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in st.session_state else '',
        placeholder="sk-..."
        )
    if openai_api_key:
        st.session_state['OPENAI_API_KEY'] = openai_api_key
        os.environ['OPENAI_API_KEY'] = openai_api_key
    else:
        st.error("Please add your OpenAI API key to continue.")
        st.info("Obtain your key from this link: https://platform.openai.com/account/api-keys")
        st.stop()
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
    character_chatbot = Image.open(f"{choose_character.lower()}.jpg")

    st.session_state.messages.append({"role": "user", "content": character_chatbot})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant",avatar=character_chatbot):
        qa_chain = setup_qa_chain(user_input)
        output = qa_chain.run({"question": user_input})
        st.markdown(output)
    st.session_state.messages.append({"role": "assistant", "content": output})
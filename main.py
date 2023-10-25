#%% ---------------------------------------------  IMPORTS  ----------------------------------------------------------#
import time
import datetime
import tempfile
from PyPDF2 import PdfReader
import nbformat
import docx2txt
import glob
import os

# Streamlit Imports
import streamlit as st
from streamlit_chat import message

# LangChain Imports
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Langchain Imports
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory

# Mistral Imports
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import OllamaEmbeddings

import chromadb

embeddings_open = OllamaEmbeddings(model="mistral", temperature=0)
llm_open = Ollama(model="mistral", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
#%% --------------------------------------------  FILE UPLOADER  -----------------------------------------------------#
def process_files_in_folder(folder_path):
    ''' This function handles the file processing in the local folder and returns the text of the file.'''

    # Get all files in the folder
    file_paths = glob.glob(os.path.join(folder_path, '*'))
    file_names = [os.path.basename(file_path) for file_path in file_paths]

    combined_text = ""
    if folder_content:
        for file_path in file_paths:
            file_extension = os.path.splitext(file_path)[-1].strip('.')
            try:
                if file_extension in ["pdf"]:
                    with open(file_path, "rb") as f:
                        pdf = PdfReader(f)
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            combined_text += str(page_text) + "\n"

                elif file_extension in ["docx"]:
                    docx = Docx2txtLoader(file_path)
                    pages = docx.load()
                    for page in pages:
                        combined_text += page.page_content + "\n"

                elif file_extension in ["html", "css", "py", "txt"]:
                    with open(file_path, "r") as f:
                        file_content = f.read()
                    st.code(file_content, language=file_extension)
                    combined_text += file_content + "\n"

                elif file_extension == "ipynb":
                    with open(file_path, "r") as f:
                        nb_content = nbformat.read(f, as_version=4)
                    nb_filtered = [cell for cell in nb_content["cells"] if cell["cell_type"] in ["code", "markdown"]]
                    for cell in nb_filtered:
                        if cell["cell_type"] == "code":
                            st.code(cell["source"], language="python")
                        elif cell["cell_type"] == "markdown":
                            st.markdown(cell["source"])
                            combined_text += cell["source"] + "\n"

            except Exception as e:
                raise Exception(f"Error reading {file_extension.upper()} file: {e}")

    return combined_text

#%% --------------------------------------------  FUNCTIONS  ---------------------------------------------------------#
# Create your own prompt by using the template below.
def build_prompt(template_num="template_1"):
    template = """You are a helpful chatbot about legal documents. Provide answers from the chat history and the context. 
    If you can not answer the question from the given contexts, say that there is no knowledge about it.

    Chat history:
    {chat_history}

    Context: 
    {context}

    Question: 
    {question}

    Helpful Answer:"""

    if template_num == "template_1":
        prompt = PromptTemplate(input_variables=["chat_history", "context", "question"], template=template)
        return prompt

    else:
        print("Please choose a valid template")



#%% --------------------------------------------  PAGE CONTENT  ------------------------------------------------------#
st.set_page_config(page_title="Home", layout="wide")
st.sidebar.image("rslt_logo_dark_mode.png", width=200)
st.sidebar.title("")
st.sidebar.title("Info:")
st.sidebar.write("""Use the newest AI technology to chat fully local with an open source LLM! Summarize and retrieve content from 
documents and create your own knowledge base. """)

# ---------------------- MAIN PAGE -------------------- #
st.sidebar.title("Choose application type and settings:")
st.title("Law Search LLM")
st.subheader("Instructions:")
with st.expander("Instructions", expanded=False):
    st.write(f"""1. Choose an application in the sidebar.
    - Chat with Mistral AI freely
    - Use RAG to retrieve and insert content from your documents to your local vector database
    - Use the vector database to retrieve the saved knowledge

2. Select a database and collection in the sidebar for the RAG and Knowledge Base Application
3. Upload a document with the Mistral RAG application to insert it into the database
4. When reopening the application, you can retrieve the content from the database with the Knowledge Base application
""")


# ---------------------- VECTOR DB -------------------- #
folder_selector = st.sidebar.selectbox("Select a folder", options=["./data/german/", "./data/dutch/"])


vector_db_path_selector = st.sidebar.selectbox("Select a database", options=["./chroma/german", "./chroma/dutch"])
collection_selector = st.sidebar.selectbox("Select a collection", options=["collection_1", "collection_2", "collection_3"])

# ---------------------- SHOW FOLDER  -------------------- #
folder_content = st.sidebar.button("Show folder content")


if folder_content:
    with st.expander("Document Expander (Press button on the right to fold or unfold)", expanded=True):
        text = process_files_in_folder(folder_selector)
        st.subheader("Uploaded Document:")
        st.write("Text: ", text)


# ---------------------- FILE UPLOAD  -------------------- #
add_button = st.sidebar.button(label="Add folder to database")

if add_button:
    with st.spinner("Adding files to database ..."):
        text = process_files_in_folder(folder_selector)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200, separators=[" ", "\n"])
        chunks = text_splitter.split_text(text)
        st.write("Chunks: ", chunks)



if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

llm_open = Ollama(model="mistral", temperature=0, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
embeddings_open = OllamaEmbeddings(model="mistral")

chroma_client = chromadb.PersistentClient(path=vector_db_path_selector)
collection = chroma_client.get_or_create_collection(name=collection_selector)

knowledge_base = Chroma(persist_directory=vector_db_path_selector, embedding_function=embeddings_open, collection_name=collection_selector)


memory = ConversationSummaryBufferMemory(
        llm=llm_open,
        return_messages=True,
        max_token_limit=500,
        memory_key= "chat_history",
        human_prefix= "### Input",
        ai_prefix= "### Response",
        output_key= "answer",
        return_chat_history= True)

# Initialize chain
if "chain" not in st.session_state:
    chain = ConversationalRetrievalChain.from_llm(
        llm_open,
        chain_type="stuff",
        retriever=knowledge_base.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": build_prompt("template_1")},
        verbose=True)
    st.session_state["chain"] = chain



# --------------------- USER INPUT --------------------- #
user_input = st.text_area("Your text: ")
if user_input:

    if user_input:
        transcript = user_input

    if 'transcript' not in st.session_state:
        st.session_state.transcript = transcript

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = [{"role": "user", "content": transcript}]
    else:
        st.session_state['chat_history'].append([{"role": "user", "content": transcript}])

    # ------------------- TRANSCRIPT ANSWER ----------------- #
    with st.spinner("Fetching answer ..."):

        # ------------------- KNOWLEDGE BASE ----------------- #
        history = st.session_state['chat_history']
        result = st.session_state.chain({"question": transcript, "chat_history": history})
        answer = result["answer"]
        st.info("Number Docs found: " + str(len(result["source_documents"])))
        #st.info("Sources: " + str(result["source_documents"]))
        #st.info("Sources: " + str(result["source_documents"][0]))

        # MEMORY BUFFER
        ##st.code("Buffer Memory: ", st.session_state.chain.memory.chat_memory.messages)
        with st.expander("Document Knowledge"):
            client = chromadb.PersistentClient(path=vector_db_path_selector)
            collection = client.get_or_create_collection(name=collection_selector)
            st.dataframe(collection.get(), use_container_width=True, height=200)

        with st.expander("Buffer Memory"):
            st.write(st.session_state.chain.memory.load_memory_variables({}))


        st.session_state.past.append(transcript)
        st.session_state.generated.append(answer)
        st.session_state['chat_history'].append({"role": "assistant", "content": answer})

if 'chat_history' in st.session_state:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

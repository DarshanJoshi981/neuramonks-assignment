import re
import sys
import pickle
import streamlit as st
from streamlit_chat import message
import os
import pandas as pd

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms import HuggingFaceHub

from dotenv import load_dotenv
load_dotenv()

llm_model = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"temperature": 0.5, "max_new_tokens": 1000})

def read_csv_to_text(file_path, batch_size=10000):
    raw_text = ""
    try:
        for batch in pd.read_csv(file_path, chunksize=batch_size):
            processed_rows = process_batch(batch)
            if processed_rows is not None:
                raw_text += "\n".join(processed_rows) + "\n"
    except FileNotFoundError:
        print("File not found.")
    except pd.errors.ParserError as e:
        print(f"ParserError: {e}")

    return raw_text

def process_batch(batch):
    processed_rows = []
    for _, row in batch.iterrows():
        try:
            processed_row = row_to_text(row)
            processed_rows.append(processed_row)
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    return processed_rows if processed_rows else None

def row_to_text(row):
    row_text = ""
    for key, value in row.items():
        row_text += f"{key}: {value}, "
    return row_text[:-2]


# @st.cache_resource
def create_vector_store(file_path, file_type):

    if file_type == 'pdf':
        pdf_loader = PyPDFLoader(file_path)
        docs = pdf_loader.load()
        raw_text = ''
        for doc in docs:
            raw_text += doc.page_content

    elif file_type == 'csv':
        raw_text = read_csv_to_text(file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=0
    )
    texts = text_splitter.split_text(raw_text)

    docs = [Document(page_content=t) for t in texts]
    vectorstore_faiss = FAISS.from_documents(
        documents=docs,
        embedding=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base"),
    )
    return vectorstore_faiss


def create_prompt_template():
    prompt_template = """
    Human: Answer the question as a full sentence from the context provided. If you don't know the answer, don't try to make up an answer.
    <context>
    {context}
    </context>
    Question: {question}
    Assistant:
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"], template=prompt_template
    )
    return prompt


# @st.cache_resource
def create_retrieval_chain(vector_store, prompt_template):
    qa = RetrievalQA.from_chain_type(
        llm = llm_model,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        chain_type_kwargs={"prompt": prompt_template},
    )

    return qa


def generate_response(chain, input_question):
    answer = chain({"query": input_question})
    return answer['result'].strip()


# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        if len(history["generated"][i]) == 0:
            message("Please reframe your question properly", key=str(i))
        else:
            message(history["generated"][i],key=str(i))


def create_folders_if_not_exist(*folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def main():
    st.set_page_config(
        page_title="Advanced RAG using LLM",
        page_icon=":mag_right:",
        layout="wide"
    )

    st.title("Advanced RAG using LLM - Neuramonks Assignment")

    # Sidebar for file upload
    st.sidebar.title("Upload File")
    uploaded_file = st.sidebar.file_uploader("", label_visibility='collapsed', type=["pdf", "csv"])

    create_folders_if_not_exist("data", "data/pdfs", "data/csv", "data/vectors")

    if "uploaded_file" not in st.session_state or st.session_state.uploaded_file != uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.generated = [f"Ask me a question about {uploaded_file.name}" if uploaded_file else ""]
        st.session_state.past = ["Hey there!"]
        st.session_state.last_uploaded_file = uploaded_file.name if uploaded_file else None

    if uploaded_file is not None:
        file_type = uploaded_file.type.split("/")[-1]  # Get file type from MIME type
        filepath = "data/pdfs/" + uploaded_file.name if file_type == "pdf" else "data/csv/" + uploaded_file.name
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
        vector_file = os.path.join('data/vectors/', f'vector_store_{uploaded_file.name}.pkl')

        # Display the uploaded file name in the sidebar
        st.sidebar.markdown(f"**Uploaded file:** {uploaded_file.name}")

        if not os.path.exists(vector_file) or "ingested_data" not in st.session_state:
                with st.spinner('Embeddings are in process...'):
                    try:
                        ingested_data = create_vector_store(filepath, file_type)
                        with open(vector_file, "wb") as f:
                            pickle.dump(ingested_data, f)
                        st.session_state.ingested_data = ingested_data
                        st.success('Embeddings are created successfully! ✅✅✅')
                    except:
                        st.warning("Looks like a scanned PDF. Please upload a Digital PDF")
                        sys.exit()

        else:
            ingested_data = st.session_state.ingested_data

        prompt = create_prompt_template()
        chain = create_retrieval_chain(ingested_data, prompt)

        user_input = st.chat_input(placeholder="Ask a question")

        if user_input:
            answer = generate_response(chain, user_input)
            st.session_state.past.append(user_input)
            response = answer
            st.session_state.generated.append(response)

        # Display conversation history using Streamlit messages
        if st.session_state.generated:
            display_conversation(st.session_state)

if __name__ == "__main__":
    main()


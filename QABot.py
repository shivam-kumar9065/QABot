# pip install langchain langchain-google-genai PyMuPDF python-docx streamlit faiss-cpu tiktoken

import os
import ssl
import certifi
import fitz  
import docx
import streamlit as st
import time

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


ssl._create_default_https_context = ssl.create_default_context
ssl._create_default_https_context().load_verify_locations(certifi.where())


os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = certifi.where()

os.environ["GOOGLE_API_KEY"] = "Your API Key "  


def extract_text_from_pdf(file):
    text = ""
    pdf_bytes = file.read()
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in pdf:
        text += page.get_text()
    pdf.close()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join(para.text for para in doc.paragraphs)


def extract_text_from_files(files):
    full_text = ""
    for file in files:
        filename = file.name.lower()
        if filename.endswith(".pdf"):
            full_text += extract_text_from_pdf(file)
        elif filename.endswith(".docx"):
            full_text += extract_text_from_docx(file)
        else:
            st.warning(f"Unsupported file type: {filename}")
    return full_text

#Retry Function 
def retry_request(func, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt < retries - 1:
                st.warning(f"Error: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                st.error(f"Failed after {retries} attempts: {e}")
                raise e


#QA Chain
def build_doc_qa_chain(document_text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(document_text)

    def create_vectorstore():
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return FAISS.from_texts(chunks, embedding=embeddings)

    vectorstore = retry_request(create_vectorstore)

    llm = GoogleGenerativeAI(model="models/gemini-pro") 

    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know.

{context}

Question: {question}
Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)

    return vectorstore, chain

def ask_question(vectorstore, chain, question):
    docs = vectorstore.similarity_search(question)
    return chain({"input_documents": docs, "question": question}, return_only_outputs=True)["output_text"]


st.set_page_config(page_title="📄 Document Q&A Bot", layout="centered")
st.title("📄 Ask Questions About Your Documents")

uploaded_files = st.file_uploader("Upload PDF or DOCX files", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("🔍 Processing documents..."):
        document_text = extract_text_from_files(uploaded_files)
        try:
            vectorstore, qa_chain = build_doc_qa_chain(document_text)
            st.success("✅ Documents loaded! Ask your questions below.")
        except Exception as e:
            st.error(f"An error occurred while processing documents: {e}")

    user_question = st.text_input("💬 Your Question:")

    if user_question:
        with st.spinner("🤖 Thinking..."):
            try:
                answer = ask_question(vectorstore, qa_chain, user_question)
                st.markdown(f"**🧠 Answer:** {answer}")
            except Exception as e:
                st.error(f"An error occurred while retrieving the answer: {e}")

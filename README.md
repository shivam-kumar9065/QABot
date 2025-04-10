# 📄 Document Q&A Bot

Ask questions about your PDF and DOCX documents using the power of **Google Gemini Pro**, **LangChain**, and **Streamlit**. This app helps you search, understand, and interact with long documents instantly — no scrolling or manual reading needed!

---

## 🚀 Features

- 📂 Upload multiple `.pdf` or `.docx` files
- 🤖 Extracts and processes full document content
- 🧠 Uses Google Gemini Pro for context-aware answers
- 🗂️ Embeds and stores chunks using FAISS vector database
- 💬 Interactive Q&A through a clean Streamlit interface
- 🔁 Built-in retry logic for API robustness

---

## 📦 Installation

> Make sure you're using **Python 3.8+**

Install all required packages with pip:
    pip install langchain langchain-google-genai PyMuPDF python-docx streamlit faiss-cpu tiktoken
    
    
📁 Project Structure
      .
      ├── app.py                # Streamlit app entry point
      ├── requirements.txt      # List of required dependencies
      └── README.md             # Project documentation



🔑 Google API Key Setup
To use Google Gemini APIs, you need an API key from Google AI Studio.

🖥️ Running the App
To launch the Streamlit app, run the following command in your terminal:
  streamlit run app.py


🧠 How It Works
    1. Upload Files: Upload your PDF or DOCX files.
    
    2. Text Extraction: The app automatically extracts all text from the files.
    
    3. Text Chunking: Large text content is chunked into smaller pieces using CharacterTextSplitter.
    
    4. Embedding: Each chunk is embedded using GoogleGenerativeAIEmbeddings.
    
    5. Vector Indexing: A FAISS vector index is built to enable semantic similarity search.
    
    6. Q&A: When you ask a question, the app retrieves the most relevant text chunks and uses Google Gemini Pro to provide an answer based on those chunks.

 📁 Supported File Types
      File Type	Supported
      .pdf	✅ Yes
      .docx	✅ Yes
      .txt	❌ No
      .doc	❌ No



  💡 Example Use Cases
        1. 🔍 Research Papers – Quickly search for answers without reading the entire document.
        
        2. 📜 Legal Documents – Ask questions about specific clauses, terms, or dates.
        
        3. 📊 Business Reports – Extract key insights from lengthy reports and presentations.
        
        4. 📝 Study Notes – Review and clarify your class notes in seconds.




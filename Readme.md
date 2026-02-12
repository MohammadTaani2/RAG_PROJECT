# 🤖 RAG Course Assistant Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using:

- 🧠 OpenAI (Embeddings + GPT-4o-mini)
- 🌲 Pinecone (Vector Database)
- 🖥 Streamlit (Web Interface)

This project was developed for the **GEN AI Course**.

---

## 📌 Project Overview

This system allows students to ask questions about course materials (PDFs).

The chatbot:
1. Extracts text from course PDFs
2. Splits them into chunks
3. Converts chunks into embeddings using OpenAI
4. Stores them in Pinecone
5. Retrieves relevant chunks when a question is asked
6. Generates an answer using GPT-4o-mini

It ensures answers are based only on the provided course materials.

---

## 🏗️ Architecture

User Question  
↓  
Generate Embedding  
↓  
Pinecone Similarity Search  
↓  
Retrieve Top-K Relevant Chunks  
↓  
Send Context + Question to GPT  
↓  
Generate Final Answer  

---

## 📂 Project Structure

```
mohammadtaani2-rag_project/
├── app1.py            # Streamlit chatbot application
├── ingest.py          # PDF ingestion & vector indexing pipeline
├── PDFs/              # Course material PDFs (data source)
└── README.md
```

---

## ⚙️ Technologies Used

- Python
- Streamlit
- OpenAI API
- Pinecone (Serverless Index)
- PyPDF2
- Requests
- dotenv

---

## 🔑 Environment Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/mohammadtaani2-rag_project.git
cd mohammadtaani2-rag_project
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If you don’t have a requirements file, install manually:

```bash
pip install streamlit pinecone-client PyPDF2 requests python-dotenv
```

---

### 3️⃣ Create a `.env` File

Inside your project folder, create a file named `.env`:

```
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
```

---

## 📥 Step 1: Ingest PDFs into Pinecone

Before running the chatbot, you must index your course PDFs.

Place your PDF files in the folder defined in `ingest.py`:

```python
pdf_folder = r"C:\Users\taani\OneDrive\Desktop\RAG_PROJECT\PDFs"
```

Then run:

```bash
python ingest.py
```

This will:
- Load PDFs
- Chunk text
- Generate embeddings
- Create Pinecone index (`myind`)
- Upload vectors

---

## 🚀 Step 2: Run the Chatbot

```bash
streamlit run app1.py
```

Then open the local URL shown in terminal (usually):

```
http://localhost:8501
```

---

## 🧠 How the RAG System Works

### 1️⃣ Embedding Generation
Uses OpenAI model:

```
text-embedding-3-small
```

Each chunk becomes a 1536-dimension vector.

---

### 2️⃣ Vector Search (Pinecone)

- Index Name: `myind`
- Metric: cosine similarity
- Top K results: 10

---

### 3️⃣ LLM Response Generation

Model used:

```
gpt-4o-mini
```

System Prompt ensures:
- Only use provided course materials
- Do not hallucinate
- Match user's language

---

## 💡 Features

✔ Retrieval-Augmented Generation  
✔ Conversation Memory  
✔ Semantic Search  
✔ Clean UI with Custom CSS  
✔ Error Handling  
✔ Serverless Pinecone Index  

---

## 📊 Example Workflow

Student asks:

> "What is backpropagation?"

System:
- Converts question to embedding
- Retrieves top relevant chunks
- Sends chunks + question to GPT
- Returns answer grounded in course materials

---

## 🔮 Future Improvements

- Add document upload from UI
- Add citation display (show source chunk)
- Use streaming responses
- Add conversation reset button
- Improve chunking strategy
- Deploy to Streamlit Cloud

---

## 👨‍🎓 Author

**Mohammad AlTa'any**  
GEN AI Course  
December 27, 2025  

---

## 📜 License

This project is for educational purposes.
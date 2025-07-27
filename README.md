# Context-Aware FAQ Chatbot with LangGraph and RAG

A smart FAQ chatbot that understands context and retrieves precise answers using LangGraph, Retrieval-Augmented Generation (RAG), and LLMs (Large Language Models). This project allows organizations to build conversational agents that can intelligently answer user queries from a custom document (like `faqs.txt`).

---

## 🔍 What is This Project?

This is a **context-aware FAQ chatbot** that:
- Uses **LangGraph** to structure the conversational flow
- Implements **RAG (Retrieval-Augmented Generation)** to enhance LLM responses with factual grounding
- Leverages **HuggingFace embeddings** and **LangChain** to semantically search content from a custom knowledge base (e.g., FAQs)
- Runs via **Streamlit** as a simple user interface

---

## ⚙️ What’s Going On Inside?

1. **Data Loading**: Reads FAQs from a text file (`faqs.txt`)
2. **Embedding & Indexing**: Converts data to vector embeddings using HuggingFace, stores in in-memory vectorstore
3. **Query Handling**: Accepts a user question, retrieves relevant context from the knowledge base
4. **Answer Generation**: Passes query + context to an LLM via LangGraph’s RAG pipeline
5. **Frontend**: A Streamlit web app lets users interact with the chatbot in real time

---

## 🎯 Purpose

The purpose of this chatbot is to:
- Automate FAQs for any domain (support, HR, product, etc.)
- Provide grounded, context-aware answers using GenAI
- Showcase a practical use of LangGraph, RAG, and LangChain in a deployable solution

---

## 💼 Use Cases

- **Customer Support Chatbot** trained on support FAQs  
- **HR Chatbot** for internal policies and employee queries  
- **Product Info Bot** trained on manuals or help docs  
- **Educational Assistant** for concept-based Q&A from course notes

---

## 📂 Files in the Project

- `app.py` — Main Streamlit app
- `README.md` 
- `faqs.txt` — *(Not included)* You must add this manually with your FAQs

## 📌 Future Improvements

- Add UI enhancements  
- Support PDF and Docx file ingestion  
- Enable feedback/rating system for chatbot answers 

---

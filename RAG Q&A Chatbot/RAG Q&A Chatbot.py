import streamlit as st
import os 
import sys
import json 
import boto3
from langchain_community.embeddings import ollamaEmbeddings
from langchain_community.vectorstores import faiss
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import create_retrival_chain
from langchain.chains import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests

def query_llm(prompt, url, api_key):
    
    payload = json.dumps({
  "project_id": "training",
  "auth_key": "",
  "messages": [
   
    {
      "role": "System",
      "content": prompt

    }
  ],
  "service": "",
  "model": "",
  "temperature": 0.01,
  "max_tokens": 4096,
  "app_id": "docai-extract"

})
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()  # Adjust this based on the expected response format
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")




from langchain.llms import BaseLLM

class CustomLLM(BaseLLM):
    def __init__(self, url, api_key):
        self.url = url
        self.api_key = api_key

    def _call(self, prompt, stop=None):
        return query_llm(prompt, self.url, self.api_key)






prompt = ChatPromptTemplate.from_strings(
    """
       Answer the Question based on the context provided only.
       please provide the most accurate response based on the question

       <context>
       {context}
       <context>

       Question:{Input}

   """

)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=ollamaEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("Integration Docs") ## Data ingestion
        st.session_state.docs = st.session_state.loader.load()  ##Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = faiss.from_documents(st.session_state.final_documents,st.session_state.embeddings)

user_prompt = st.text_input("Enter your query from integration document")

# Initialize your custom LLM
custom_llm = CustomLLM(url="", api_key="")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("vector Database is ready")

import time

if user_prompt:
    document_chain = create_stuff_documents_chain(custom_llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrival_chain(retriever,document_chain)


    start = time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt})
    print(f"Response time:{time.process_time()-start}")

    st.write(response['answer'])


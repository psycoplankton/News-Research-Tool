import os
import streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourceChains
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

#load all the environment variables from .env file 
load_dotenv() 

# Instantiate LLM with required params
llm = OpenAI(temperature = 0.9, max_tokens=500)


#Define the app layout
st.title("News Research Tool")
st.sidebar.title("News Article URLs")

urls=[]
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "vetorstore.pkl"

main_placefolder = st.empty()


if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data Loading .........")

    data = loader.load()
    #split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size = 1000
    )

    main_placefolder.text("Text Splitting Started ...........")
    docs = text_splitter.split_documents(data)
    #create embeddings 
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    #Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)
    main_placefolder.text("Embedding vectors savec at {file_path} ........")


query = main_placefolder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load()
            chain = RetrievalQAWithSourceChains.from_llm(llm = llm, retriever=vectorstore_openai.as_retriever())
            result = chain({"Question": query}, return_only_outputs = True)
            #result = {"answer": "", "sources": []}
            
            st.header("Answer")
            st.subheader(result["answer"])

            #display sources
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)



 
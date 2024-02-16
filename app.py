from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import sys

DB_FAISS_PATH="vectors/db_faiss"
loader=CSVLoader(file_path="/Users/shaikmujeeburrahman/Downloads/llama2_csv_llm/data/tips.csv",encoding="utf-8", csv_args={'delimiter': ','})
data=loader.load()
# print(data)



# split the text into chunks

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
chunks=text_splitter.split_documents(data)

# Download Sentence Transformers Embedding From Hugging Face
embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

# COnverting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
docsearch = FAISS.from_documents(chunks, embeddings)

docsearch.save_local(DB_FAISS_PATH)

# query = "What is the highest tip amount paid by a customer?"

# docs = docsearch.similarity_search(query, k=3)

# # print("Result", docs)

llm = CTransformers(model="/Users/shaikmujeeburrahman/Downloads/llama-2-7b-chat.ggmlv3.q4_0.bin",model_type="llama",
 max_new_tokens=512,
 temperature=0.1)

qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())


import streamlit as st

def get_response(query):
    chat_history=[]
    result = qa({"question": query, "chat_history": chat_history})
    return result['answer']

def main():
    st.image("/Users/shaikmujeeburrahman/Desktop/Meta.jpg")
    st.title("Conversational Retrieval System using LLama2 Quantized Model")
    st.subheader("Created by Mujeeb")
    st.write("Enter your question below:")

    query = st.text_input("Input Prompt:")

    if query:
        result = get_response(query)
        st.write("Response:", result)

if __name__ == "__main__":
    main()



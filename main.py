import os
import streamlit as st
import pickle
import time
import faiss
from dotenv import load_dotenv

from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# Load environment variables
load_dotenv()  # Required for OpenAI API key

# Streamlit UI
st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Collect URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
faiss_index_path = "faiss_index.bin"
metadata_path = "faiss_metadata.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # Load data from URLs
    loader = UnstructuredURLLoader(urls=[url for url in urls if url])  # Exclude empty inputs
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings and save to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save FAISS index separately
    faiss.write_index(vectorstore_openai.index, faiss_index_path)

    # Save metadata separately
    metadata = {
        "docstore": vectorstore_openai.docstore,
        "index_to_docstore_id": vectorstore_openai.index_to_docstore_id
    }

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

# Query input
query = st.text_input("Question: ")

if query:
    if os.path.exists(faiss_index_path) and os.path.exists(metadata_path):
        # Load FAISS index
        faiss_index = faiss.read_index(faiss_index_path)

        # Load metadata
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        # Reconstruct FAISS VectorStore
        vectorstore = FAISS(
            index=faiss_index,
            docstore=metadata["docstore"],
            index_to_docstore_id=metadata["index_to_docstore_id"],
            embedding_function=OpenAIEmbeddings()  # Ensure embedding function is provided
        )

        # Create retrieval chain
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

        # Execute query
        result = chain({"question": query}, return_only_outputs=True)

        # Display answer
        st.header("Answer")
        st.write(result["answer"])

        # Display sources
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)

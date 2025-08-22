import os
from pathlib import Path
import streamlit as st
from bs4 import BeautifulSoup

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders.sitemap import SitemapLoader

# Initialize Streamlit page
# st.set_page_config(page_title="Infant Food Recipes")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Validate required environment variables
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
DOC_DIR_PATH = os.environ.get("DOC_DIR_PATH", "docs")

# Check for required API keys
if not PINECONE_API_KEY:
    st.error("❌ PINECONE_API_KEY not found in environment variables")
    st.stop()

if not PINECONE_INDEX:
    st.error("❌ PINECONE_INDEX not found in environment variables")
    st.stop()

# Ensure docs directory exists
if not os.path.exists(DOC_DIR_PATH):
    os.makedirs(DOC_DIR_PATH, exist_ok=True)

# Function to load documents from docs folder
def load_documents():
    if not os.path.exists(DOC_DIR_PATH):
        st.error(f"Docs folder not found at: {DOC_DIR_PATH}")
        return []

    documents = []
    # Manually iterate to catch per-file errors and continue
    for root, _, files in os.walk(DOC_DIR_PATH):
        for fname in files:
            if fname.lower().endswith('.docx'):
                fpath = os.path.join(root, fname)
                try:
                    loader = Docx2txtLoader(fpath)
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    st.warning(f"⚠️ Skipped '{fpath}': {e}")
    print(f"Loaded {len(documents)} documents from docs folder")
    return documents

# Function to split documents into texts
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=50, separators=[" ", ",", "\n"])
    texts = text_splitter.split_documents(documents)
    return texts

# Function to remove navigation and header elements from HTML content
def remove_nav_and_header_elements(content: BeautifulSoup) -> str:
    nav_elements = content.find_all("nav")
    header_elements = content.find_all("header")

    for element in nav_elements + header_elements:
        element.decompose()

    return str(content.get_text())

# Function to upload embeddings to Pinecone
def embeddings_on_pinecone(texts):
    # Use a 1024-dimensional model to match the Pinecone index dimension (1024)
    # Examples of 1024-d models: "intfloat/e5-large", "BAAI/bge-large-en-v1.5"
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-large",
        encode_kwargs={"normalize_embeddings": True}
    )
    PineconeVectorStore.from_documents(texts, embeddings, index_name=PINECONE_INDEX)

# Function to process uploaded documents - saves directly to docs folder
def process_uploaded_documents():
    try:
        # Save uploaded files directly to docs folder
        for source_doc in st.session_state.source_docs:
            save_path = Path(DOC_DIR_PATH, source_doc.name)
            with open(save_path, mode='wb') as w:
                w.write(source_doc.getvalue())
            st.info(f"Saved {source_doc.name} to docs folder")
        
        # Process all documents in docs folder
        documents = load_documents()
        if documents:
            texts = split_documents(documents)
            embeddings_on_pinecone(texts)
            st.success(f"Successfully processed {len(documents)} documents and created embeddings!")
        else:
            st.warning("No documents found after upload")
    except Exception as e:
        st.error(f"An error occurred processing uploads: {e}")

# Function to process existing documents from docs folder
def process_docs_folder():
    try:
        documents = load_documents()
        if documents:
            texts = split_documents(documents)
            embeddings_on_pinecone(texts)
            st.success(f"Successfully processed {len(documents)} documents from docs folder and created embeddings!")
        else:
            st.warning("No documents found in docs folder")
    except Exception as e:
        st.error(f"An error occurred processing docs folder: {e}")

# Function to process documents from a sitemap URL
def process_sitemapdocs():
    try:
        print(f"Processing sitemap: {st.session_state.sitemapurl}")
        sitemap_loader = SitemapLoader(
            st.session_state.sitemapurl,
            parsing_function=remove_nav_and_header_elements
        )
        documents = sitemap_loader.load()
        if documents:
            texts = split_documents(documents)
            embeddings_on_pinecone(texts)
            st.success(f"Successfully processed {len(documents)} documents from sitemap and created embeddings!")
        else:
            st.warning("No documents were loaded from the sitemap")
    except Exception as e:
        st.error(f"An error occurred processing sitemap: {e}")

# Function to initialize the application
def boot():
    st.title("🚀 Train Your Own Data")
    st.markdown("Upload documents or process existing docs to create vector embeddings for your AI system.")
    
    # Create tabs for different data sources
    tab1, tab2, tab3 = st.tabs(["📁 Upload Documents", "📂 Process Docs Folder", "🌐 Sitemap URL"])
    
    with tab1:
        st.header("Upload DOCX Documents")
        st.info(f"📁 Files will be saved to: `{DOC_DIR_PATH}`")
        st.session_state.source_docs = st.file_uploader(
            label="Select DOCX files to upload to docs folder and process", 
            type="docx", 
            accept_multiple_files=True
        )
        
        if st.button("Upload & Process Documents", key="upload_btn"):
            if st.session_state.source_docs:
                with st.spinner("Uploading and processing documents..."):
                    process_uploaded_documents()
            else:
                st.warning("Please upload at least one DOCX file")
    
    with tab2:
        st.header("Process Existing Documents")
        st.info(f"📍 Processing documents from: `{DOC_DIR_PATH}`")
        
        # Show files in docs folder
        if os.path.exists(DOC_DIR_PATH):
            docx_files = [f for f in os.listdir(DOC_DIR_PATH) if f.endswith('.docx')]
            if docx_files:
                st.write(f"**Found {len(docx_files)} DOCX files:**")
                for file in docx_files:
                    st.write(f"• {file}")
            else:
                st.warning("No DOCX files found in docs folder")
        else:
            st.error(f"Docs folder does not exist: {DOC_DIR_PATH}")
        
        if st.button("Process Existing Documents", key="docs_btn"):
            with st.spinner("Processing existing documents..."):
                process_docs_folder()
    
    with tab3:
        st.header("Process Sitemap URL")
        st.session_state.sitemapurl = st.text_area(
            "Enter the sitemap URL to crawl and process", 
            key="sitemap_input",
            placeholder="https://example.com/sitemap.xml"
        )
        
        if st.button("Process Sitemap", key="sitemap_btn"):
            if st.session_state.sitemapurl and st.session_state.sitemapurl.strip():
                with st.spinner("Processing sitemap documents..."):
                    process_sitemapdocs()
            else:
                st.warning("Please enter a valid sitemap URL")

if __name__ == '__main__':
    boot()

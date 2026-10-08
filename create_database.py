from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


# --------------------------------------------------
# LOAD PDF
# --------------------------------------------------

loader = PyPDFLoader(
    "data/career_guidance.pdf"
)

documents = loader.load()

print("PDF loaded successfully!")

print(
    "Number of pages:",
    len(documents)
)


# --------------------------------------------------
# SPLIT PDF INTO CHUNKS
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(

    chunk_size=1000,

    chunk_overlap=200
)

chunks = text_splitter.split_documents(
    documents
)

print(
    "Number of chunks:",
    len(chunks)
)


# --------------------------------------------------
# CREATE EMBEDDINGS
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(

    model="gemini-embedding-001"
)


# --------------------------------------------------
# CREATE CHROMADB
# --------------------------------------------------

db = Chroma.from_documents(

    documents=chunks,

    embedding=embeddings,

    persist_directory="./chroma_db"
)


print("Embeddings created successfully!")

print("ChromaDB created successfully!")
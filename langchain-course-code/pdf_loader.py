import os
from dotenv import load_dotenv, find_dotenv
import glob
from langchain_openai import ChatOpenAI  # updated import
from langchain.document_loaders import PyPDFLoader  # or UnstructuredPDFLoader

# Load .env variables
load_dotenv(find_dotenv())

# Setup LLM
llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

# Path to data folder
pdf_folder = os.path.join(os.getcwd(), 'data')
print(f"Looking in folder: {pdf_folder}")

if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)
    print("📁 Created 'data' folder. Add some PDFs and rerun.")

print("Files in folder:", os.listdir(pdf_folder))

# Find PDFs
pdf_paths = glob.glob(os.path.join(pdf_folder, "*.pdf"))
print(f"✅ Found {len(pdf_paths)} PDF(s): {pdf_paths}")

all_pages = []
for pdf_path in pdf_paths:
    print(f"🔍 Loading: {pdf_path}")
    loader = PyPDFLoader(pdf_path)  # or UnstructuredPDFLoader for scanned PDFs
    pages = loader.load()
    print(f"   -> Loaded {len(pages)} page(s)")
    all_pages.extend(pages)

print(f"📄 Total pages loaded: {len(all_pages)}")

if all_pages:
    print(all_pages[0].page_content[:700])
else:
    print("⚠️ No pages found! Add readable PDFs.")

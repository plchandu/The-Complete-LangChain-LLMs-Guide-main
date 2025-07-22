import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# Setup LLM
llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

# Use script directory to avoid cwd issues
script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_folder = os.path.join(script_dir, 'data')

# Get all .pdf files in folder
pdf_paths = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
print(f"✅ Found {len(pdf_paths)} PDF(s): {pdf_paths}")

# Load pages from all PDFs
all_pages = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    pages = loader.load()
    all_pages.extend(pages)

# Print the first page content
if all_pages:
    print("📄 First page content:\n")
    print(all_pages[0].page_content[:700])
else:
    print("⚠️ No pages loaded!")

import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from pydantic import BaseModel, Field
from typing import List

# Load OpenAI API key
load_dotenv()
llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

# ==== Sample emails ====
emails = [
    """Subject: Travel Itinerary\nFrom: alice@example.com\nDate: July 1, 2025\nWe are flying to Paris and Rome this summer vacation.""",
    """Subject: Insurance Claim\nFrom: bob@insurance.com\nDate: July 3, 2025\nThere was a minor fire in the kitchen. We need to initiate a claim.""",
    """Subject: Meeting Follow-up\nFrom: ceo@bigcorp.com\nDate: July 5, 2025\nThanks for attending. Next steps involve budget review and product roadmap."""
]

# ==== Output Model ====
class EmailInfo(BaseModel):
    sender: str = Field(description="The sender email address")
    subject: str = Field(description="Subject of the email")
    date: str = Field(description="Date the email was sent")
    key_topics: List[str] = Field(description="Important topics mentioned")

parser = PydanticOutputParser(pydantic_object=EmailInfo)
format_instructions = parser.get_format_instructions()

# ==== Prompt ====
template = """
Extract the following information from the email:

- sender
- subject
- date
- key_topics (as a list of keywords)

Return the output in the format below:
{format_instructions}

email: {email}
"""
prompt = ChatPromptTemplate.from_template(template)

# ==== Process Emails ====
parsed_emails = []
documents = []

for email in emails:
    messages = prompt.format_messages(email=email, format_instructions=format_instructions)
    output = llm(messages)
    parsed = parser.parse(output.content)
    parsed_emails.append(parsed)

    # Create Document to store in vector DB
    doc_text = f"Subject: {parsed.subject}\nSender: {parsed.sender}\nDate: {parsed.date}\nTopics: {', '.join(parsed.key_topics)}"
    documents.append(Document(page_content=doc_text, metadata={"email_date": parsed.date}))

# ==== Store in Vector DB (FAISS) ====
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# ==== Test Vector Search ====
query = "fire damage in house"
results = vectorstore.similarity_search(query, k=2)

print("\n🔍 Top Matches:")
for doc in results:
    print(doc.page_content)

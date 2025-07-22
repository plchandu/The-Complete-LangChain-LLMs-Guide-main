import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from typing import List

# Load API key
load_dotenv()
llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

# ==== Sample emails ====
emails = [
    """Subject: Travel Itinerary\nFrom: alice@example.com\nDate: July 1, 2025\nWe are flying to Paris and Rome this summer vacation.""",
    """Subject: Insurance Claim\nFrom: bob@insurance.com\nDate: July 3, 2025\nThere was a minor fire in the kitchen. We need to initiate a claim.""",
    """Subject: Meeting Follow-up\nFrom: ceo@bigcorp.com\nDate: July 5, 2025\nThanks for attending. Next steps involve budget review and product roadmap."""
]

# ==== Define output structure ====
class EmailInfo(BaseModel):
    sender: str = Field(description="The sender email address")
    subject: str = Field(description="Subject of the email")
    date: str = Field(description="Date the email was sent")
    key_topics: List[str] = Field(description="Important topics mentioned")

parser = PydanticOutputParser(pydantic_object=EmailInfo)
format_instructions = parser.get_format_instructions()

# ==== Prompt Template ====
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

# ==== Process all emails ====
results = []
for email in emails:
    messages = prompt.format_messages(email=email, format_instructions=format_instructions)
    output = llm(messages)
    parsed = parser.parse(output.content)
    results.append(parsed)

# ==== View Results ====
for i, r in enumerate(results):
    print(f"\n📩 Email {i+1}")
    print(f"Sender: {r.sender}")
    print(f"Subject: {r.subject}")
    print(f"Date: {r.date}")
    print(f"Topics: {r.key_topics}")

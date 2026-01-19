import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# ==========================================
# 1. Setup Environment and LLM
# ==========================================
# Load environment variables (like OPENAI_API_KEY)
load_dotenv()

# Initialize the Chat Model (GPT-3.5 Turbo)
# Temperature is set to 0.0 for deterministic outputs (consistent extraction)
llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

# ==========================================
# 2. Define Sample Data
# ==========================================
# A list of raw email texts to process. 
# In a real app, these could come from an API or database.
emails = [
    """Subject: Travel Itinerary\nFrom: alice@example.com\nDate: July 1, 2025\nWe are flying to Paris and Rome this summer vacation.""",
    """Subject: Insurance Claim\nFrom: bob@insurance.com\nDate: July 3, 2025\nThere was a minor fire in the kitchen. We need to initiate a claim.""",
    """Subject: Meeting Follow-up\nFrom: ceo@bigcorp.com\nDate: July 5, 2025\nThanks for attending. Next steps involve budget review and product roadmap."""
]

# ==========================================
# 3. Define Output Structure (Pydantic)
# ==========================================
# We define a Pydantic model to strictly enforce the shape of the data we want to extract.
# This ensures that the LLM's output can be parsed programmatically.
class EmailInfo(BaseModel):
    sender: str = Field(description="The sender email address")
    subject: str = Field(description="Subject of the email")
    date: str = Field(description="Date the email was sent")
    key_topics: List[str] = Field(description="Important topics mentioned")

# Create the Parser based on the Pydantic model
parser = PydanticOutputParser(pydantic_object=EmailInfo)

# Get instructions for the LLM on how to format its JSON response to match our model
format_instructions = parser.get_format_instructions()

# ==========================================
# 4. Create the Prompt Template
# ==========================================
# The template includes:
# - Instructions on what to extract
# - The 'format_instructions' injected from the parser (CRITICAL for valid JSON)
# - A placeholder object '{email}' for the actual input data
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

# ==========================================
# 5. Process Emails (Batch Processing)
# ==========================================
results = []
for email in emails:
    # A. Format the prompt with the specific email content
    messages = prompt.format_messages(email=email, format_instructions=format_instructions)
    
    # B. Send to LLM
    output = llm(messages)
    
    # C. Parse the text response into a Python object (EmailInfo)
    parsed = parser.parse(output.content)
    results.append(parsed)

# ==========================================
# 6. Display Results
# ==========================================
for i, r in enumerate(results):
    print(f"\n📩 Email {i+1}")
    print(f"Sender: {r.sender}")
    print(f"Subject: {r.subject}")
    print(f"Date: {r.date}")
    print(f"Topics: {r.key_topics}")

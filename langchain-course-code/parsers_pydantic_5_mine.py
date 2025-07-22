import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain_openai import ChatOpenAI  # ✅ Updated per deprecation notice
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator  # ✅ Use `pydantic` directly
from typing import List

# Load environment variables
load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# Setup ChatOpenAI
llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

# Email content
email_response = """
Here's our itinerary for our upcoming trip to Europe.
There will be 5 of us on this vacation trip.
We leave from Denver, Colorado airport at 8:45 pm, and arrive in Amsterdam 10 hours later
at Schipol Airport.
We'll grab a ride to our airbnb and maybe stop somewhere for breakfast before 
taking a nap.

Some sightseeing will follow for a couple of hours. 
We will then go shop for gifts 
to bring back to our children and friends.  

The next morning, at 7:45am we'll drive to to Belgium, Brussels - it should only take aroud 3 hours.
While in Brussels we want to explore the city to its fullest - no rock left unturned!
"""

# Pydantic model to structure the parsed output
class VacationInfo(BaseModel):
    leave_time: str = Field(description="Departure time to Europe")
    leave_from: str = Field(description="Departure location (city, airport, state)")
    cities_to_visit: List[str] = Field(description="List of cities to visit")
    num_people: int = Field(description="Number of people on the trip")

    @validator("num_people")
    def check_positive(cls, v):
        if v <= 0:
            raise ValueError("Number of people must be positive")
        return v

# Setup parser
parser = PydanticOutputParser(pydantic_object=VacationInfo)
format_instructions = parser.get_format_instructions()

# Prompt Template
email_prompt = ChatPromptTemplate.from_template("""
From the following email, extract the following information regarding this trip:

email: {email}

{format_instructions}
""")

# Prepare messages and get response
messages = email_prompt.format_messages(
    email=email_response,
    format_instructions=format_instructions
)

response = llm(messages)

# Parse the LLM output
vacation = parser.parse(response.content)

# Print results
print(f"Type: {type(vacation)}")
print(f"Leave Time: {vacation.leave_time}")
print(f"Leave From: {vacation.leave_from}")
print(f"Num People: {vacation.num_people}")
print("Cities to Visit:")
for city in vacation.cities_to_visit:
    print(f"  - {city}")

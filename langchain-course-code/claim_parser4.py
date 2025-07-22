import os
from dotenv import load_dotenv, find_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Load API key
load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# Chat model
chat = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

# Email input
claim_email = """
Hi,

I recently experienced some storm damage to my roof and would like to file a claim under my homeowner’s insurance policy.

The storm occurred on July 12th in Austin, Texas, and caused visible damage to the shingles and part of the attic. My policy number is H12345678TX.

Please let me know what documents I need to submit, and whether an adjuster will be visiting for inspection.

Thanks,
John Doe
"""

# Output schema
storm_date_schema = ResponseSchema(
    name="storm_date",
    description="The date when the storm occurred, in 'Month DD' or 'YYYY-MM-DD' format if possible"
)

location_schema = ResponseSchema(
    name="location",
    description="City and state where the storm occurred"
)

policy_number_schema = ResponseSchema(
    name="policy_number",
    description="The policy number for the homeowner's insurance claim"
)

response_schema = [storm_date_schema, location_schema, policy_number_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schema)
format_instructions = output_parser.get_format_instructions()

# Prompt with instructions
claim_template = """
From the following email, extract the following information:

storm_date: The date when the storm occurred.
location: The location (city and state) where the storm happened.
policy_number: The policy number mentioned in the email.

Return the result as JSON using the following keys:
storm_date
location
policy_number

email: {email}
{format_instructions}
"""

# Create and run prompt
claim_prompt = ChatPromptTemplate.from_template(claim_template)
messages = claim_prompt.format_messages(email=claim_email,
                                        format_instructions=format_instructions)
response = chat(messages)

# Parse and print result
output_dict = output_parser.parse(response.content)
print(output_dict)

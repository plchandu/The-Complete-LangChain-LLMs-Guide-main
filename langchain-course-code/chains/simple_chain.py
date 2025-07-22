import os
from dotenv import find_dotenv,load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Email example
email_text = """
Hi, I want to file a claim under my home insurance policy. 
The recent hailstorm on July 12th in Austin, Texas damaged the roof and attic. 
My policy number is H12345678TX. Let me know the next steps.
"""

# Define prompt template
template = """
Extract the following details from the insurance claim email:
- policy_number
- date_of_incident
- location
- damage_description

Format the output as csv

Email:
{email}
"""

prompt = PromptTemplate(
    input_variables=["email"],
    template=template
)

llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.0)

# LLMChain creation
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
response = chain.run(email=email_text)
print(response)
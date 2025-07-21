import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI  # ✅ correct import for new SDK

# Load environment variables
load_dotenv(find_dotenv())

# Get your key
api_key = os.getenv("OPENAI_API_KEY")

print(api_key)

# Debug check (optional)
print("API Key Loaded:", bool(api_key))  # Should print True

#Instantiate OpenAI client with the keyclear

client = OpenAI(api_key=api_key)

# Function to run ChatCompletion
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    #return response.choices[0].message.content
    return response.choices[0].message.content

# Sample input
customer_review = """
Your product is terrible! I don't know how 
you were able to get this to the market.
I don't want this! Actually no one should want this.
Seriously! Give me money now!
"""

tone = "Proper British English in a nice, warm, respectful tone"
language = "Turkish"

prompt = f"""
Please rewrite the following review in {tone}, and then
translate the revised review into {language}.

Review:
'''{customer_review}'''
"""

# Run and print
print(get_completion(prompt))

import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# Load OpenAI API key
load_dotenv()
llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")

# === Step 1: Summarize Review ===
summary_prompt = PromptTemplate(
    input_variables=["review"],
    template="""
Summarize the following product review in 1-2 sentences:

Review: {review}
Summary:"""
)

summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")

# === Step 2: Extract Key Points ===
keypoints_prompt = PromptTemplate(
    input_variables=["review"],
    template="""
Extract the key points from this product review as bullet points:

Review: {review}
Key Points:"""
)

keypoints_chain = LLMChain(llm=llm, prompt=keypoints_prompt, output_key="key_points")

# === Step 3: Detect Categories ===
categories_prompt = PromptTemplate(
    input_variables=["review"],
    template="""
Identify relevant product features/categories mentioned in the review (e.g., battery life, screen quality, customer service, price, etc.):

Review: {review}
Categories:"""
)

categories_chain = LLMChain(llm=llm, prompt=categories_prompt, output_key="categories")

# === Step 4: Generate Customer Response ===
response_prompt = PromptTemplate(
    input_variables=["summary", "key_points"],
    template="""
Write a helpful and polite response to the customer based on the following summary and key points:

Summary: {summary}
Key Points: {key_points}

Response:"""
)

response_chain = LLMChain(llm=llm, prompt=response_prompt, output_key="response")

# === Combine with SequentialChain ===
review_chain = SequentialChain(
    chains=[summary_chain, keypoints_chain, categories_chain, response_chain],
    input_variables=["review"],
    output_variables=["summary", "key_points", "categories", "response"],
    verbose=True
)

# === Example Review ===
review_text = """
I recently bought this phone and the battery lasts only half a day. 
The screen is bright and vibrant, but the customer service was terrible when I asked for help.
Also, the price feels too high for the features it offers.
"""

# === Run the Chain ===
result = review_chain.invoke({"review": review_text})

# === Print Results ===
print("\n📋 Summary:\n", result["summary"])
print("\n🔑 Key Points:\n", result["key_points"])
print("\n📂 Categories:\n", result["categories"])
print("\n✉️ Response:\n", result["response"])

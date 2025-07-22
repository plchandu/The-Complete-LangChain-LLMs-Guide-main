import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

# Load API key
load_dotenv()
llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

# === Input Email ===
email_text = """
Subject: Insurance Claim Request
From: john.doe@example.com
Date: July 10, 2025

Hello Team,

There was a flood in our basement due to a pipe burst. The carpets are soaked and some electronics got damaged.
Please guide us on how to initiate a claim and what documents we need.

Regards,
John
"""

# === Step 1: Summarize the email ===
summary_prompt = PromptTemplate(
    input_variables=["email"],
    template="Summarize the following email in 1-2 sentences:\n\n{email}"
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")

# === Step 2: Detect Claim Type ===
claim_type_prompt = PromptTemplate(
    input_variables=["summary"],
    template="Based on the summary, what type of insurance claim is this (e.g. fire, flood, theft, auto)?\n\nSummary: {summary}"
)
claim_type_chain = LLMChain(llm=llm, prompt=claim_type_prompt, output_key="claim_type")

# === Step 3: Generate Final Response ===
response_prompt = PromptTemplate(
    input_variables=["summary", "claim_type"],
    template="""You are an insurance agent. Write a professional response based on the following details:

Summary: {summary}
Claim Type: {claim_type}

Your response:"""
)
response_chain = LLMChain(llm=llm, prompt=response_prompt, output_key="final_response")

# === Sequential Chain ===
chain = SequentialChain(
    chains=[summary_chain, claim_type_chain, response_chain],
    input_variables=["email"],
    output_variables=["summary", "claim_type", "final_response"],
    verbose=True,
)

# === Run the chain ===
result = chain.invoke({"email": email_text})

# === Output ===
print("\n=== Output ===")
print(f"Summary: {result['summary']}")
print(f"Claim Type: {result['claim_type']}")
print(f"Response:\n{result['final_response']}")

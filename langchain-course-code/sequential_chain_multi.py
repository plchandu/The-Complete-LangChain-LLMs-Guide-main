from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# Step 1: Setup LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Step 2: Prompt to generate product description
desc_prompt = PromptTemplate(
    input_variables=["product", "audience"],
    template="Write a detailed product description for a {product} targeting {audience}."
)

desc_chain = LLMChain(llm=llm, prompt=desc_prompt, output_key="description")

# Step 3: Prompt to create a catchy tagline from the description
tagline_prompt = PromptTemplate(
    input_variables=["description"],
    template="Create a catchy marketing tagline based on the following product description:\n{description}"
)

tagline_chain = LLMChain(llm=llm, prompt=tagline_prompt, output_key="tagline")

# Step 4: SequentialChain - Multiple input/output
multi_chain = SequentialChain(
    chains=[desc_chain, tagline_chain],
    input_variables=["product", "audience"],
    output_variables=["description", "tagline"],
    verbose=True
)

# Step 5: Run the chain
result = multi_chain.run({
    "product": "smart water bottle",
    "audience": "fitness enthusiasts"
})

# Step 6: Print results
print("📝 Product Description:\n", result["description"])
print("\n💡 Tagline:\n", result["tagline"])

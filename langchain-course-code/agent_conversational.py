import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.llms import OpenAI
from langchain.agents import Tool, initialize_agent, load_tools
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Load environment variables (API keys, etc.) from a .env file
load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# ==========================================
# 1. Initialize the Large Language Model (LLM)
# ==========================================
# We use the OpenAI class (defaulting to text-davinci-003 or similar legacy models if not specified).
# 'temperature=0.0' ensures deterministic output (less creative, more precise).
llm = OpenAI(temperature=0.0)

# ==========================================
# 2. Set up Memory
# ==========================================
# ConversationBufferMemory stores the history of the conversation.
# 'memory_key="chat_history"' tells the agent where to inject the history into the prompt.
memory = ConversationBufferMemory(memory_key="chat_history")

# ==========================================
# 3. Create a Custom "General Purpose" Tool
# ==========================================
# We create a simple chain that takes a query and sends it directly to the LLM.
# This serves as a fallback or general reasoning tool.

prompt = PromptTemplate(
    input_variables=["query"],
    template="{query}"
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

# Initialize the Tool object
llm_tool = Tool(
    name="Language Model",
    func=llm_chain.run,
    description="Use this tool for general queries and logic" 
)

# ==========================================
# 4. Load Pre-built Tools
# ==========================================
# 'llm-math' is a tool that uses the LLM to generate Python code for math and executes it.
tools = load_tools(
    ['llm-math'],
    llm=llm
)

# Add our custom 'Language Model' tool to the list of available tools
tools.append(llm_tool) 

# ==========================================
# 5. Initialize the Conversational Agent
# ==========================================
# 'conversational-react-description' is an agent type that:
# - Holds a conversation (uses memory)
# - Uses the ReAct framework (Reasoning + Acting) to decide which tool to use
# - Decides based on the 'description' field of each tool
conversational_agent = initialize_agent(
    agent="conversational-react-description",
    tools=tools,
    llm=llm,
    verbose=True,       # Prints the agent's thought process (Observation/Action/Thought)
    max_iterations=3,   # Limits the agent to 3 steps to prevent infinite loops
    memory=memory
)

# ==========================================
# 6. Run the Agent
# ==========================================

# Query 1: Requires calculation (Mathematics)
query = "How old is a person born in 1917 in 2023"
    
# Query 2: Requires memory context ("that person") + calculation
query_two = "How old would that person be if their age is multiplied by 100?"
    
# Print the hidden prompt template used by the agent (for debugging/learning)
print(conversational_agent.agent.llm_chain.prompt.template)

print("\n--- Running Query 1 ---")
result = conversational_agent(query)

print("\n--- Running Query 2 ---")
results = conversational_agent(query_two)

# print(result['output'])

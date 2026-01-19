import os 
import openai
from dotenv import find_dotenv, load_dotenv
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# ==========================================
# 1. Setup Environment
# ==========================================
# Load environment variables (API keys) from .env file
load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# ==========================================
# 2. Define Model Configurations
# ==========================================
llm_model = "gpt-3.5-turbo"

# ==========================================
# 3. Create Messages
# ==========================================
# For Chat Models, inputs are usually a list of messages (HumanMessage, SystemMessage, AIMessage)
prompt = "How old is the Universe"
messages = [HumanMessage(content=prompt)]

# ==========================================
# 4. Initialize Models
# ==========================================
# 'OpenAI' is for text completion models (Davinci, etc.) - Legacy
llm = OpenAI(temperature=0.7)

# 'ChatOpenAI' is for chat models (GPT-3.5, GPT-4)
# temperature=0.7 means more creative/variable responses (0.0 is deterministic)
chat_model = ChatOpenAI(temperature=0.7)

# ==========================================
# 5. Run Predictions
# ==========================================
# LLM Usage (Completion):
# print(llm.predict("What is the weather in WA DC"))
# print("==========")

# Chat Model Usage:
# predict_messages takes a list of Message objects
print(chat_model.predict_messages(messages).content)

# predict is a helper method that can also take a string directly
# print(chat_model.predict("What is the weather in WA DC"))





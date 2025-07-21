import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
#from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")

#==== Using OpenAI Chat API =======
llm_model = "gpt-3.5-turbo"

chat = ChatOpenAI(temperature=0.9, model=llm_model)
# open_ai = OAI( 
#     temperature=0.7,
#     model_name="text-davinci-003",  # ✅ This is for completion models, not chat
#     openai_api_key=api_key 
# )


# LLMChain
prompt = PromptTemplate(
    input_variables=["language"],
    template="How do you say good morning in {language}"
)

chain = LLMChain(llm=chat, prompt=prompt)
print(chain.run(language="German"))
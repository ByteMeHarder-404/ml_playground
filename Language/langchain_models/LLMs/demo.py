from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

lang=OpenAI(model='gpt-3.5-turbo-instruct')
res=lang.invoke('What is capital of America?')
print(res)
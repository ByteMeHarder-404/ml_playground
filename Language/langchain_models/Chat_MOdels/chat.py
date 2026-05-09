from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

mode=ChatGoogleGenerativeAI(model='gemini-3-flash-preview')
res=mode.invoke('What is capital of Estonia')
print(res.text)
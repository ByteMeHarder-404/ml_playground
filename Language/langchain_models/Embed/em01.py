from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
emd=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

doc=['Computer','Steve Jobs','Steve Wozniak']
vec=emd.embed_documents(doc)
print(str(vec))
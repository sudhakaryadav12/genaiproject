from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

import os

# 🔐 Set your API key
os.environ["OPENAI_API_KEY"] = "your_api_key_here"

# 1️⃣ Load document
loader = TextLoader("data.txt")  # your text file
documents = loader.load()

# 2️⃣ Split into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# 3️⃣ Create embeddings
embeddings = OpenAIEmbeddings()

# 4️⃣ Store in FAISS vector DB
vectorstore = FAISS.from_documents(docs, embeddings)

# 5️⃣ Create retriever
retriever = vectorstore.as_retriever()

# 6️⃣ Create LLM
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0
)

# 7️⃣ Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 8️⃣ Ask question
query = "What is the document about?"
response = qa_chain(query)

# 9️⃣ Print answer
print("Answer:")
print(response["result"])

print("\nSources:")
for doc in response["source_documents"]:
    print(doc.page_content[:200])

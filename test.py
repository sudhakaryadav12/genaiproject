import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = "your-api-key"

# 1. Load documents (create a sample file first)
loader = TextLoader("sample.txt")
documents = loader.load()

# 2. Split documents into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# 3. Create embeddings
embeddings = OpenAIEmbeddings()

# 4. Store in vector DB (FAISS)
vectorstore = FAISS.from_documents(docs, embeddings)

# 5. Create retriever
retriever = vectorstore.as_retriever()

# 6. Create RAG QA chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever
)

# 7. Ask a question
query = "What is this document about?"
result = qa.run(query)

print("Answer:", result)

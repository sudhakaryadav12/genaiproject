from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchFieldDataType, SearchableField, VectorSearch,
    HnswAlgorithmConfiguration, VectorSearchProfile
)
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

# ========================
# CONFIG
# ========================
SEARCH_ENDPOINT = "https://<your-search>.search.windows.net"
SEARCH_KEY = "<your-search-key>"
INDEX_NAME = "sample-index"

AZURE_OPENAI_ENDPOINT = "https://<your-openai>.openai.azure.com/"
AZURE_OPENAI_KEY = "<your-openai-key>"
EMBEDDING_MODEL = "text-embedding-3-large"

# ========================
# OPENAI CLIENT
# ========================
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# ========================
# CREATE INDEX
# ========================
index_client = SearchIndexClient(
    endpoint=SEARCH_ENDPOINT,
    credential=AzureKeyCredential(SEARCH_KEY)
)

fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="content", type=SearchFieldDataType.String),
    SimpleField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single))
]

vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(name="my-hnsw")
    ],
    profiles=[
        VectorSearchProfile(
            name="my-vector-profile",
            algorithm_configuration_name="my-hnsw"
        )
    ]
)

index = SearchIndex(
    name=INDEX_NAME,
    fields=fields,
    vector_search=vector_search
)

index_client.create_or_update_index(index)

print("✅ Index created")

# ========================
# GENERATE EMBEDDINGS
# ========================
def get_embedding(text):
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

# ========================
# UPLOAD DOCUMENTS
# ========================
search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_KEY)
)

docs = [
    {
        "id": "1",
        "content": "Azure OpenAI is powerful for AI applications."
    },
    {
        "id": "2",
        "content": "Vector search enables semantic search."
    }
]

# Add embeddings
for doc in docs:
    doc["contentVector"] = get_embedding(doc["content"])

# Upload
result = search_client.upload_documents(documents=docs)

print("✅ Documents indexed:", result)

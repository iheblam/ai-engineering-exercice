import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

load_dotenv()

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "invoices"

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
)

prompt = ChatPromptTemplate.from_template("""
Answer the question using only the invoice context.
If the context does not contain the answer, say I don't know.
Mention the source invoice filename in your answer.
The context contains a sample of matching chunks, not all invoices.
Do not claim to count or sum all invoices in the database.

Question:
{question}

Context:
{context}
""")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
)


def format_docs(documents):
    return "\n\n".join(
        f"Metadata: {doc.metadata}\n{doc.page_content}"
        for doc in documents
    )


def create_filter(field, value):
    text_fields = ["code_facture", "date", "bill_to", "ship_to"]
    number_fields = ["total", "shipping", "discount", "amount"]

    if field not in text_fields + number_fields:
        raise ValueError("Unknown filter field.")

    condition = FieldCondition(
        key=f"metadata.{field}", match=MatchValue(value=value)
    )
    if field in number_fields:
        number = float(value.replace("$", "").replace(",", ""))
        condition = FieldCondition(
            key=f"metadata.{field}", range=Range(gte=number, lte=number)
        )

    return Filter(must=[condition])



field = input(
    "Filter field (code_facture, date, bill_to, ship_to, total, shipping, discount, amount) "
    "or Enter for no filter: "
).strip().lower().replace(" ", "_")

search_kwargs = {"k": 5}
if field:
    value = input("Exact filter value (dates: YYYY-MM-DD): ").strip()
    try:
        search_kwargs["filter"] = create_filter(field, value)
    except ValueError:
        raise SystemExit("Check the field name and use a number for money fields.")

retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt | llm | StrOutputParser()
)

while True:
    question = input("Ask something (/quit to exit): ")
    if question.lower() in ["/quit", "/bye"]:
        break
    print("AI:", rag_chain.invoke(question))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

file_paths = ["attention_all_u_need.txt"]
documents = []
for path in file_paths:
    loader = TextLoader(path,encoding='utf-8')
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""],
)

chunks = text_splitter.split_documents(documents)
print(len(chunks))


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={'normalize_embeddings': False}
)

texts = [chunk.page_content for chunk in chunks]

embeddings = embedding_model.embed_documents(texts)

vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding_model)

retriever = vectorstore.as_retriever()
llm = OllamaLLM(model="mistral")
rag_pipeline = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

prompt=input("Enter your question: ")

response = rag_pipeline.invoke(prompt)
print("Result : ",response['result'])
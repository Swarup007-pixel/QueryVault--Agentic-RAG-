import pypdf
import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import chain
from langchain.tools import tool

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''load_Document''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def load_document(file_path:str) -> list[Document]:
    reader = pypdf.PdfReader(file_path)
    return[        
        Document(
                page_content = page.extract_text() or "",
                metadata = {"source":file_path, "page": i},
        )for i, page in enumerate(reader.pages)       
    ]


#file_path = input("enter the path of the file you want to explore")
#docs = load_document(file_path)

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''text_split'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def text_spliter(document : list[Document], chunk_size = 1000, chunk_overlap = 200, add_start_index = True) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size =chunk_size  , chunk_overlap = chunk_overlap , add_start_index = add_start_index
        )
    return text_splitter.split_documents(document)



# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''sentence-transformer (embedding)'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''and vector storage'''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def Embedding(model_name : str = "sentence-transformers/all-MiniLM-L6-v2"):
     embedding = HuggingFaceEmbeddings(
                model_name = model_name,
                encode_kwargs={"normalize_embeddings": True}
            )
     return embedding
#embedding = Embedding()

def vector_store(chunk: list[Document],embedding_name , storage_name :str = "./chroma_db"):
    return Chroma.from_documents(
        collection_name="Query_Vault",
        documents=chunk,
        embedding = embedding_name,
        persist_directory= storage_name
    )
#vector = vector_store(chunk, embedding)
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''retriever'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
@chain
def retriever(query:str)->List[Document]:
    embeded_query = embedding.embed_query(query)
    vector_search = vector.similarity_search_by_vector(embeded_query)
    return vector_search
#query = "the questiom you wanna ask"
#answer = retriever.invoke(query)

#''''''''''''''''''''''''''''''''''''''''''''''''''''''''LLM/Agent integration'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''retriever tool'''''''''''''''''''''''''''''''''''''''''''''''''''''''
@tool
def retrieve_query(query:str)->List[Document]:
    output_of_retriever = retriever.invoke(query)
    return "\n".join([docs.page_content for docs in output_of_retriever])

#''''''''''''''''''''''''''''''''''''''''''''''''''''''''agent workflow'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

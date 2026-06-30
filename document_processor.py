import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

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
def Embedding(document: list[Document] , model_name : str = "sentence-transformers/all-MiniLM-L6-v2"):
    embedding = HuggingFaceEmbeddings(
        model_name = model_name,
        encode_kwargs={"normalize_embeddings": True}
    )
    texts = [doc.page_content for doc in document]
    return embedding.embed_documents(texts)

#embedding_of_docs = Embedding(chunk)

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

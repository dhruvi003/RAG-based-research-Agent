from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from pathlib import Path 
from typing import List


def load_documents(file_path: str)-> list[Document]:
    
    """
    Load documents from PDF or text file.
    Returns list of Document objects
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f'File not found:{file_path}')
    
    if path.suffix == '.pdf':
        loader = PyPDFLoader(str(path))
    elif path.suffix in ['.txt','.md']:
        loader = TextLoader(str(path))
    else:
        raise ValueError(f'Unsupported file type: {path.suffix}')
    
    documents = loader.load()
    print(f'Loaded {len(documents)} from {path.name}')
    return documents

#res = load_documents("data\W17-4419.pdf")

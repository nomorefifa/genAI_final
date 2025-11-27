import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Text splitting
# -----------------------------

def build_text_splitter(chunk_size: int = 700, chunk_overlap: int = 120):
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", ". ", "! ", "? ", "\n", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any]

def chunk_document(doc_text: str, source: str, splitter=None):
    """문서를 청크 단위로 나누기"""
    if splitter is None:
        splitter = build_text_splitter()

    pieces = splitter.split_text(doc_text)
    chunks = []
    for i, piece in enumerate(pieces):
        meta = {
            "source": source,
            "chunk_id": i,
        }
        chunks.append(
            Chunk(
                id=f"{os.path.basename(source)}::chunk_{i}",
                text=piece,
                metadata=meta,
            )
        )
    return chunks

# -----------------------------
# OpenAI Embedding / Chat helper
# -----------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def chat_with_openai(messages: List[Dict[str, str]]) -> str:
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    return resp.choices[0].message.content
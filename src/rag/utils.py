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

def build_prompt(query: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """RAG를 위한 프롬프트 생성"""
    sys = (
        "당신은 근거 기반으로 답하는 조교입니다.\n"
        "- 제공된 컨텍스트를 우선적으로 사용하세요.\n"
        "- 확실한 근거가 없으면 '모른다'고 말하세요.\n"
    )
    ctx_lines = []
    for idx, c in enumerate(contexts, start=1):
        src = os.path.basename(str(c.get("source", "unknown")))
        cid = c.get("chunk_id", "?")
        snippet = c.get("text", "").strip()
        ctx_lines.append(f"[{idx}] SOURCE={src} | CHUNK={cid}\n{snippet}")
    ctx_block = "\n\n".join(ctx_lines)
    user = (
        f"질문: {query}\n\n"
        f"다음은 검색으로 수집한 컨텍스트입니다. 필요한 부분만 사용하세요.\n\n"
        f"{ctx_block}\n\n"
        f"요구사항:\n"
        f"- 근거가 불충분하면 '모른다'고 답변\n"
        f"- 핵심만 한국어로 간결히\n"
        f"- 마지막에 '출처' 섹션을 반드시 포함하되, 아래 포맷을 그대로 사용:\n"
        f"  출처:\n"
        f"  - SOURCE:CHUNK 형태로 나열 (예: hnsw.pdf:12, hnsw.pdf:13)\n"
        f"  - 대괄호 숫자 인용([1], [2] 등) 사용 금지"
    )

    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]
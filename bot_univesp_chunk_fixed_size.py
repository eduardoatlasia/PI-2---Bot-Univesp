#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bot UNIVESP - RAG do Manual do Aluno usando:
- Crawler (requests + BeautifulSoup) com correção de encoding
- Chunking básico
- Embeddings (sentence-transformers: paraphrase-multilingual-MiniLM-L12-v2)
- FAISS (Inner Product com vetores normalizados)
- Gemma 3 270M via Ollama (com fallback /api/chat -> /api/generate)

Uso:
  # CLI
  python bot_univesp.py

  # API
  python bot_univesp.py --api
  curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
    -d '{"query":"Como funciona o fórum de dúvidas?"}'

Opções:
  --reindex   força reindexar (ignora arquivos salvos)
Env:
  OLLAMA_HOST  (default: http://localhost:11434)
  OLLAMA_MODEL (default: gemma3:270m)
  RAG_REINDEX  (default: 0; se 1, reindexa no startup da API)
"""

import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple

from contextlib import asynccontextmanager

import requests
from bs4 import BeautifulSoup

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# -----------------------
# Configs
# -----------------------
SOURCE_URL = "https://apps.univesp.br/manual-do-aluno/"
MODEL_EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b"  )

CHUNK_SIZE =    3000
CHUNK_OVERLAP = 150
TOP_K = 10
SYSTEM_PROMPT = (
    "1. Você é um assistente que responde com base no Manual do Aluno da UNIVESP. Fizemos um RAG fornecendo trechos."
    "2. Se não houver informação nos trechos fornecidos, diga claramente que não encontrou no material. "
    "3. Responda em português claro e objetivo."
    "4. Revise a reposta e garanta que não há redundância."
    "5. Realize inferências simples se necessário. Por exemplo: se há cursos tecnólogos, licenciaturas e bacharelados, e não há menção a licenciatura ou tecnólogo, você pode inferir que o curso é bacharelado e suas regras se aplicam a esse curso."
)

# Persistência
INDEX_PATH = "faiss_univesp.index"
CHUNKS_PATH = "chunks_univesp.json"
EMB_PATH = "embeddings_univesp.npy"
RAW_TXT_PATH = "manual_univesp.txt"

# -----------------------
# Crawler + limpeza
# -----------------------
def fetch_utf8(url: str) -> str:
    """Baixa HTML e tenta garantir UTF-8 (com fallback latin1->utf8 contra mojibake)."""
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    # tenta usar o encoding informado; se None, força utf-8
    r.encoding = r.encoding or "utf-8"
    html = r.text
    # fallback simples: se detectar mojibake típico, tenta reparar
    if any(p in html for p in ("Ã©", "Ã§", "Ã£", "Ãº", "Ã¡", "Ãª", "Ã³", "Ã­")):
        try:
            html = html.encode("latin1").decode("utf-8")
        except Exception:
            pass
    return html

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup(["nav", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # limpeza
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text

# -----------------------
# Chunking
# -----------------------
def chunk_text(s: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    """Chunk por tamanho de caractere com tentativa de quebra em fim de frase."""
    chunks = []
    n = len(s)
    start = 0
    while start < n:
        end = min(start + size, n)
        chunk = s[start:end]
        if end < n:
            last_dot = chunk.rfind(".")
            if last_dot > int(size * 0.6):
                chunk = chunk[:last_dot + 1]
                end = start + len(chunk)
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, end)
    return chunks

# -----------------------
# Snippets e destaque
# -----------------------
def _keywords(query: str) -> List[str]:
    return [w.lower() for w in re.findall(r"\w+", query, flags=re.UNICODE) if len(w) >= 4]

def make_snippet(text: str, query: str, max_chars: int = 360) -> str:
    kws = _keywords(query)
    if not kws:
        return (text[:max_chars].rsplit(" ", 1)[0] + "…") if len(text) > max_chars else text
    lower = text.lower()
    positions = [lower.find(k) for k in kws if lower.find(k) != -1]
    hit_pos = min(positions) if positions else -1
    if hit_pos == -1:
        return (text[:max_chars].rsplit(" ", 1)[0] + "…") if len(text) > max_chars else text
    half = max_chars // 2
    start = max(hit_pos - half, 0)
    end = min(start + max_chars, len(text))
    start = max(end - max_chars, 0)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "… " + snippet
    if end < len(text):
        snippet = snippet + " …"
    return snippet

def highlight_terms(snippet: str, query: str) -> str:
    kws = sorted(set(_keywords(query)), key=len, reverse=True)
    s = snippet
    for kw in kws:
        s = re.sub(rf"(?i)\b({re.escape(kw)})\b", r"**\1**", s)
    return s

# -----------------------
# Estruturas do RAG
# -----------------------
@dataclass
class RagStore:
    model: SentenceTransformer
    index: faiss.IndexFlatIP
    embeddings: np.ndarray
    chunks: List[str]

def _save_artifacts(text: str, chunks: List[str], embs: np.ndarray, index: faiss.IndexFlatIP):
    with open(RAW_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    np.save(EMB_PATH, embs)
    faiss.write_index(index, INDEX_PATH)

def _load_artifacts():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH) and os.path.exists(EMB_PATH)):
        return None, None, None
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embs = np.load(EMB_PATH)
    index = faiss.read_index(INDEX_PATH)
    return chunks, embs, index

def build_store(reindex: bool = False) -> RagStore:
    model = SentenceTransformer(MODEL_EMB)

    if not reindex:
        loaded = _load_artifacts()
        if loaded[0] is not None:
            chunks, embs, index = loaded
            if embs.ndim == 2:
                return RagStore(model, index, embs.astype("float32"), chunks)

    # Reindexa do zero
    html = fetch_utf8(SOURCE_URL)
    text = html_to_text(html)
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    embs = model.encode(chunks, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype="float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    _save_artifacts(text, chunks, embs, index)
    return RagStore(model, index, embs, chunks)

def retrieve(store: RagStore, query: str, k=TOP_K) -> List[Tuple[int, float]]:
    q = store.model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")
    scores, idxs = store.index.search(q, k)
    return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]

def build_prompt(query: str, contexts: List[str]) -> List[dict]:
    context_block = "\n\n---\n\n".join(contexts)
    user_msg = (
        f"Pergunta: {query}\n\n"

        f"=== TRECHOS ===\n{context_block}\n\n"
        f"=== FIM DOS TRECHOS ===\n\n"
        f"Resposta objetiva:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

# -----------------------
# Ollama com fallback
# -----------------------
def call_ollama(messages: list[dict], model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST, stream: bool = False) -> str:
    """
    Tenta /api/chat (se disponível). Se 404/405, faz fallback para /api/generate
    convertendo o histórico de mensagens em um único prompt.
    """
    def _chat() -> str:
        url = f"{host}/api/chat"
        payload = {"model": model, "messages": messages, "stream": stream}
        r = requests.post(url, json=payload, timeout=120)
        if r.status_code in (404, 405):  # endpoint indisponível nesta versão
            raise RuntimeError("OLLAMA_NO_CHAT_ENDPOINT")
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "")

    def _generate_fallback() -> str:
        parts = []
        for m in messages:
            role = m.get("role", "user")
            if role == "system":
                parts.append(f"[system]\n{m['content']}\n[/system]")
            elif role == "assistant":
                parts.append(f"Assistant:\n{m['content']}")
            else:  # user
                parts.append(f"User:\n{m['content']}")
        prompt = "\n\n".join(parts)

        url = f"{host}/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False}
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")

    try:
        return _chat()
    except Exception as e:
        # Se não há /api/chat, ou deu 404/405, usamos generate
        if isinstance(e, RuntimeError) and str(e) == "OLLAMA_NO_CHAT_ENDPOINT":
            return _generate_fallback()
        if hasattr(e, "response") and getattr(e.response, "status_code", None) in (404, 405):
            return _generate_fallback()
        # Outros erros: propaga
        raise

# -----------------------
# FastAPI com lifespan (sem DeprecationWarning)
# -----------------------
STORE: RagStore | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global STORE
    reindex_flag = os.getenv("RAG_REINDEX", "0") == "1"
    STORE = build_store(reindex=reindex_flag)
    yield
    # (não há recursos para fechar)

app = FastAPI(title="RAG UNIVESP (Manual do Aluno) + Gemma3 1B", lifespan=lifespan)

@app.get("/healthz")
def healthz():
    return {"status": "ok", "chunks": len(STORE.chunks) if STORE else 0}

@app.post("/chat")
def chat(payload: dict):
    query = (payload.get("query") or "").strip()
    if not query:
        return JSONResponse({"error": "query vazia"}, status_code=400)

    results = retrieve(STORE, query, k=TOP_K)
    contexts = [STORE.chunks[i] for i, _ in results]
    messages = build_prompt(query, contexts)
    answer = call_ollama(messages)

    sources = []
    for i, s in results:
        chunk_text_val = STORE.chunks[i]
        snippet = make_snippet(chunk_text_val, query, max_chars=360)
        sources.append({
            "chunk_id": i,
            "score": round(s, 6),
            "text": chunk_text_val,                         # chunk completo
            "snippet": snippet,                             # recorte curto
            "snippet_marked": highlight_terms(snippet, query)  # recorte com destaque
        })

    return {"answer": answer, "sources": sources, "k": TOP_K}

# -----------------------
# CLI simples
# -----------------------
def cli(force_reindex: bool = False):
    global STORE
    if STORE is None:
        STORE = build_store(reindex=force_reindex)

    print("RAG UNIVESP (Manual do Aluno) - Gemma3 270M")
    print("Digite sua pergunta (ou 'sair'):")

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaindo.")
            break
        if not q or q.lower() in {"sair", "exit", "quit"}:
            break

        try:
            hits = retrieve(STORE, q, k=TOP_K)
            ctxs = [STORE.chunks[i] for i, _ in hits]
            msgs = build_prompt(q, ctxs)
            resp = call_ollama(msgs)
        except requests.exceptions.ConnectionError:
            print("\n[ERRO] Não consegui conectar ao Ollama em", OLLAMA_HOST)
            print("Verifique se o serviço está rodando e se o modelo foi baixado:")
            print("  $ ollama serve   (se necessário)")
            print("  $ ollama pull gemma3:270m\n")
            continue
        except Exception as e:
            print("\n[ERRO] Falha ao chamar o modelo:", repr(e), "\n")
            continue

        print("\nRESPOSTA:\n", resp, "\n")
        #print("Fontes:")
        #for i, s in hits:
        #    full = STORE.chunks[i]
        #    snip = make_snippet(full, q, max_chars=360)
        #    print(f"- #{i} (score={s:.3f}):\n  {highlight_terms(snip, q)}\n")

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true", help="sobe a API FastAPI em http://0.0.0.0:8000")
    parser.add_argument("--reindex", action="store_true", help="força reindexação do conteúdo")
    parser.add_argument("--host", default="0.0.0.0", help="host da API (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="porta da API (default: 8000)")
    args = parser.parse_args()

    if args.api:
        # Lifespan já cuida do build; a flag RAG_REINDEX=1 força reindex
        if args.reindex:
            os.environ["RAG_REINDEX"] = "1"
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        cli(force_reindex=args.reindex)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG UNIVESP — Crawler HTML+PDF + LLM-assisted Chunking (com fallbacks)
- Crawler BFS: segue links internos (mesmo domínio; opcionalmente mesmo path)
- Baixa somente text/html e application/pdf (ou URLs .pdf)
- Extração de blocos:
    * HTML: parágrafos/listas/tabelas com hints de headings
    * PDF: parágrafos por quebras duplas de linha
- Segmentação assistida por LLM (JSON) com validação por embeddings
- Fallback: chunk semântico (embeddings) -> split por tamanho
- Index: FAISS (Inner Product + vetores normalizados)
- Geração: via Ollama (chat -> generate)

Env principais:
  SOURCE_URL=https://apps.univesp.br/manual-do-aluno/
  CRAWL_MAX_PAGES=40
  CRAWL_MAX_DEPTH=2
  CRAWL_SAME_PATH=1
  USE_LLM_CHUNK=1
"""

import os
import re
import json
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from collections import deque
from urllib.parse import urlparse, urljoin, urldefrag

from contextlib import asynccontextmanager

import requests
from bs4 import BeautifulSoup, Tag

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# -----------------------
# Configs
# -----------------------
SOURCE_URL = os.getenv("SOURCE_URL", "https://apps.univesp.br/manual-do-aluno/")
MODEL_EMB = os.getenv("MODEL_EMB", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")

# Crawler
CRAWL_MAX_PAGES = int(os.getenv("CRAWL_MAX_PAGES", "40"))
CRAWL_MAX_DEPTH = int(os.getenv("CRAWL_MAX_DEPTH", "2"))
CRAWL_SAME_PATH = os.getenv("CRAWL_SAME_PATH", "1") == "1"
CRAWL_DELAY_SEC = float(os.getenv("CRAWL_DELAY_SEC", "0.2"))  # polidez leve
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "45"))

# LLM-assisted chunking (liga/desliga)
USE_LLM_CHUNK = os.getenv("USE_LLM_CHUNK", "1") == "1"
LLM_MAX_BLOCKS = int(os.getenv("LLM_MAX_BLOCKS", "80"))  # blocos por janela
LLM_TARGET_MAX_CHARS = int(os.getenv("LLM_TARGET_MAX_CHARS", "1200"))

# Parâmetros de chunk semântico (fallback 1)
SEM_MIN_CHARS = 400
SEM_MAX_CHARS = 1200
SEM_OVERLAP_CHARS = 120
SEM_BREAK_ABS = 0.62
SEM_BREAK_DROP = 0.15

# Split por tamanho (fallback 2)
CHUNK_SIZE_FALLBACK = 1000
CHUNK_OVERLAP_FALLBACK = 120

TOP_K = 15
SYSTEM_PROMPT = (
    "1. Você é um assistente que responde com base no Manual do Aluno da UNIVESP e documentos vinculados. "
    "2. Responda em português claro e objetivo. "
    "3. Revise a reposta e garanta que não há redundância. "
    "4. Se algo não aparecer no material, responda negativamente e justifique que não há menção no material."
)

# Persistência
INDEX_PATH = "faiss_univesp.index"
CHUNKS_PATH = "chunks_univesp.json"      # lista de dicts com metadados
EMB_PATH = "embeddings_univesp.npy"
RAW_TXT_PATH = "manual_univesp.txt"

# -----------------------
# Utilidades de texto
# -----------------------
def _clean_text(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()

# -----------------------
# Extração PDF
# -----------------------
def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    """
    Tenta extrair texto de PDF:
      1) pdfminer.six (melhor)
      2) PyPDF2 (fallback simples)
      3) retorna "" se nada disponível
    """
    # Try pdfminer.six
    try:
        from pdfminer.high_level import extract_text
        import io
        return _clean_text(extract_text(io.BytesIO(pdf_bytes)) or "")
    except Exception:
        pass

    # Try PyPDF2
    try:
        import PyPDF2, io
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for p in reader.pages:
            t = p.extract_text() or ""
            pages.append(t)
        return _clean_text("\n".join(pages))
    except Exception:
        pass

    print("[WARN] Não foi possível extrair texto do PDF (instale pdfminer.six ou PyPDF2).")
    return ""

# -----------------------
# Fetchers
# -----------------------
def fetch_html(url: str) -> str | None:
    r = requests.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    ct = r.headers.get("Content-Type", "").lower()
    # permite text/html mesmo que venha sem charset
    if "text/html" in ct or "text/" in ct or ct == "":
        r.encoding = r.encoding or "utf-8"
        html = r.text
        if any(p in html for p in ("Ã©", "Ã§", "Ã£", "Ãº", "Ã¡", "Ãª", "Ã³", "Ã­")):
            try:
                html = html.encode("latin1").decode("utf-8")
            except Exception:
                pass
        return html
    return None

def fetch_pdf(url: str) -> bytes | None:
    r = requests.get(url, timeout=HTTP_TIMEOUT, stream=True)
    r.raise_for_status()
    ct = r.headers.get("Content-Type", "").lower()
    if "application/pdf" in ct or url.lower().endswith(".pdf"):
        return r.content
    return None

# -----------------------
# HTML parsing em blocos
# -----------------------
def _table_to_markdown(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        rows.append(cells)
    if not rows:
        return ""
    header = rows[0]
    sep = ["---"] * len(header)
    out = [" | ".join(header), " | ".join(sep)]
    for r in rows[1:]:
        out.append(" | ".join(r))
    return "\n".join(out)

def _block_to_text(node: Tag) -> str:
    if node.name in ("ul", "ol"):
        items = []
        for li in node.find_all("li", recursive=False):
            t = _clean_text(li.get_text(" ", strip=True))
            if t:
                items.append(f"- {t}")
        return "\n".join(items)
    if node.name == "table":
        return _table_to_markdown(node)
    return _clean_text(node.get_text(" ", strip=True))

def gather_blocks_with_headings_from_html(html: str, url: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "nav", "aside", "footer", "form"]):
        t.decompose()

    main = soup.find(["main", "article"]) or soup.body or soup
    headings = {f"h{i}" for i in range(1, 7)}

    blocks: List[Dict[str, Any]] = []
    stack: List[Tuple[int, str]] = []  # (level, title)

    for el in main.descendants:
        if not isinstance(el, Tag):
            continue
        if el.name in headings:
            level = int(el.name[1])
            title = _clean_text(el.get_text(" ", strip=True))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        elif el.name in ("p", "ul", "ol", "pre", "blockquote", "table"):
            text = _block_to_text(el)
            if not text:
                continue
            blocks.append({
                "text": text,
                "type": el.name,
                "level": stack[-1][0] if stack else 0,
                "heading_path": [t for _, t in stack] if stack else [],
                "url": url,
                "content_type": "text/html",
            })

    if not blocks:
        body_text = _clean_text(main.get_text("\n", strip=True))
        if body_text:
            blocks.append({
                "text": body_text,
                "type": "body",
                "level": 0,
                "heading_path": ["Documento"],
                "url": url,
                "content_type": "text/html",
            })
    return blocks

# -----------------------
# PDF -> blocos
# -----------------------
def gather_blocks_from_pdf_bytes(pdf_bytes: bytes, url: str) -> List[Dict[str, Any]]:
    text = pdf_bytes_to_text(pdf_bytes)
    if not text:
        return []
    # separa por parágrafos (dupla quebra preferencial)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    blocks: List[Dict[str, Any]] = []
    for p in paras:
        blocks.append({
            "text": _clean_text(p),
            "type": "pdf_paragraph",
            "level": 0,
            "heading_path": ["PDF"],
            "url": url,
            "content_type": "application/pdf",
        })
    return blocks

# -----------------------
# Descoberta de links (HTML)
# -----------------------
def discover_links(html: str, base_url: str, origin: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        abs_url = urljoin(origin, href)
        abs_url, _ = urldefrag(abs_url)  # remove #fragment
        out.append(abs_url)
    return out

def same_domain(url: str, root: str) -> bool:
    return urlparse(url).netloc == urlparse(root).netloc

def same_path_prefix(url: str, root: str) -> bool:
    # mantém no mesmo prefixo de path do SOURCE_URL
    up, rp = urlparse(url).path.rstrip("/"), urlparse(root).path.rstrip("/")
    return up.startswith(rp)

def allowed_url(url: str, root: str) -> bool:
    if not same_domain(url, root):
        return False
    if CRAWL_SAME_PATH and not same_path_prefix(url, root):
        return False
    return True

# -----------------------
# Crawler BFS (HTML + PDF)
# -----------------------
def crawl_seed(seed: str) -> List[Dict[str, Any]]:
    visited = set()
    q = deque()
    q.append((seed, 0))
    pages_fetched = 0

    all_blocks: List[Dict[str, Any]] = []

    while q and pages_fetched < CRAWL_MAX_PAGES:
        url, depth = q.popleft()
        url, _ = urldefrag(url)
        if url in visited: 
            continue
        visited.add(url)

        try:
            is_pdf = url.lower().endswith(".pdf")
            pdf_bytes = None
            html = None

            if is_pdf:
                pdf_bytes = fetch_pdf(url)
            else:
                # tenta HTML; se vier PDF por content-type, também cobre
                resp = requests.get(url, timeout=HTTP_TIMEOUT, stream=True)
                resp.raise_for_status()
                ct = resp.headers.get("Content-Type", "").lower()
                if "application/pdf" in ct or url.lower().endswith(".pdf"):
                    pdf_bytes = resp.content
                elif "text/html" in ct or "text/" in ct or ct == "":
                    resp.encoding = resp.encoding or "utf-8"
                    html = resp.text
                    if any(p in html for p in ("Ã©", "Ã§", "Ã£", "Ãº", "Ã¡", "Ãª", "Ã³", "Ã­")):
                        try:
                            html = html.encode("latin1").decode("utf-8")
                        except Exception:
                            pass
                else:
                    # ignora outros tipos
                    continue

            if html is not None:
                pages_fetched += 1
                # blocos HTML
                blocks = gather_blocks_with_headings_from_html(html, url)
                all_blocks.extend(blocks)

                # descoberta de links
                if depth < CRAWL_MAX_DEPTH:
                    for link in discover_links(html, SOURCE_URL, url):
                        if allowed_url(link, SOURCE_URL) and link not in visited:
                            q.append((link, depth + 1))

            elif pdf_bytes is not None:
                pages_fetched += 1
                # blocos PDF
                blocks = gather_blocks_from_pdf_bytes(pdf_bytes, url)
                if blocks:
                    all_blocks.extend(blocks)

            time.sleep(CRAWL_DELAY_SEC)  # polidez

        except requests.exceptions.RequestException as e:
            print(f"[WARN] Falha ao buscar {url}: {e}")
            continue

    return all_blocks

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
# LLM-assisted chunking
# -----------------------
SEG_SCHEMA = {
    "type": "object",
    "properties": {
        "chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "start_idx": {"type": "integer"},
                    "end_idx": {"type": "integer"}
                },
                "required": ["start_idx", "end_idx"]
            }
        }
    },
    "required": ["chunks"]
}

def extract_first_json(s: str) -> dict | None:
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return json.loads(s[start:end+1])
    except Exception:
        return None

def llm_segment_blocks(blocks: List[Dict[str, Any]], call_llm, base_url: str) -> List[Dict[str, Any]]:
    if not blocks:
        return []
    chunks: List[Dict[str, Any]] = []
    for start in range(0, len(blocks), LLM_MAX_BLOCKS):
        window = blocks[start:start + LLM_MAX_BLOCKS]
        catalog = [{
            "i": i,
            "type": b["type"],
            "heading": " > ".join(b.get("heading_path", []))[:120],
            "text": b["text"][:900]
        } for i, b in enumerate(window)]

        system = (
            "Você é um segmentador de documentos. "
            "Agrupe blocos em segmentos SEMANTICAMENTE coesos, respeitando listas/tabelas e evitando cortes no meio do raciocínio."
        )
        user = (
            "Divida os blocos abaixo em segmentos. Regras:\n"
            f"- Cada segmento deve ter até ~{LLM_TARGET_MAX_CHARS} caracteres quando concatenado.\n"
            "- Não invente conteúdo; apenas defina fronteiras.\n"
            "- Use este JSON estrito (sem comentários):\n"
            + json.dumps(SEG_SCHEMA, ensure_ascii=False)
            + "\n\nBlocos:\n"
            + json.dumps(catalog, ensure_ascii=False)
        )

        resp = call_llm([
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ])
        data = extract_first_json(resp) or {"chunks": [{"start_idx": 0, "end_idx": len(window)-1}]}
        segs = data.get("chunks", [])

        for seg in segs:
            si, ei = seg.get("start_idx", 0), seg.get("end_idx", len(window)-1)
            if not (0 <= si <= ei < len(window)):
                continue
            seg_blocks = window[si:ei+1]
            merged = "\n".join(b["text"] for b in seg_blocks).strip()
            if not merged:
                continue
            hp = seg_blocks[0].get("heading_path", []) if seg_blocks else []
            types = sorted(list({b["type"] for b in seg_blocks}))
            chunks.append({
                "text": merged,
                "section_title": seg.get("title") or (hp[-1] if hp else None),
                "heading_path": hp,
                "url": seg_blocks[0]["url"] if seg_blocks else base_url,
                "element_types": types,
                "summary": seg.get("summary"),
                "content_type": seg_blocks[0].get("content_type", "text/plain"),
            })
    return chunks

# -----------------------
# Fallback 1: chunk semântico por embeddings
# -----------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def semantic_boundaries(embs: np.ndarray, levels: List[int],
                        abs_thresh: float = SEM_BREAK_ABS, drop_thresh: float = SEM_BREAK_DROP) -> List[int]:
    n = embs.shape[0]
    if n == 1:
        return [0]
    sims = [cosine_sim(embs[i], embs[i+1]) for i in range(n-1)]
    boundaries = [0]
    for i in range(n-1):
        sim = sims[i]
        prev_sim = sims[i-1] if i > 0 else sim
        next_sim = sims[i+1] if i < len(sims)-1 else sim
        local_drop = max(prev_sim - sim, 0.0) + max(next_sim - sim, 0.0)
        heading_hint = (i+1 < n and levels[i+1] < levels[i])
        if (sim < abs_thresh) or (local_drop > drop_thresh) or heading_hint:
            boundaries.append(i+1)
    out, seen = [], set()
    for b in boundaries:
        if b not in seen:
            out.append(b); seen.add(b)
    return out

def semantic_chunk_from_blocks(blocks: List[Dict[str, Any]], emb_model: SentenceTransformer, base_url: str) -> List[Dict[str, Any]]:
    if not blocks:
        return []
    texts = [b["text"] for b in blocks]
    embs = emb_model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    embs = np.asarray(embs, dtype="float32")
    levels = [b.get("level", 0) for b in blocks]
    starts = semantic_boundaries(embs, levels)

    sections: List[Dict[str, Any]] = []
    for si, sj in zip(starts, starts[1:] + [len(blocks)]):
        seg = blocks[si:sj]
        text = "\n".join(s["text"] for s in seg).strip()
        hp = seg[0].get("heading_path", []) if seg else []
        types = sorted(list({s["type"] for s in seg}))
        if not text:
            continue
        # aplica min/max + overlap
        if len(text) <= SEM_MAX_CHARS:
            sections.append({
                "text": text,
                "section_title": hp[-1] if hp else None,
                "heading_path": hp,
                "url": seg[0]["url"] if seg else base_url,
                "element_types": types,
                "content_type": seg[0].get("content_type", "text/plain"),
            })
        else:
            start_idx = 0
            while start_idx < len(text):
                end_idx = min(start_idx + SEM_MAX_CHARS, len(text))
                piece = text[start_idx:end_idx]
                last_dot = piece.rfind(".")
                if end_idx < len(text) and last_dot > int(SEM_MAX_CHARS * 0.5):
                    piece = piece[:last_dot+1]
                    end_idx = start_idx + len(piece)
                sections.append({
                    "text": piece.strip(),
                    "section_title": hp[-1] if hp else None,
                    "heading_path": hp,
                    "url": seg[0]["url"] if seg else base_url,
                    "element_types": types,
                    "content_type": seg[0].get("content_type", "text/plain"),
                })
                start_idx = max(end_idx - SEM_OVERLAP_CHARS, end_idx)
    # filtra micro-chunks + une vizinhos pequenos
    out = []
    buf = ""
    meta = None
    for sec in sections:
        if len(sec["text"]) < SEM_MIN_CHARS:
            buf = (buf + "\n" + sec["text"]).strip() if buf else sec["text"]
            meta = meta or sec
            continue
        if buf:
            meta2 = dict(meta)
            meta2["text"] = buf
            out.append(meta2)
            buf, meta = "", None
        out.append(sec)
    if buf:
        meta2 = dict(meta)
        meta2["text"] = buf
        out.append(meta2)
    return out

# -----------------------
# Fallback 2: split por tamanho (simples)
# -----------------------
def size_split(text: str, size: int, overlap: int) -> List[str]:
    if len(text) <= size:
        return [text]
    chunks, n, start = [], len(text), 0
    while start < n:
        end = min(start + size, n)
        piece = text[start:end]
        if end < n:
            last_dot = piece.rfind(".")
            if last_dot > int(size * 0.6):
                piece = piece[:last_dot+1]
                end = start + len(piece)
        chunks.append(piece.strip())
        start = max(end - overlap, end)
    return chunks

def size_chunk_from_blocks(blocks: List[Dict[str, Any]], base_url: str) -> List[Dict[str, Any]]:
    if not blocks:
        return []
    whole = "\n".join(b["text"] for b in blocks)
    hp = blocks[0].get("heading_path", []) if blocks else []
    types = sorted(list({b["type"] for b in blocks}))
    ct = blocks[0].get("content_type", "text/plain") if blocks else "text/plain"
    out = []
    for c in size_split(whole, CHUNK_SIZE_FALLBACK, CHUNK_OVERLAP_FALLBACK):
        out.append({
            "text": c,
            "section_title": hp[-1] if hp else None,
            "heading_path": hp,
            "url": blocks[0]["url"] if blocks else base_url,
            "element_types": types,
            "content_type": ct,
        })
    return out

# -----------------------
# Estruturas do RAG
# -----------------------
@dataclass
class RagStore:
    model: SentenceTransformer
    index: faiss.IndexFlatIP
    embeddings: np.ndarray
    chunks: List[Dict[str, Any]]  # {"text","section_title","heading_path","url","element_types","summary"?,"content_type"}

def _save_artifacts(raw_text: str, chunks: List[Dict[str, Any]], embs: np.ndarray, index: faiss.IndexFlatIP):
    with open(RAW_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(raw_text)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    np.save(EMB_PATH, embs)
    faiss.write_index(index, INDEX_PATH)

def _load_artifacts():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH) and os.path.exists(EMB_PATH)):
        return None, None, None
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks_json = json.load(f)
    if chunks_json and isinstance(chunks_json[0], str):
        chunks = [{"text": t, "section_title": None, "heading_path": [], "url": SOURCE_URL, "element_types": [], "content_type": "text/plain"} for t in chunks_json]
    else:
        chunks = chunks_json
    embs = np.load(EMB_PATH)
    index = faiss.read_index(INDEX_PATH)
    return chunks, embs, index

# -----------------------
# Ollama com fallback
# -----------------------
def call_ollama(messages: list[dict], model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST, stream: bool = False) -> str:
    def _chat() -> str:
        url = f"{host}/api/chat"
        payload = {"model": model, "messages": messages, "stream": stream}
        r = requests.post(url, json=payload, timeout=120)
        if r.status_code in (404, 405):
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
            else:
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
        if isinstance(e, RuntimeError) and str(e) == "OLLAMA_NO_CHAT_ENDPOINT":
            return _generate_fallback()
        if hasattr(e, "response") and getattr(e.response, "status_code", None) in (404, 405):
            return _generate_fallback()
        raise

# -----------------------
# Build Store (crawl + ingest + index)
# -----------------------
def validate_coherence(chunks: List[Dict[str, Any]], emb_model: SentenceTransformer, thresh: float = 0.60) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in chunks:
        text = c["text"]
        if len(text) <= 600:
            out.append(c); continue
        paras = [p for p in re.split(r"\n+", text) if p.strip()]
        if len(paras) < 2:
            out.append(c); continue
        embs = emb_model.encode(paras, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        embs = np.asarray(embs, dtype="float32")
        sims = [float(np.dot(embs[i], embs[i+1])) for i in range(len(embs)-1)]
        mean_sim = float(np.mean(sims)) if sims else 1.0
        if mean_sim >= thresh and len(text) <= LLM_TARGET_MAX_CHARS * 1.5:
            out.append(c)
        else:
            for piece in size_split(text, CHUNK_SIZE_FALLBACK, CHUNK_OVERLAP_FALLBACK):
                c2 = dict(c); c2["text"] = piece
                out.append(c2)
    return out

def build_store(reindex: bool = False) -> RagStore:
    model = SentenceTransformer(MODEL_EMB)

    if not reindex:
        loaded = _load_artifacts()
        if loaded[0] is not None:
            chunks, embs, index = loaded
            if embs.ndim == 2:
                return RagStore(model, index, embs.astype("float32"), chunks)

    # 1) crawl coletando blocos (HTML + PDF)
    blocks = crawl_seed(SOURCE_URL)
    if not blocks:
        raise RuntimeError("Crawler não coletou nenhum conteúdo.")

    # 2) LLM-assisted chunking (opcional)
    llm_chunks: List[Dict[str, Any]] = []
    if USE_LLM_CHUNK:
        try:
            llm_chunks = llm_segment_blocks(blocks, call_ollama, SOURCE_URL)
        except Exception as e:
            print(f"[WARN] LLM chunking falhou: {e}")
            llm_chunks = []

    # 3) Fallback semântico por embeddings (se nada veio do LLM)
    if not llm_chunks:
        llm_chunks = semantic_chunk_from_blocks(blocks, model, SOURCE_URL)

    # 4) Validação de coerência + eventual split por tamanho
    final_chunks = validate_coherence(llm_chunks, model)

    # 5) Texto bruto (opcional para debug)
    all_text = "\n\n---\n\n".join([c["text"] for c in final_chunks])

    # 6) Embeddings + FAISS
    texts = [c["text"] for c in final_chunks]
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype="float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    _save_artifacts(all_text, final_chunks, embs, index)
    return RagStore(model, index, embs, final_chunks)

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
# FastAPI com lifespan
# -----------------------
@dataclass
class _StoreHolder:
    store: RagStore | None = None

STORE_HOLDER = _StoreHolder()

@asynccontextmanager
async def lifespan(app: FastAPI):
    reindex_flag = os.getenv("RAG_REINDEX", "0") == "1"
    STORE_HOLDER.store = build_store(reindex=reindex_flag)
    yield

app = FastAPI(title="RAG UNIVESP — Crawler HTML+PDF + LLM-assisted Chunking", lifespan=lifespan)

@app.get("/healthz")
def healthz():
    st = STORE_HOLDER.store
    return {"status": "ok", "chunks": len(st.chunks) if st else 0, "use_llm_chunk": USE_LLM_CHUNK}

@app.post("/chat")
def chat(payload: dict):
    st = STORE_HOLDER.store
    if st is None:
        return JSONResponse({"error": "store indisponível"}, status_code=500)

    query = (payload.get("query") or "").strip()
    if not query:
        return JSONResponse({"error": "query vazia"}, status_code=400)

    results = retrieve(st, query, k=TOP_K)
    contexts = [st.chunks[i]["text"] for i, _ in results]
    messages = build_prompt(query, contexts)
    try:
        answer = call_ollama(messages)
    except requests.exceptions.ConnectionError:
        return JSONResponse({"error": f"Não consegui conectar ao Ollama em {OLLAMA_HOST}"}, status_code=502)

    sources = []
    for i, s in results:
        c = st.chunks[i]
        snippet = make_snippet(c["text"], query, max_chars=360)
        sources.append({
            "chunk_id": i,
            "score": round(s, 6),
            "text": c["text"],
            "snippet": snippet,
            "snippet_marked": highlight_terms(snippet, query),
            "section_title": c.get("section_title"),
            "heading_path": c.get("heading_path"),
            "url": c.get("url"),
            "element_types": c.get("element_types"),
            "summary": c.get("summary"),
            "content_type": c.get("content_type"),
        })

    return {"answer": answer, "sources": sources, "k": TOP_K}

# -----------------------
# CLI
# -----------------------
def cli(force_reindex: bool = False):
    if STORE_HOLDER.store is None:
        STORE_HOLDER.store = build_store(reindex=force_reindex)

    print("RAG UNIVESP — Crawler HTML+PDF + LLM-assisted Chunking")
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
            hits = retrieve(STORE_HOLDER.store, q, k=TOP_K)
            ctxs = [STORE_HOLDER.store.chunks[i]["text"] for i, _ in hits]
            msgs = build_prompt(q, ctxs)
            resp = call_ollama(msgs)
        except requests.exceptions.ConnectionError:
            print("\n[ERRO] Não consegui conectar ao Ollama em", OLLAMA_HOST)
            print("Verifique se o serviço está rodando e se o modelo foi baixado.")
            continue
        except Exception as e:
            print("\n[ERRO] Falha ao chamar o modelo:", repr(e), "\n")
            continue

        print("\nRESPOSTA:\n", resp, "\n")

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
        if args.reindex:
            os.environ["RAG_REINDEX"] = "1"
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        cli(force_reindex=args.reindex)

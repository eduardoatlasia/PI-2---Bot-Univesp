#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bot UNIVESP - RAG do Manual do Aluno (chunk estruturado)
- Crawler (requests + BeautifulSoup) com correção de encoding
- Chunking estruturado por headings (H1–H6), com subdivisão por parágrafo
- Embeddings (sentence-transformers: paraphrase-multilingual-MiniLM-L12-v2)
- FAISS (Inner Product com vetores normalizados)
- Gemma 3 via Ollama (com fallback /api/chat -> /api/generate)

Uso:
  python bot_univesp.py
  python bot_univesp.py --api
  curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
    -d '{"query":"Como funciona o fórum de dúvidas?"}'

Opções:
  --reindex   força reindexar (ignora arquivos salvos)
Env:
  OLLAMA_HOST  (default: http://localhost:11434)
  OLLAMA_MODEL (default: gemma3:1b-it-fp16)
  RAG_REINDEX  (default: 0; se 1, reindexa no startup da API)
"""

import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from contextlib import asynccontextmanager

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")

# Limites para subdivisão de seções (aplicados apenas se a seção for muito longa)
CHUNK_SIZE = 1200          # tamanho alvo por chunk (caracteres) para subdivisão
CHUNK_OVERLAP = 150       # overlap quando subdividir uma seção longa
TOP_K = 15
SYSTEM_PROMPT = (
    "1. Você é um assistente que responde dúvidas dos alunos da Univesp."
    "2. Use os trechos fornecidos do material do aluno e a BASE DE CONHECIMENTO para responder."
    "3. Se não encontrar a resposta nos trechos, responda 'Não sei' ou 'Não encontrei essa informação no Manual do Aluno'."
    "4. Responda em português claro e objetivo."
    "5. Revise a reposta e garanta que não há redundância. Não crie itens para os cursos que não existem. Exemplo: listar Ciência de Dados e Bacharelado em Ciência de dados - são a mesma coisa"
    "6. Realize inferências simples se necessário."
    "----"
    "BASE DE CONHECIMENTO"
    "Os cursos a seguir são de nível superior e3 suas durações são: "
    "- Letras (Licenciatura, 3 anos)"
    "- Matemática (Licenciatura, 3 anos)"
    "- Pedagogia (Licenciatura, 3 anos)"
    "- Tecnólogo em Gestão pública (Tecnólogo, 3 anos)"
    "- Tecnologia em processos Gerenciais (Tecnólogo, 3 anos)"
    "- Administração (Bacharelado, 4 anos)"
    "- Tecnologia da Informação (Bacharelado, 3 anos)"
    "- Ciência de dados (Bacharelado, 4 anos)"
    "- Engenharia de Computação (Bacharelado, 5 anos)"
    "- Engenharia de Produção (Bacharelado, 5 anos)"
)

# Persistência
INDEX_PATH = "faiss_univesp.index"
CHUNKS_PATH = "chunks_univesp.json"      # agora armazena lista de dicts com metadados
EMB_PATH = "embeddings_univesp.npy"
RAW_TXT_PATH = "manual_univesp.txt"

# -----------------------
# Crawler + limpeza
# -----------------------
def fetch_utf8(url: str) -> str:
    """Baixa HTML e tenta garantir UTF-8 (com fallback latin1->utf8 contra mojibake)."""
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    r.encoding = r.encoding or "utf-8"
    html = r.text
    if any(p in html for p in ("Ã©", "Ã§", "Ã£", "Ãº", "Ã¡", "Ãª", "Ã³", "Ã­")):
        try:
            html = html.encode("latin1").decode("utf-8")
        except Exception:
            pass
    return html

def _clean_text(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()

def _table_to_markdown(table: Tag) -> str:
    # Converte <table> em markdown simples
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        rows.append(cells)
    if not rows:
        return ""
    # Cabeçalho (se houver th na primeira linha)
    header = rows[0]
    sep = ["---"] * len(header)
    out = [" | ".join(header), " | ".join(sep)]
    for r in rows[1:]:
        out.append(" | ".join(r))
    return "\n".join(out)

def _collect_block_text(node: Tag) -> str:
    """Coleta texto de um bloco preservando quebras; converte listas e tabelas."""
    if node.name in ("ul", "ol"):
        items = []
        for li in node.find_all("li", recursive=False):
            li_text = _clean_text(li.get_text(" ", strip=True))
            if li_text:
                items.append(f"- {li_text}")
        return "\n".join(items)
    if node.name == "table":
        return _table_to_markdown(node)
    # Paragraph-like
    text = _clean_text(node.get_text(" ", strip=True))
    return text

def _subsplit_long_text(text: str, size: int, overlap: int) -> List[str]:
    """Quebra um texto longo respeitando parágrafos; aplica split suave + overlap."""
    if len(text) <= size:
        return [text]
    # 1) quebre por parágrafos
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks = []
    buf = ""
    for p in paras:
        # se um único parágrafo for muito maior que 'size', faça um soft split nele
        if len(p) > size * 1.5:
            # split por sentenças básicas (pontos finais, !, ?)
            sentences = re.split(r"(?<=[\.\!\?])\s+", p)
            for s in sentences:
                if not s.strip():
                    continue
                if len(buf) + len(s) + 1 > size and buf:
                    chunks.append(buf.strip())
                    buf = s
                else:
                    buf = (buf + " " + s).strip() if buf else s.strip()
        else:
            if len(buf) + len(p) + 1 > size and buf:
                chunks.append(buf.strip())
                buf = p
            else:
                buf = (buf + "\n" + p).strip() if buf else p
    if buf:
        chunks.append(buf.strip())

    # 2) Se ainda houver chunks muito grandes, aplique janela deslizante com overlap
    final = []
    for c in chunks:
        if len(c) <= size:
            final.append(c)
            continue
        start = 0
        while start < len(c):
            end = min(start + size, len(c))
            piece = c[start:end]
            # tenta cortar no fim de frase para suavizar
            last_dot = piece.rfind(".")
            if end < len(c) and last_dot > int(size * 0.5):
                piece = piece[:last_dot + 1]
                end = start + len(piece)
            final.append(piece.strip())
            start = max(end - overlap, end)
    return final

def html_to_structured_chunks(html: str, base_url: str) -> List[Dict[str, Any]]:
    """
    Cria chunks estruturados por hierarquia de headings (H1–H6).
    Para cada seção, agrega parágrafos/listas/tabelas até o próximo heading
    de mesmo nível ou superior. Se a seção for longa, subdivide respeitando parágrafos.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remover ruído
    for tag in soup(["script", "style", "noscript", "nav", "aside", "footer", "form"]):
        tag.decompose()

    # Heurística de "main"
    main = soup.find(["main", "article"]) or soup.body or soup

    # Coletar todos nós que são headings ou blocos
    headings = set([f"h{i}" for i in range(1, 7)])
    blocks = []
    for el in main.descendants:
        if isinstance(el, Tag):
            if el.name in headings:
                level = int(el.name[1])
                title = _clean_text(el.get_text(" ", strip=True))
                blocks.append(("heading", level, el, title))
            elif el.name in ("p", "ul", "ol", "pre", "blockquote", "table"):
                blocks.append(("block", None, el, None))

    # Varre e forma seções por heading
    chunks: List[Dict[str, Any]] = []
    stack: List[Tuple[int, str]] = []  # (level, title)
    current_content: List[str] = []
    current_meta: Dict[str, Any] = {
        "section_title": None,
        "heading_path": [],
        "url": base_url,
        "element_types": set(),
    }

    def flush_section():
        if not current_content:
            return
        section_text = _clean_text("\n".join([c for c in current_content if c.strip()]))
        if not section_text:
            return
        # subdivide se necessário
        parts = _subsplit_long_text(section_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for part in parts:
            chunks.append({
                "text": part,
                "section_title": current_meta["section_title"],
                "heading_path": list(current_meta["heading_path"]),
                "url": current_meta["url"],
                "element_types": sorted(list(current_meta["element_types"])),
            })

    for kind, level, node, title in blocks:
        if kind == "heading":
            # finaliza a seção anterior
            flush_section()
            current_content = []
            current_meta = {
                "section_title": title,
                "heading_path": [],
                "url": base_url,
                "element_types": set(),
            }
            # ajusta a pilha de headings
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current_meta["heading_path"] = [t for _, t in stack]
        else:
            # bloco de conteúdo
            text = _collect_block_text(node)
            if text:
                current_content.append(text)
                current_meta["element_types"].add(node.name)

    # flush final
    flush_section()

    # Se a página não tinha headings, cria um único chunk com corpo inteiro
    if not chunks:
        body_text = _clean_text(main.get_text("\n", strip=True))
        for part in _subsplit_long_text(body_text, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append({
                "text": part,
                "section_title": "Documento",
                "heading_path": ["Documento"],
                "url": base_url,
                "element_types": ["body"],
            })

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
    chunks: List[Dict[str, Any]]  # cada chunk: {"text", "section_title", "heading_path", "url", "element_types"}

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
    # retrocompatibilidade: se era lista de strings, converte para dicts simples
    if chunks_json and isinstance(chunks_json[0], str):
        chunks = [{"text": t, "section_title": None, "heading_path": [], "url": SOURCE_URL, "element_types": []} for t in chunks_json]
    else:
        chunks = chunks_json
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
    structured_chunks = html_to_structured_chunks(html, SOURCE_URL)

    # Texto bruto (apenas para inspeção/depuração)
    all_text = "\n\n---\n\n".join([c["text"] for c in structured_chunks])

    texts = [c["text"] for c in structured_chunks]
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype="float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    _save_artifacts(all_text, structured_chunks, embs, index)
    return RagStore(model, index, embs, structured_chunks)

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

app = FastAPI(title="RAG UNIVESP (Manual do Aluno) + Gemma3 1B — Chunk Estruturado", lifespan=lifespan)

@app.get("/healthz")
def healthz():
    st = STORE_HOLDER.store
    return {"status": "ok", "chunks": len(st.chunks) if st else 0}

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
            "text": c["text"],                               # chunk completo
            "snippet": snippet,                               # recorte curto
            "snippet_marked": highlight_terms(snippet, query),
            "section_title": c.get("section_title"),
            "heading_path": c.get("heading_path"),
            "url": c.get("url"),
            "element_types": c.get("element_types"),
        })

    return {"answer": answer, "sources": sources, "k": TOP_K}

# -----------------------
# CLI
# -----------------------
def cli(force_reindex: bool = False):
    if STORE_HOLDER.store is None:
        STORE_HOLDER.store = build_store(reindex=force_reindex)

    print("RAG UNIVESP (Manual do Aluno) - Gemma3 — Chunk Estruturado")
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
            print("Verifique se o serviço está rodando e se o modelo foi baixado:")
            print("  $ ollama serve")
            print("  $ ollama pull gemma3:270m\n")
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

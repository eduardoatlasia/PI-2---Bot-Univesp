# streamlit_app.py
import json
import time
from typing import Dict, Any, List
import re
import requests
import streamlit as st

# ---------------------------
# Config da página
# ---------------------------
st.set_page_config(
    page_title="Assistente UNIVESP Chat",
    page_icon="🎓",
    layout="wide",
)

def render_answer_with_think(answer: str):
    """
    Transforma blocos <think>...</think> em seções colapsadas azuis.
    """
    pattern = r"<think>(.*?)</think>"
    def repl(match):
        content = match.group(1).strip()
        html_block = f"""
        <details style='margin:0.5em 0;padding:0.5em;border-left:4px solid #1E90FF;background-color:#F0F8FF;border-radius:6px;'>
          <summary style='color:#1E90FF;font-weight:bold;cursor:pointer;'>💭 Raciocínio </summary>
          <div style='margin-top:0.5em;color:#004080;'>{content}</div>
        </details>
        """
        return html_block
    html_answer = re.sub(pattern, repl, answer, flags=re.DOTALL)
    return html_answer

# ---------------------------
# Helpers HTTP
# ---------------------------
def safe_post(url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        if r.headers.get("content-type", "").startswith("application/json"):
            data = r.json()
        else:
            data = {"error": f"Resposta não-JSON ({r.status_code})", "raw": r.text[:8000]}
        if r.status_code >= 400:
            return {"error": data.get("error") or f"HTTP {r.status_code}", "raw": data}
        return data
    except requests.exceptions.ConnectionError:
        return {"error": "Não consegui conectar ao backend. Verifique a API FastAPI.", "raw": None}
    except requests.exceptions.Timeout:
        return {"error": "Timeout ao chamar o backend.", "raw": None}
    except Exception as e:
        return {"error": f"Falha inesperada: {e}", "raw": None}


def safe_get(url: str, timeout: int = 15) -> Dict[str, Any]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.headers.get("content-type", "").startswith("application/json"):
            data = r.json()
        else:
            data = {"error": f"Resposta não-JSON ({r.status_code})", "raw": r.text[:2000]}
        if r.status_code >= 400:
            return {"error": data.get("error") or f"HTTP {r.status_code}", "raw": data}
        return data
    except requests.exceptions.ConnectionError:
        return {"error": "Não consegui conectar ao backend.", "raw": None}
    except Exception as e:
        return {"error": f"Falha inesperada: {e}", "raw": None}

# ---------------------------
# Estado
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = []

if "backend_url" not in st.session_state:
    st.session_state.backend_url = "http://localhost:8000"

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.title("⚙️ Configurações")

    st.session_state.backend_url = st.text_input(
        "URL da API FastAPI",
        value=st.session_state.backend_url,
        help="Ex.: http://localhost:8000",
    )

    cols = st.columns(2)
    with cols[0]:
        if st.button("🔍 Health check"):
            hc = safe_get(f"{st.session_state.backend_url}/healthz")
            if "error" in hc:
                st.error(hc["error"])
            else:
                st.success("API ok!")
                st.caption(hc)
    with cols[1]:
        if st.button("🧹 Limpar conversa"):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.subheader("Exportar")
    if st.session_state.messages:
        export_json = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
        st.download_button(
            label="⬇️ Baixar conversa (.json)",
            data=export_json.encode("utf-8"),
            file_name=f"chat_univesp_{int(time.time())}.json",
            mime="application/json",
        )

    st.divider()
    st.caption(
        "Este app chama POST /chat com payload {'query': '<pergunta>'} e exibe 'answer' e 'sources'."
    )

# ---------------------------
# UI principal
# ---------------------------
st.title("🎓 Assistente UNIVESP — Pergunte sobre o Manual do Aluno")

# Renderiza histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(msg["content"])
            # Exibir fontes se houver
            if "sources" in msg and msg["sources"]:
                with st.expander(f"📚 Fontes ({len(msg['sources'])})", expanded=False):
                    for j, s in enumerate(msg["sources"], start=1):
                        section = s.get("section_title") or "Seção"
                        heading_path = " › ".join(s.get("heading_path") or [])
                        score = s.get("score")
                        url = s.get("url")
                        snippet_marked = s.get("snippet_marked") or s.get("snippet") or ""
                        with st.container(border=True):
                            st.markdown(f"**{j}. {section}**  \n"
                                        f"{('`' + heading_path + '`') if heading_path else ''}")
                            st.markdown(f"Score: `{score}`")
                            if url:
                                st.markdown(f"[Abrir página]({url})")
                            st.markdown("---")
                            # snippet com marcação em **bold**
                            st.markdown(snippet_marked)
                            with st.expander("Ver chunk completo"):
                                st.markdown(s.get("text") or "")

        else:
            st.markdown(msg["content"])

# Entrada do usuário
user_input = st.chat_input("Digite sua pergunta sobre o Manual do Aluno…")

if user_input:
    # Adiciona pergunta ao histórico
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Chama backend
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_ ⏳... pensando ... _")

        payload = {"query": user_input}
        data = safe_post(f"{st.session_state.backend_url}/chat", payload)

        if "error" in data:
            placeholder.error(data["error"])
            if data.get("raw"):
                with st.expander("Detalhes do erro"):
                    st.write(data["raw"])
            # Também grava no histórico para manter rastro do erro
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ {data['error']}",
            })
        else:
            answer = data.get("answer", "").strip() or "_(sem conteúdo)_"
            sources = data.get("sources", []) or []

            # Renderiza resposta final
            placeholder.markdown(render_answer_with_think(answer), unsafe_allow_html=True)

            # Renderiza fontes (também ficam no histórico)
            if sources:
                with st.expander(f"📚 Fontes ({len(sources)})", expanded=False):
                    for j, s in enumerate(sources, start=1):
                        section = s.get("section_title") or "Seção"
                        heading_path = " › ".join(s.get("heading_path") or [])
                        score = s.get("score")
                        url = s.get("url")
                        snippet_marked = s.get("snippet_marked") or s.get("snippet") or ""
                        with st.container(border=True):
                            st.markdown(f"**{j}. {section}**  \n"
                                        f"{('`' + heading_path + '`') if heading_path else ''}")
                            st.markdown(f"Score: `{score}`")
                            if url:
                                st.markdown(f"[Abrir página]({url})")
                            st.markdown("---")
                            st.markdown(snippet_marked)
                            with st.expander("Ver chunk completo"):
                                st.markdown(s.get("text") or "")

            # Atualiza histórico com a resposta e fontes
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

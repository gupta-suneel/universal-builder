"""
universal_builder.py
=====================
A conversational, multi-purpose builder powered by the DeepSeek API.

Write complex multi-chapter books, generate SaaS code, design database
schemas, and more -- all through one chat interface. Runs locally on a Mac
or as a private cloud app on Streamlit Community Cloud.

FEATURES
  1. DYNAMIC PERSONA SHIFTER   - changing the sidebar persona applies on the
     very next message. The system prompt is rebuilt fresh on every call.
     Includes a "Custom" persona you can edit freely.
  2. DEEPSEEK REASONING CHAINS - the 'deepseek-reasoner' (R1) model's hidden
     chain-of-thought (delta.reasoning_content) streams live into a clean,
     collapsible "AI Thinking Process" box.
  3. LOCAL FILE AUTO-SAVING    - say "Save this chapter as kinematics.txt" and
     it writes to ~/Documents/ai-workspace/universal_book_builder/output/.
     A download button is always offered too (works on the cloud).
  4. PERSISTENT API KEY        - the key is read from secrets / env / a saved
     local file. Once provided it is never asked for again.
  5. PASSWORD GATE             - set APP_PASSWORD to keep the cloud app private.
  6. VERSATILE CONTROLS        - model picker, creativity slider, response-length
     limit, custom system prompt, response download, and conversation export.

Run locally:   streamlit run universal_builder.py
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# 0. CONSTANTS & PATHS
# ---------------------------------------------------------------------------

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

APP_DIR = Path.home() / "Documents" / "ai-workspace" / "universal_book_builder"
OUTPUT_DIR = APP_DIR / "output"
CONFIG_FILE = APP_DIR / "config.json"   # stores a remembered API key locally

MODELS = {
    "DeepSeek V3 (deepseek-chat) - fast, general purpose": "deepseek-chat",
    "DeepSeek R1 (deepseek-reasoner) - deep step-by-step thinking": "deepseek-reasoner",
}

# ---------------------------------------------------------------------------
# 1. PERSONAS  (the dropdown that drives behaviour)
# ---------------------------------------------------------------------------

PERSONAS = {
    "Master Author (multi-chapter books)": (
        "You are a master non-fiction and fiction author. You write rich, "
        "well-structured, multi-chapter books. Maintain narrative and tonal "
        "continuity across chapters, use clear headings, and keep the reader "
        "engaged. When asked for a chapter, deliver a complete, polished, "
        "publication-ready chapter -- never an outline unless explicitly asked."
    ),
    "Senior Full-Stack Engineer (SaaS apps)": (
        "You are an elite full-stack software engineer. You design and write "
        "clean, production-grade, secure code for SaaS applications. Default to "
        "modern best practices, include brief setup notes, and wrap all code in "
        "fenced code blocks with the correct language tag."
    ),
    "Database Architect (schemas & SQL)": (
        "You are a senior database architect. You design normalized, scalable "
        "schemas and write correct, well-commented SQL. Always wrap SQL in "
        "```sql fenced code blocks and note the target engine when relevant."
    ),
    "Technical Educator (clear explanations)": (
        "You are a patient technical educator. You explain complex topics in "
        "plain language with concrete examples and analogies, building from "
        "first principles so a motivated beginner can follow along."
    ),
    "Marketing Copywriter": (
        "You are a sharp marketing copywriter. You write persuasive, on-brand "
        "copy -- landing pages, emails, ads, and social posts -- with strong "
        "hooks and clear calls to action."
    ),
    "Business Strategist": (
        "You are a seasoned business strategist. You give structured, practical "
        "advice on product, go-to-market, pricing, and operations, with clear "
        "trade-offs and concrete next steps."
    ),
    "General Assistant": (
        "You are a helpful, knowledgeable, and concise assistant."
    ),
    "Custom (edit your own)": "",  # filled in from the sidebar text area
}

CUSTOM_PERSONA_LABEL = "Custom (edit your own)"

# ---------------------------------------------------------------------------
# 2. SECRETS / CONFIG HELPERS
# ---------------------------------------------------------------------------

def get_secret(key: str, default=None):
    """Safely read a Streamlit secret without crashing when none are defined."""
    try:
        return st.secrets[key]
    except Exception:
        return default


def read_saved_key():
    """Return a previously remembered API key from the local config file."""
    try:
        if CONFIG_FILE.exists():
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            return data.get("DEEPSEEK_API_KEY")
    except Exception:
        pass
    return None


def write_saved_key(api_key: str):
    """Remember the API key locally so it is not asked for again."""
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(
            json.dumps({"DEEPSEEK_API_KEY": api_key}), encoding="utf-8"
        )
        return True
    except Exception:
        return False


def resolve_api_key():
    """
    Resolve the API key once, in priority order:
      1. Streamlit secret  (used on Streamlit Community Cloud)
      2. Environment variable
      3. Remembered local config file
    Returns (key, source) where key may be None.
    """
    key = get_secret("DEEPSEEK_API_KEY")
    if key:
        return key, "secret"
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key, "environment"
    key = read_saved_key()
    if key:
        return key, "saved file"
    return None, None


# ---------------------------------------------------------------------------
# 3. PASSWORD GATE
# ---------------------------------------------------------------------------

def check_password() -> bool:
    """
    If APP_PASSWORD is configured (as a secret or env var), require it before
    showing the app. If it is not configured (typical local use), allow access.
    """
    expected = get_secret("APP_PASSWORD") or os.environ.get("APP_PASSWORD")
    if not expected:
        return True
    if st.session_state.get("auth_ok"):
        return True

    st.title("Universal Builder")
    st.caption("This app is private. Please enter the password to continue.")
    with st.form("login_form"):
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")
    if submitted:
        if pw == expected:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


# ---------------------------------------------------------------------------
# 4. FILE-OPERATIONS MODULE  (local auto-saving + downloads)
# ---------------------------------------------------------------------------

SAVE_PATTERN = re.compile(
    r"save\s+(?:this|it|the)?\s*"
    r"(?:chapter|file|code|content|page|document)?\s*"
    r"as\s+"
    r"[\"']?([\w\-. ]+\.[A-Za-z0-9]+)[\"']?",
    re.IGNORECASE,
)


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def detect_save_request(text: str):
    match = SAVE_PATTERN.search(text or "")
    return match.group(1).strip() if match else None


def extract_saveable_content(message_text: str) -> str:
    """Save just the code if the message has code blocks, else the full text."""
    if not message_text:
        return ""
    code_blocks = re.findall(r"```[\w+\-]*\n(.*?)```", message_text, re.DOTALL)
    if code_blocks:
        return "\n\n".join(b.rstrip() for b in code_blocks).strip() + "\n"
    return message_text.strip() + "\n"


def save_file(filename: str, content: str):
    """
    Try to write content to OUTPUT_DIR/filename.
    Returns (path_or_None, error_or_None). On read-only cloud filesystems the
    write may fail gracefully -- the caller still offers a download button.
    """
    safe_name = Path(filename).name
    try:
        out_dir = ensure_output_dir()
        full_path = out_dir / safe_name
        full_path.write_text(content, encoding="utf-8")
        return full_path, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def handle_possible_save(user_text: str):
    """If the latest message is a save request, package the content to save."""
    filename = detect_save_request(user_text)
    if not filename:
        return None
    last_assistant = next(
        (m["content"] for m in reversed(st.session_state.messages)
         if m["role"] == "assistant"),
        None,
    )
    if not last_assistant:
        return {"filename": filename, "content": None}
    content = extract_saveable_content(last_assistant)
    path, error = save_file(filename, content)
    return {"filename": filename, "content": content, "path": path, "error": error}


# ---------------------------------------------------------------------------
# 5. DEEPSEEK CLIENT & STREAMING
# ---------------------------------------------------------------------------

def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def build_api_messages(system_prompt: str):
    """
    Rebuild the message list fresh on every call.

    KEY FIX (persona shifter): the system prompt is injected HERE, from the
    CURRENT sidebar selection, every single time. It is never stored in
    history, so switching personas mid-chat takes effect immediately.
    """
    system_msg = {"role": "system", "content": system_prompt}
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant")
    ]
    return [system_msg] + history


def stream_response(client, model_id, messages, temperature, max_tokens):
    """Stream a DeepSeek completion, rendering reasoning + answer separately."""
    reasoning_box = st.empty()
    answer_box = st.empty()
    reasoning_text = ""
    answer_text = ""

    # Build kwargs. The reasoner model ignores sampling params, so only the
    # chat model receives temperature.
    kwargs = {"model": model_id, "messages": messages, "stream": True,
              "max_tokens": int(max_tokens)}
    if model_id == "deepseek-chat":
        kwargs["temperature"] = float(temperature)

    stream = client.chat.completions.create(**kwargs)

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        # Feature 2: capture R1's hidden reasoning chain.
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning_text += delta.reasoning_content
            with reasoning_box.container():
                with st.expander("AI Thinking Process...", expanded=True):
                    st.markdown(reasoning_text)

        if getattr(delta, "content", None):
            answer_text += delta.content
            answer_box.markdown(answer_text + "_")

    answer_box.markdown(answer_text)
    if reasoning_text:
        with reasoning_box.container():
            with st.expander("AI Thinking Process...", expanded=False):
                st.markdown(reasoning_text)
    return answer_text


# ---------------------------------------------------------------------------
# 6. STREAMLIT APP
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Universal Builder", page_icon=":bricks:",
                       layout="wide")

    if not check_password():
        return

    st.title("Universal Builder")
    st.caption("Books, SaaS apps, schemas and more - powered by DeepSeek.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    api_key, key_source = resolve_api_key()

    # --- Sidebar ----------------------------------------------------------
    with st.sidebar:
        st.header("Controls")

        # API key handling: ask only if we don't already have one.
        if api_key:
            st.success(f"API key loaded (from {key_source}).")
            with st.expander("Change API key"):
                new_key = st.text_input("New DeepSeek API key", type="password")
                if st.button("Update & remember") and new_key:
                    write_saved_key(new_key)
                    st.success("Saved. Reloading...")
                    st.rerun()
        else:
            st.warning("No API key found yet.")
            entered = st.text_input("DeepSeek API key", type="password",
                                    help="Get one at platform.deepseek.com")
            remember = st.checkbox("Remember this key on this computer", value=True)
            if entered:
                api_key = entered
                if remember:
                    write_saved_key(entered)

        st.divider()

        model_label = st.selectbox("Model", list(MODELS.keys()), index=0)
        model_id = MODELS[model_label]

        persona_name = st.selectbox("AI Persona", list(PERSONAS.keys()), index=0,
                                    help="Applies to your NEXT message.")

        # Custom persona editor.
        if persona_name == CUSTOM_PERSONA_LABEL:
            system_prompt = st.text_area(
                "Custom system prompt",
                value=st.session_state.get("custom_prompt",
                                           "You are a helpful expert assistant."),
                height=140,
            )
            st.session_state.custom_prompt = system_prompt
        else:
            system_prompt = PERSONAS[persona_name]
        st.info(f"Active persona: **{persona_name}**")

        st.divider()
        st.subheader("Generation settings")
        temperature = st.slider(
            "Creativity (temperature)", 0.0, 1.5, 0.7, 0.1,
            help="Higher = more creative. ~1.3 for creative writing, "
                 "0.0 for precise code. (Ignored by the R1 reasoner model.)",
        )
        max_tokens = st.slider("Max response length (tokens)", 256, 8192, 4096, 256)

        st.divider()
        st.caption(f"Local saves go to:\n`{OUTPUT_DIR}`")

        # Export the whole conversation.
        if st.session_state.messages:
            transcript = "\n\n".join(
                f"## {m['role'].title()}\n\n{m['content']}"
                for m in st.session_state.messages
            )
            st.download_button(
                "Download full conversation (.md)",
                data=transcript,
                file_name=f"conversation-{datetime.now():%Y%m%d-%H%M}.md",
                mime="text/markdown",
            )
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    # --- Replay history ---------------------------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Chat input -------------------------------------------------------
    prompt = st.chat_input("Ask for a chapter, code, a schema - or say 'Save this as file.txt'")
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Feature 3: intercept save requests before calling the model.
    save_result = handle_possible_save(prompt)
    if save_result is not None:
        with st.chat_message("assistant"):
            fname = save_result["filename"]
            if not save_result["content"]:
                reply = (
                    f"I couldn't find any generated content to save as **{fname}** "
                    f"yet. Ask me to write it first, then say "
                    f"*\"Save this as {fname}\"*."
                )
                st.markdown(reply)
            else:
                if save_result.get("path"):
                    reply = f"Saved **{fname}** to:\n\n`{save_result['path']}`"
                else:
                    reply = (
                        f"Prepared **{fname}** for download (local saving isn't "
                        f"available in this environment)."
                    )
                st.markdown(reply)
                st.download_button(
                    f"Download {fname}",
                    data=save_result["content"],
                    file_name=fname,
                )
        st.session_state.messages.append({"role": "assistant", "content": reply})
        return

    if not api_key:
        with st.chat_message("assistant"):
            st.error("Please add your DeepSeek API key in the sidebar to continue.")
        return

    # --- Call the model ---------------------------------------------------
    client = get_client(api_key)
    api_messages = build_api_messages(system_prompt)  # persona rebuilt live

    with st.chat_message("assistant"):
        try:
            answer_text = stream_response(
                client, model_id, api_messages, temperature, max_tokens
            )
        except Exception as exc:  # noqa: BLE001
            answer_text = f"Something went wrong calling DeepSeek:\n\n`{exc}`"
            st.error(answer_text)
        else:
            # Offer a quick download of this single response.
            if answer_text:
                st.download_button(
                    "Download this response",
                    data=answer_text,
                    file_name=f"response-{datetime.now():%Y%m%d-%H%M%S}.md",
                    mime="text/markdown",
                    key=f"dl-{len(st.session_state.messages)}",
                )

    st.session_state.messages.append({"role": "assistant", "content": answer_text})


if __name__ == "__main__":
    main()

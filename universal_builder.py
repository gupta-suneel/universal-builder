"""
universal_builder.py
=====================
A conversational studio for writing complex physics & math books (and SaaS
code) powered by the DeepSeek API. Runs locally on a Mac or as a private
cloud app on Streamlit Community Cloud.

CORE FEATURES
  1. DYNAMIC PERSONA SHIFTER   - switching persona applies on the next message;
     the system prompt is rebuilt fresh every call. Includes book-writing
     personas with physics-figure and math conventions baked in.
  2. DEEPSEEK REASONING CHAINS - the R1 reasoner's hidden chain-of-thought
     streams into a collapsible "AI Thinking Process" box.
  3. LOCAL FILE AUTO-SAVING    - "Save this chapter as kinematics.txt" writes to
     ~/Documents/ai-workspace/universal_book_builder/output/ (+ a download).
  4. PERSISTENT API KEY        - read from secrets / env / saved file; asked once.
  5. PASSWORD GATE             - set APP_PASSWORD to keep the cloud app private.

PHYSICS / MATH BOOK TOOLS
  6. BOOK STYLE GUIDE          - a persistent notation/level panel injected into
     every prompt so chapters stay consistent.
  7. LATEX MATH RENDERING      - equations in $...$ / $$...$$ render natively.
  8. FIGURE & MATH LAB         - runs the AI's matplotlib/SymPy code, renders the
     figure inline so you can VERIFY it, prints output, and exports PNG.
  9. CONTINUE WRITING          - one click to seamlessly extend a long chapter.
 10. MANUSCRIPT ASSEMBLER      - merge saved chapters into one downloadable book.

Run locally:   streamlit run universal_builder.py
"""

import io
import json
import os
import re
import contextlib
import traceback
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
CONFIG_FILE = APP_DIR / "config.json"

MODELS = {
    "DeepSeek V3 (deepseek-chat) - fast, general purpose": "deepseek-chat",
    "DeepSeek R1 (deepseek-reasoner) - deep step-by-step thinking": "deepseek-reasoner",
}

# ---------------------------------------------------------------------------
# 1. SHARED RULE BLOCKS  (the physics/math knowledge baked into the AI)
# ---------------------------------------------------------------------------

MATH_RULES = """
MATH & NOTATION RULES (always follow):
- Write ALL mathematics in LaTeX: inline as $...$ and displayed as $$...$$.
  These render in the app, so never write equations as plain text.
- Define every symbol the first time it appears, and keep notation identical
  across chapters. Obey the BOOK STYLE GUIDE if one is provided.
- Number displayed/important equations so they can be cross-referenced.
- State and check the units; keep every equation dimensionally consistent.
- Sanity-check results against limiting cases (small angle, large/zero mass,
  t->0 or t->infinity) and mention the check.
- When a derivation matters, ALSO provide a short, runnable SymPy snippet in a
  ```python code block that verifies the key result. The user can run it in the
  app's Figure & Math Lab to confirm it.
""".strip()

PHYSICS_FIGURE_RULES = """
PHYSICS FIGURE RULES (always follow when a figure is involved):
General accuracy
- Draw figures FROM THE REAL EQUATION OR DATA computed with numpy -- never
  sketch an approximate shape by eye.
- Preserve the meaning-carrying features: equilibria, turning points,
  asymptotes, intercepts, symmetry and limiting behaviour.
- Use ax.set_aspect('equal') for anything geometric (angles, circles, orbits)
  so circles stay circular and right angles stay right angles.
- Label every axis with units, add a legend for multiple curves, and annotate
  the key physical points directly.

Free-body diagrams & vectors
- Every force is an arrow starting on the body, pointing the correct way, with
  length roughly proportional to its magnitude, and labeled. Invent no forces.
- Draw vector components to scale and at true right angles. Show the coordinate
  axes and the chosen sign convention.

Fields
- Field lines never cross; equipotential / wavefront lines stay perpendicular
  to the field lines.

Pulleys (high-error area -- be strict)
- ONE continuous ideal rope has ONE tension: it is identical on both sides of a
  frictionless, massless pulley. Do not label the two sides T1 and T2.
- Mechanical advantage = the number of rope strands that actually support the
  load. A fixed pulley only redirects force (advantage 1); a movable pulley is
  held by 2 strands (force = W/2); a block-and-tackle with N supporting strands
  gives W/N. The drawing must literally show that many strands.
- The rope is TANGENT to the wheel and never routed through its centre.
  Supporting strands hang vertical and parallel.
- Distinguish fixed (anchored) from movable (rises with the load) pulleys. Show
  the wheel, a centre axle, and the mount/hook to a hatched support. Hang masses
  as blocks with weight mg down and tension T up.
- Know the canonical setups cold: the Atwood machine (two masses over one fixed
  pulley, equal tension throughout) and the single movable-pulley lift.

Verification
- Before finalising, check the invariants: one rope -> one tension; strand count
  matches the stated mechanical advantage; every rope tangent to its wheel;
  limiting cases look right.
""".strip()

FIGURE_EMISSION = """
HOW TO EMIT FIGURES (the app renders them automatically and inline):
- The app EXECUTES your matplotlib code and shows the finished image right where
  you place it. The reader sees a book with embedded figures -- never code.
- So whenever a figure belongs in the text, drop a SINGLE self-contained
  ```python code block at exactly that point in the writing.
- Begin every figure block with a caption comment on the first line, like:
      # FIGURE: Figure 2.1 - Potential energy well
- Compute everything from the real equations with numpy/matplotlib, set axis
  labels with units, a legend if needed, and ax.set_aspect('equal') for geometry.
- Do NOT write "see the figure below / run this code / as shown in the plot you
  can generate". Just place the block; it becomes the figure.
- One figure per code block. Make each block runnable on its own (do its own
  imports). plt.show() is optional and harmless.
- For a symbolic/numeric VERIFICATION (not a drawing), use a ```python block that
  prints its result; the app shows that result in a small collapsible note.
""".strip()

# ---------------------------------------------------------------------------
# 2. PERSONAS
# ---------------------------------------------------------------------------

PERSONAS = {
    "Physics & Math Author (chapters)": (
        "You are a master physics and mathematics author writing a rigorous, "
        "engaging multi-chapter book. You produce complete, polished, "
        "publication-ready chapters with clear structure, worked examples, and "
        "intuition alongside rigor. You weave figures directly into the writing "
        "wherever they aid understanding. Maintain continuity of voice, notation, "
        "and numbering across chapters.\n\n"
        + MATH_RULES + "\n\n" + PHYSICS_FIGURE_RULES + "\n\n" + FIGURE_EMISSION
    ),
    "Mechanics Diagram Engineer (FBD/pulleys/inclines)": (
        "You are a mechanics figure specialist. You produce correct free-body "
        "diagrams and mechanics schematics (pulleys, inclines, springs, levers) "
        "as clean matplotlib code that the app renders inline. You are meticulous "
        "about physical correctness.\n\n"
        + PHYSICS_FIGURE_RULES + "\n\n" + FIGURE_EMISSION + "\n\n" + MATH_RULES
    ),
    "Figure & Plot Coder (matplotlib)": (
        "You turn physics and math into accurate matplotlib figures that the app "
        "renders inline. You always compute curves from the real equation with "
        "numpy and set sensible axes, units, legends and aspect ratio.\n\n"
        + PHYSICS_FIGURE_RULES + "\n\n" + FIGURE_EMISSION + "\n\n" + MATH_RULES
    ),
    "Derivation Verifier (SymPy)": (
        "You verify mathematical derivations. For each claimed result, you give "
        "a clear step-by-step derivation AND a self-contained ```python SymPy "
        "snippet that proves it symbolically (and numerically where helpful), "
        "printing a clear PASS/representation at the end.\n\n"
        + MATH_RULES + "\n\n" + FIGURE_EMISSION
    ),
    "Problem Set Writer (with solutions)": (
        "You write physics/math problem sets with full, verified solutions. Each "
        "problem states given/find, the solution shows every step, and you "
        "include a SymPy or numpy check of the final answer in a ```python block. "
        "Vary difficulty and label it.\n\n"
        + MATH_RULES + "\n\n" + PHYSICS_FIGURE_RULES + "\n\n" + FIGURE_EMISSION
    ),
    "Senior Full-Stack Engineer (SaaS apps)": (
        "You are an elite full-stack software engineer. You design and write "
        "clean, production-grade, secure code for SaaS applications. Wrap all "
        "code in fenced code blocks with the correct language tag and include "
        "brief setup notes."
    ),
    "Technical Educator (clear explanations)": (
        "You are a patient technical educator. You explain complex topics in "
        "plain language with concrete examples and analogies, building from "
        "first principles.\n\n" + MATH_RULES
    ),
    "General Assistant": "You are a helpful, knowledgeable, and concise assistant.",
    "Custom (edit your own)": "",
}

CUSTOM_PERSONA_LABEL = "Custom (edit your own)"

# ---------------------------------------------------------------------------
# 3. SECRETS / CONFIG HELPERS
# ---------------------------------------------------------------------------

def get_secret(key: str, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default


def read_saved_key():
    try:
        if CONFIG_FILE.exists():
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8")).get("DEEPSEEK_API_KEY")
    except Exception:
        pass
    return None


def write_saved_key(api_key: str):
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps({"DEEPSEEK_API_KEY": api_key}), encoding="utf-8")
        return True
    except Exception:
        return False


def resolve_api_key():
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
# 4. PASSWORD GATE
# ---------------------------------------------------------------------------

def check_password() -> bool:
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
# 5. FILE OPERATIONS
# ---------------------------------------------------------------------------

SAVE_PATTERN = re.compile(
    r"save\s+(?:this|it|the)?\s*"
    r"(?:chapter|file|code|content|page|document|figure)?\s*"
    r"as\s+[\"']?([\w\-. ]+\.[A-Za-z0-9]+)[\"']?",
    re.IGNORECASE,
)


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def detect_save_request(text: str):
    match = SAVE_PATTERN.search(text or "")
    return match.group(1).strip() if match else None


def extract_saveable_content(message_text: str) -> str:
    if not message_text:
        return ""
    code_blocks = re.findall(r"```[\w+\-]*\n(.*?)```", message_text, re.DOTALL)
    if code_blocks:
        return "\n\n".join(b.rstrip() for b in code_blocks).strip() + "\n"
    return message_text.strip() + "\n"


def save_file(filename: str, content: str):
    safe_name = Path(filename).name
    try:
        out_dir = ensure_output_dir()
        full_path = out_dir / safe_name
        full_path.write_text(content, encoding="utf-8")
        return full_path, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def handle_possible_save(user_text: str):
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


def assemble_manuscript():
    """Concatenate saved .md/.txt chapters in OUTPUT_DIR into one manuscript."""
    if not OUTPUT_DIR.exists():
        return None, "No saved chapters found yet."
    parts = sorted(
        [p for p in OUTPUT_DIR.iterdir()
         if p.suffix.lower() in (".md", ".txt") and p.name != "manuscript_assembled.md"]
    )
    if not parts:
        return None, "No .md or .txt chapters found in the output folder yet."
    pieces = []
    for p in parts:
        try:
            pieces.append(f"\n\n# {p.stem}\n\n" + p.read_text(encoding="utf-8"))
        except Exception:
            continue
    manuscript = "\n".join(pieces).strip() + "\n"
    out = OUTPUT_DIR / "manuscript_assembled.md"
    try:
        out.write_text(manuscript, encoding="utf-8")
    except Exception:
        out = None
    return manuscript, (str(out) if out else None)


# ---------------------------------------------------------------------------
# 6. INHERENT FIGURE RENDERING  (auto-run figure code, embed images inline)
# ---------------------------------------------------------------------------

# Split a reply into ordered ("text", str) and ("code", lang, body) segments.
_CODE_SPLIT = re.compile(r"```([\w+\-]*)\s*\n(.*?)```", re.DOTALL)


def split_segments(text: str):
    segments = []
    pos = 0
    for m in _CODE_SPLIT.finditer(text or ""):
        if m.start() > pos:
            segments.append(("text", text[pos:m.start()]))
        segments.append(("code", (m.group(1) or "").lower().strip(), m.group(2)))
        pos = m.end()
    if pos < len(text or ""):
        segments.append(("text", text[pos:]))
    return segments


def _caption_from_code(code: str):
    """Pull a '# FIGURE: ...' caption from the first lines of a code block."""
    for line in code.splitlines()[:4]:
        m = re.match(r"\s*#\s*FIGURE:\s*(.+)", line, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def run_python_code(code: str):
    """
    Execute code with matplotlib(Agg)/numpy/sympy preloaded.
    Returns dict: {png: bytes|None, stdout: str, error: str|None}.
    Results are cached per code-hash so reruns are instant.
    NOTE: runs code in-process; intended for your own / the model's code on a
    private, password-gated app.
    """
    import hashlib
    cache = st.session_state.setdefault("fig_cache", {})
    key = hashlib.md5(code.encode("utf-8")).hexdigest()
    if key in cache:
        return cache[key]

    out = io.StringIO()
    result = {"png": None, "stdout": "", "error": None}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        try:
            import sympy as sp
        except Exception:
            sp = None
        plt.close("all")
        ns = {"plt": plt, "matplotlib": matplotlib, "np": np, "numpy": np,
              "sp": sp, "sympy": sp, "__name__": "__main__"}
        with contextlib.redirect_stdout(out):
            exec(code, ns)  # noqa: S102 - intentional, user-controlled content
        result["stdout"] = out.getvalue()
        fignums = plt.get_fignums()
        if fignums:
            buf = io.BytesIO()
            plt.figure(fignums[-1]).savefig(buf, format="png", dpi=200,
                                             bbox_inches="tight")
            result["png"] = buf.getvalue()
        plt.close("all")
    except Exception:
        result["stdout"] = out.getvalue()
        result["error"] = traceback.format_exc()
    cache[key] = result
    return result


def render_book_message(text, show_code, msg_index=0, client=None, model_id="deepseek-chat"):
    """Render a reply as a book: prose as markdown, figure code as live images,
    verification code as a small collapsible note. Failed figures get a one-click
    'Redraw this figure' repair button."""
    for bi, seg in enumerate(split_segments(text)):
        if seg[0] == "text":
            if seg[1].strip():
                st.markdown(seg[1])
            continue
        _, lang, body = seg
        uid = f"{msg_index}-{bi}"
        # Non-python code (sql, js, ...) -> show as a normal code block.
        if lang not in ("", "python", "py"):
            st.code(body, language=lang or None)
            continue
        # Python -> execute. Figure -> image; print-only -> note; else -> code.
        res = run_python_code(body)
        if res["png"]:
            st.image(res["png"], caption=_caption_from_code(body), use_container_width=True)
            if show_code:
                with st.expander("Figure code"):
                    st.code(body, language="python")
            st.download_button("Download figure (PNG)", data=res["png"],
                               file_name="figure.png", mime="image/png",
                               key=f"png-{uid}")
        elif res["error"]:
            cap = _caption_from_code(body) or "this figure"
            st.warning(f"A figure could not be rendered ({cap}).")
            if client is not None:
                if st.button("Redraw this figure", key=f"redraw-{uid}"):
                    with st.spinner("Redrawing the figure..."):
                        fixed = repair_figure(client, model_id, body, res["error"])
                    if fixed:
                        msgs = st.session_state.messages
                        msgs[msg_index]["content"] = msgs[msg_index]["content"].replace(
                            body, fixed + "\n", 1)
                        st.rerun()
                    else:
                        st.warning("Couldn't auto-fix it - try asking in the chat to "
                                   "redraw it differently.")
            with st.expander("Show figure code & error"):
                st.code(body, language="python")
                st.code(res["error"])
        elif res["stdout"].strip():
            with st.expander("Verification / computed result"):
                st.text(res["stdout"])
                if show_code:
                    st.code(body, language="python")
        elif show_code:
            st.code(body, language="python")


# ---------------------------------------------------------------------------
# 7. DEEPSEEK CLIENT & STREAMING
# ---------------------------------------------------------------------------

def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def repair_figure(client, model_id, code, error):
    """Ask the model to fix one broken figure block. Returns corrected code or None."""
    sys = ("You fix broken matplotlib figure code. Return ONLY one corrected, "
           "self-contained ```python code block that runs without error and draws "
           "the intended figure. Keep any '# FIGURE:' caption comment. Compute from "
           "real equations with numpy. No prose, no explanation.")
    user = (f"This figure code failed:\n\n```python\n{code}\n```\n\n"
            f"Error / traceback:\n{error}\n\nReturn the corrected code block only.")
    kwargs = {"model": model_id,
              "messages": [{"role": "system", "content": sys},
                           {"role": "user", "content": user}],
              "max_tokens": 2000}
    if model_id == "deepseek-chat":
        kwargs["temperature"] = 0.2
    try:
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
    except Exception:
        return None
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return (m.group(1).strip() if m else text.strip()) or None


def build_api_messages(system_prompt: str):
    """Rebuild the message list fresh every call so the persona/style guide are
    always current (the persona-shifter fix)."""
    system_msg = {"role": "system", "content": system_prompt}
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant")
    ]
    return [system_msg] + history


def stream_response(client, model_id, messages, temperature, max_tokens):
    reasoning_box = st.empty()
    answer_box = st.empty()
    reasoning_text = ""
    answer_text = ""

    kwargs = {"model": model_id, "messages": messages, "stream": True,
              "max_tokens": int(max_tokens)}
    if model_id == "deepseek-chat":
        kwargs["temperature"] = float(temperature)

    stream = client.chat.completions.create(**kwargs)
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
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
# 8. STREAMLIT APP
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Universal Builder", page_icon=":bricks:", layout="wide")

    if not check_password():
        return

    st.title("Universal Builder")
    st.caption("A studio for physics & math books - powered by DeepSeek.")

    # Session state.
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("book_guide", "")
    st.session_state.setdefault("pending_prompt", None)

    api_key, key_source = resolve_api_key()

    # ----- Sidebar -------------------------------------------------------
    with st.sidebar:
        st.header("Controls")

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
        if persona_name == CUSTOM_PERSONA_LABEL:
            system_prompt = st.text_area(
                "Custom system prompt",
                value=st.session_state.get("custom_prompt",
                                           "You are a helpful expert assistant."),
                height=120,
            )
            st.session_state.custom_prompt = system_prompt
        else:
            system_prompt = PERSONAS[persona_name]
        st.caption(f"Active persona: **{persona_name}**")

        with st.expander("Book Style Guide (kept across chapters)"):
            st.session_state.book_guide = st.text_area(
                "Notation, audience, level & conventions",
                value=st.session_state.book_guide,
                height=140,
                placeholder=("e.g. Undergraduate level. Use SI units. Vectors in "
                             "bold, e.g. F. Denote acceleration a, tension T. "
                             "Number equations per chapter (2.1, 2.2...)."),
                help="This is injected into every message so chapters stay consistent.",
            )

        st.divider()
        st.subheader("Generation settings")
        temperature = st.slider("Creativity (temperature)", 0.0, 1.5, 0.4, 0.1,
                                help="~0.3 for precise math, ~1.0+ for prose. "
                                     "Ignored by the R1 reasoner.")
        max_tokens = st.slider("Max response length (tokens)", 256, 8192, 4096, 256)

        st.divider()
        st.subheader("Book tools")
        st.session_state.show_code = st.checkbox(
            "Show the code behind figures", value=st.session_state.get("show_code", False),
            help="Off by default so it reads like a finished book. Figures render "
                 "automatically either way.")
        if st.button("Assemble saved chapters -> manuscript"):
            manuscript, where = assemble_manuscript()
            if manuscript is None:
                st.warning(where)
            else:
                st.success("Assembled." + (f" Saved to {where}" if where else ""))
                st.download_button("Download manuscript.md", data=manuscript,
                                   file_name="manuscript_assembled.md",
                                   mime="text/markdown")
        if st.session_state.messages and st.button("Continue writing (extend last reply)"):
            st.session_state.pending_prompt = (
                "Continue exactly where you left off, seamlessly, without "
                "repeating anything you already wrote."
            )
            st.rerun()

        if st.session_state.messages:
            transcript = "\n\n".join(
                f"## {m['role'].title()}\n\n{m['content']}" for m in st.session_state.messages
            )
            st.download_button("Download conversation (.md)", data=transcript,
                               file_name=f"conversation-{datetime.now():%Y%m%d-%H%M}.md",
                               mime="text/markdown")
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    # ----- Replay history (figures rendered inline, like a book) ---------
    show_code = st.session_state.get("show_code", False)
    render_client = get_client(api_key) if api_key else None
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_book_message(msg["content"], show_code, idx,
                                    render_client, model_id)
            else:
                st.markdown(msg["content"])

    # ----- Input (chat box OR a queued action) ---------------------------
    prompt = st.chat_input("Ask for a chapter, a figure, a derivation - or 'Save this as ch1.md'")
    if not prompt and st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Save interception.
    save_result = handle_possible_save(prompt)
    if save_result is not None:
        with st.chat_message("assistant"):
            fname = save_result["filename"]
            if not save_result["content"]:
                reply = (f"I couldn't find content to save as **{fname}** yet. "
                         f"Generate it first, then say *\"Save this as {fname}\"*.")
                st.markdown(reply)
            else:
                if save_result.get("path"):
                    reply = f"Saved **{fname}** to:\n\n`{save_result['path']}`"
                else:
                    reply = f"Prepared **{fname}** for download (local saving unavailable here)."
                st.markdown(reply)
                st.download_button(f"Download {fname}", data=save_result["content"],
                                   file_name=fname)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        return

    if not api_key:
        with st.chat_message("assistant"):
            st.error("Please add your DeepSeek API key in the sidebar to continue.")
        return

    # Compose the full system prompt: persona + the persistent style guide.
    full_system = system_prompt
    if st.session_state.book_guide.strip():
        full_system += "\n\nBOOK STYLE GUIDE (obey strictly):\n" + st.session_state.book_guide.strip()

    client = get_client(api_key)
    api_messages = build_api_messages(full_system)

    failed = False
    with st.chat_message("assistant"):
        try:
            # Stream the draft live (shows progress); figures get rendered on the
            # rerun below so they appear inline within the finished text.
            answer_text = stream_response(client, model_id, api_messages, temperature, max_tokens)
        except Exception as exc:  # noqa: BLE001
            answer_text = f"Something went wrong calling DeepSeek:\n\n`{exc}`"
            st.error(answer_text)
            failed = True

    st.session_state.messages.append({"role": "assistant", "content": answer_text})
    # Re-run so the reply is re-rendered as a book with figures executed inline.
    if not failed:
        st.rerun()


if __name__ == "__main__":
    main()

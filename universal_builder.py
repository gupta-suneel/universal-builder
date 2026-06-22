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
  2. DEEPSEEK REASONING CHAINS - in Thinking mode, the model's hidden
     chain-of-thought streams into a collapsible "AI Thinking Process" box.
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
import time
import contextlib
import traceback
from datetime import datetime
from pathlib import Path

import streamlit as st
from openai import OpenAI

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

# ---------------------------------------------------------------------------
# 0. CONSTANTS & PATHS
# ---------------------------------------------------------------------------

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

APP_DIR = Path.home() / "Documents" / "ai-workspace" / "universal_book_builder"
OUTPUT_DIR = APP_DIR / "output"
CONFIG_FILE = APP_DIR / "config.json"

# DeepSeek's current API models (the older deepseek-chat / deepseek-reasoner
# names are legacy aliases scheduled for retirement on 2026/07/24). Thinking
# mode is now a parameter, not a separate model.
MODELS = {
    "DeepSeek V4 Flash - fast & economical": "deepseek-v4-flash",
    "DeepSeek V4 Pro - frontier reasoning (best for hard math)": "deepseek-v4-pro",
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

USE THE VETTED TOOLKIT FIRST (this is critical for correct geometry):
- A pre-tested physics-figure library is preloaded as `phys`. Its drawing
  functions have CORRECT geometry baked in (ropes tangent to pulleys, equal
  tension, force arrows proportional, true angles). PREFER these over hand-drawn
  matplotlib whenever one fits -- hand-drawn schematics are usually wrong.
- Available helpers (call with a matplotlib Axes `ax`):
{catalog}
- Pattern: create `fig, ax = plt.subplots()`, call the helper, done. Example:
      # FIGURE: Figure 3.2 - Atwood machine
      import matplotlib.pyplot as plt
      import physlib as phys
      fig, ax = plt.subplots()
      phys.atwood(ax, m1=3, m2=5)
- If you must customise, you may pass functions (e.g. phys.potential_well(ax,
  V=lambda x: 0.25*x**4 - x**2)) or draw extra matplotlib on the same `ax`.
- Only hand-roll matplotlib when NO helper fits; then compute from real
  equations with numpy and use ax.set_aspect('equal') for geometric figures.

GENERAL RULES:
- Do NOT write "see the figure below / run this code". Just place the block.
- One figure per code block; each block runnable on its own (do its own imports).
- For a symbolic/numeric VERIFICATION (not a drawing), use a ```python block that
  prints its result; the app shows it in a small collapsible note.
""".strip()

try:
    import physlib as _physlib
    FIGURE_EMISSION = FIGURE_EMISSION.replace("{catalog}", _physlib.CATALOG)
except Exception:
    FIGURE_EMISSION = FIGURE_EMISSION.replace("{catalog}", "(toolkit catalog unavailable)")

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
# 3b. CLOUD MEMORY  (auto-save / auto-restore via Supabase)
# ---------------------------------------------------------------------------
# Streamlit Cloud is stateless, so we persist the conversation to a free
# Supabase table. Configure SUPABASE_URL and SUPABASE_KEY (service_role) in
# the app's Secrets. The table is created once with:
#   create table if not exists book_sessions (
#     key text primary key, data jsonb, updated_at timestamptz default now());

SUPA_TABLE = "book_sessions"


def get_store():
    """Return {'url','key'} if cloud memory is configured, else None."""
    url = get_secret("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    key = get_secret("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY")
    if url and key and requests is not None:
        return {"url": str(url).rstrip("/"), "key": str(key)}
    return None


def _supa_headers(store):
    return {"apikey": store["key"], "Authorization": f"Bearer {store['key']}",
            "Content-Type": "application/json"}


def load_state(store, project):
    """Fetch a saved project's data dict from Supabase, or None."""
    try:
        r = requests.get(f"{store['url']}/rest/v1/{SUPA_TABLE}",
                         params={"key": f"eq.{project}", "select": "data"},
                         headers=_supa_headers(store), timeout=12)
        if r.status_code == 200:
            rows = r.json()
            if rows:
                return rows[0].get("data")
    except Exception:
        return None
    return None


def save_state(store, project, data):
    """Upsert a project's data dict to Supabase. Returns True on success."""
    try:
        r = requests.post(
            f"{store['url']}/rest/v1/{SUPA_TABLE}",
            params={"on_conflict": "key"},
            headers={**_supa_headers(store),
                     "Prefer": "resolution=merge-duplicates,return=minimal"},
            json=[{"key": project, "data": data}], timeout=12)
        return r.status_code in (200, 201, 204)
    except Exception:
        return False


# ---- Local persistence + timestamped backups (dependable when run locally) --
PROJECTS_DIR = APP_DIR / "projects"
BACKUPS_DIR = APP_DIR / "backups"


def _safe_name(name):
    return re.sub(r"[^\w\- ]", "_", name or "").strip() or "My Book"


def save_local(project, data, keep_backups=40):
    """Write the project JSON to disk plus a timestamped backup. Best-effort."""
    try:
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        (PROJECTS_DIR / f"{_safe_name(project)}.json").write_text(
            json.dumps(data), encoding="utf-8")
        BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        (BACKUPS_DIR / f"{_safe_name(project)}__{ts}.json").write_text(
            json.dumps(data), encoding="utf-8")
        backups = list_backups(project)
        for old in backups[keep_backups:]:
            try:
                old.unlink()
            except Exception:
                pass
        return True
    except Exception:
        return False


def load_local(project):
    try:
        p = PROJECTS_DIR / f"{_safe_name(project)}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def list_backups(project):
    """Most-recent-first list of backup files for a project."""
    if not BACKUPS_DIR.exists():
        return []
    prefix = _safe_name(project) + "__"
    return sorted([p for p in BACKUPS_DIR.iterdir() if p.name.startswith(prefix)],
                  reverse=True)


def persist_now(store, project):
    """Save the current session to local disk AND cloud (if configured)."""
    data = {
        "messages": st.session_state.get("messages", []),
        "book_guide": st.session_state.get("book_guide", ""),
        "custom_prompt": st.session_state.get("custom_prompt", ""),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_local(project, data)          # always keep a local copy + backup
    if store:
        save_state(store, project, data)


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
        try:
            import physlib as phys  # vetted physics-figure toolkit
        except Exception:
            phys = None
        plt.close("all")
        ns = {"plt": plt, "matplotlib": matplotlib, "np": np, "numpy": np,
              "sp": sp, "sympy": sp, "phys": phys, "__name__": "__main__"}
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


def render_book_message(text, show_code, msg_index=0, client=None, model_id="deepseek-v4-flash"):
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
              "max_tokens": 2000, "temperature": 0.2,
              "extra_body": {"thinking": {"type": "disabled"}}}
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


def stream_response(client, model_id, messages, temperature, max_tokens, thinking):
    reasoning_box = st.empty()
    answer_box = st.empty()
    reasoning_text = ""
    answer_text = ""

    kwargs = {"model": model_id, "messages": messages, "stream": True,
              "max_tokens": int(max_tokens)}
    if thinking:
        # Thinking mode reasons step-by-step and returns reasoning_content.
        # It ignores temperature, so we don't send it.
        kwargs["reasoning_effort"] = "high"
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    else:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        kwargs["temperature"] = float(temperature)

    # Establish the stream with a few retries for transient network/API hiccups.
    stream = None
    last_err = None
    for attempt in range(3):
        try:
            stream = client.chat.completions.create(**kwargs)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < 2:
                st.caption(f"Connection hiccup, retrying ({attempt + 1}/2)...")
                time.sleep(1.5 * (attempt + 1))
    if stream is None:
        raise last_err

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
    st.session_state.setdefault("project", "My Book")
    st.session_state.setdefault("loaded_project", None)

    api_key, key_source = resolve_api_key()
    if not api_key and st.session_state.get("api_key_session"):
        api_key, key_source = st.session_state["api_key_session"], "this session"
    store = get_store()

    # ----- Sidebar -------------------------------------------------------
    with st.sidebar:
        st.header("Controls")

        if api_key:
            st.success(f"API key loaded (from {key_source}).")
            with st.expander("Change API key"):
                with st.form("change_key_form", clear_on_submit=True):
                    new_key = st.text_input("New DeepSeek API key", type="password")
                    if st.form_submit_button("Update & save") and new_key:
                        write_saved_key(new_key)
                        st.session_state["api_key_session"] = new_key
                        st.rerun()
        else:
            st.warning("No API key found yet.")
            with st.form("api_key_form"):
                entered = st.text_input("DeepSeek API key", type="password",
                                        help="Get one at platform.deepseek.com")
                remember = st.checkbox("Remember this key on this computer", value=True)
                submitted = st.form_submit_button("Save & use key")
            if submitted and entered:
                st.session_state["api_key_session"] = entered
                if remember:
                    write_saved_key(entered)
                st.rerun()
            elif submitted and not entered:
                st.error("Please paste your key first.")

        st.divider()
        st.subheader("Memory & backups")
        new_project = st.text_input("Project (book) name",
                                    value=st.session_state.project)
        if new_project and new_project != st.session_state.project:
            st.session_state.project = new_project
            st.session_state.loaded_project = None
            st.rerun()
        if store:
            st.success("Auto-save: cloud + local disk")
        else:
            st.caption("Auto-save: local disk (every reply). Add SUPABASE keys in "
                       "Secrets to also sync to the cloud.")
        if st.button("Reload this project"):
            st.session_state.loaded_project = None
            st.rerun()
        _bk = list_backups(st.session_state.project)
        if _bk:
            with st.expander(f"Restore an earlier version ({len(_bk)} backups)"):
                _labels = [b.stem.split("__", 1)[-1] for b in _bk]
                _pick = st.selectbox("Saved versions (newest first)", _labels,
                                     key="backup_pick")
                if st.button("Restore this version"):
                    try:
                        _d = json.loads(_bk[_labels.index(_pick)].read_text(encoding="utf-8"))
                        st.session_state.messages = _d.get("messages", [])
                        st.session_state.book_guide = _d.get("book_guide",
                                                             st.session_state.book_guide)
                        st.rerun()
                    except Exception:
                        st.warning("Could not read that backup.")

        st.divider()
        model_label = st.selectbox("Model", list(MODELS.keys()), index=0)
        model_id = MODELS[model_label]
        thinking = st.checkbox(
            "Thinking mode (deeper reasoning; shows thought process)",
            value=True,
            help="On: the model reasons step-by-step before answering - best for "
                 "hard math and derivations, and it shows its thinking. The "
                 "creativity slider is ignored in this mode. Off: faster, and the "
                 "creativity slider applies.")

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
                                     "Ignored while Thinking mode is on.")
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

        # Export the finished book (figures + math included).
        if st.session_state.messages:
            with st.expander("Export finished book"):
                if st.button("Build Word (.docx)"):
                    try:
                        import bookexport
                        ensure_output_dir()
                        out = OUTPUT_DIR / f"{_safe_name(st.session_state.project)}.docx"
                        bookexport.build_docx(st.session_state.messages,
                                              st.session_state.project, str(out))
                        st.download_button("Download .docx", data=out.read_bytes(),
                                           file_name=out.name,
                                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                        st.caption(f"Also saved to {out}")
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Word export failed: {exc}")
                if st.button("Build printable HTML (then Print -> Save as PDF)"):
                    try:
                        import bookexport
                        html = bookexport.build_html(st.session_state.messages,
                                                     st.session_state.project)
                        st.download_button("Download .html", data=html,
                                           file_name=f"{_safe_name(st.session_state.project)}.html",
                                           mime="text/html")
                        st.caption("Open it in your browser, then Print -> Save as PDF.")
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"HTML export failed: {exc}")

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

    # ----- Restore memory (cloud if configured, else local disk) ---------
    project = st.session_state.project
    if st.session_state.loaded_project != project:
        data = load_state(store, project) if store else None
        if data is None:
            data = load_local(project)          # local disk fallback
        st.session_state.messages = (data or {}).get("messages", [])
        if data and data.get("book_guide"):
            st.session_state.book_guide = data["book_guide"]
        if data and data.get("custom_prompt"):
            st.session_state.custom_prompt = data["custom_prompt"]
        st.session_state.loaded_project = project

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
        persist_now(store, project)
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
            answer_text = stream_response(client, model_id, api_messages,
                                          temperature, max_tokens, thinking)
        except Exception as exc:  # noqa: BLE001
            answer_text = f"Something went wrong calling DeepSeek:\n\n`{exc}`"
            st.error(answer_text)
            failed = True

    st.session_state.messages.append({"role": "assistant", "content": answer_text})
    persist_now(store, project)  # auto-save to cloud memory
    # Re-run so the reply is re-rendered as a book with figures executed inline.
    if not failed:
        st.rerun()


if __name__ == "__main__":
    main()

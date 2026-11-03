import re
import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
import fitz

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DB_DIR = ROOT / "database"
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "history.db"

st.set_page_config(page_title="RhetorAI", page_icon="📚", layout="wide")

MOVE_NAMES = {
    "Move 1": "Move 1 — Establishing a Territory",
    "Move 2": "Move 2 — Establishing a Niche",
    "Move 3": "Move 3 — Occupying the Niche",
}

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                sentence_count INTEGER NOT NULL,
                move1_count INTEGER NOT NULL,
                move2_count INTEGER NOT NULL,
                move3_count INTEGER NOT NULL
            )
        """)

def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def demo_classify(sentence):
    """Rule-based prototype only; replace with a trained model for research use."""
    text = sentence.lower()
    move3 = [
        r"\bthis study\b", r"\bthis paper\b", r"\bthis research\b",
        r"\bwe propose\b", r"\bwe present\b", r"\bour approach\b",
        r"\bour work\b", r"\bthe aim of this\b", r"\bthe purpose of this\b"
    ]
    move2 = [
        r"\bhowever\b", r"\bnevertheless\b", r"\bdespite\b",
        r"\blittle is known\b", r"\bremains unclear\b", r"\bresearch gap\b",
        r"\blimited research\b", r"\bfew studies\b", r"\bhas not been\b",
        r"\bhave not been\b", r"\bthere is a lack\b", r"\bnot yet\b"
    ]
    if any(re.search(p, text) for p in move3):
        return MOVE_NAMES["Move 3"]
    if any(re.search(p, text) for p in move2):
        return MOVE_NAMES["Move 2"]
    return MOVE_NAMES["Move 1"]

def extract_pdf_text(uploaded_file):
    with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)

def save_history(df):
    counts = df["Predicted Move"].value_counts()
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO analyses (created_at, sentence_count, move1_count, move2_count, move3_count) VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(timespec="seconds"),
                len(df),
                int(counts.get(MOVE_NAMES["Move 1"], 0)),
                int(counts.get(MOVE_NAMES["Move 2"], 0)),
                int(counts.get(MOVE_NAMES["Move 3"], 0)),
            ),
        )

init_db()

st.title("📚 RhetorAI")
st.caption("Automated rhetorical move analysis in computer science research introductions")
st.warning("Prototype status: predictions currently use simple rules, not a trained Transformer. Do not treat these predictions as validated research results.")

page = st.sidebar.radio(
    "Navigation",
    ["Home Dashboard", "Research Paper Analyzer", "Analysis History", "Model & Research Info"]
)

if page == "Home Dashboard":
    st.header("Home Dashboard")
    st.write("RhetorAI aims to classify Introduction sentences according to the CARS (Create a Research Space) model.")
    a, b, c = st.columns(3)
    a.metric("Rhetorical categories", "3")
    b.metric("Input formats", "Text + PDF")
    c.metric("Current classifier", "Rule-based demo")
    st.subheader("The CARS moves")
    col1, col2, col3 = st.columns(3)
    col1.markdown("### 🔵 Move 1\n**Establishing a Territory**\n\nIntroduce the field, topic, and its importance.")
    col2.markdown("### 🟠 Move 2\n**Establishing a Niche**\n\nIdentify a gap, limitation, or unresolved issue.")
    col3.markdown("### 🟢 Move 3\n**Occupying the Niche**\n\nState the study's purpose, approach, or contribution.")
    st.info("Next research milestone: prepare an annotated dataset, train a baseline model, then fine-tune and evaluate a Transformer.")

elif page == "Research Paper Analyzer":
    st.header("Research Paper Analyzer")
    input_mode = st.radio("Input method", ["Paste Introduction text", "Upload PDF"], horizontal=True)
    text = ""
    if input_mode == "Paste Introduction text":
        text = st.text_area("Paste the Introduction section", height=260, placeholder="Paste the research paper's Introduction here...")
    else:
        uploaded = st.file_uploader("Upload a text-based PDF", type=["pdf"])
        if uploaded:
            try:
                extracted = extract_pdf_text(uploaded)
                st.success("PDF text extracted. Review the text and keep only the Introduction section.")
                text = st.text_area("Extracted text (editable)", value=extracted, height=260)
            except Exception as exc:
                st.error(f"Could not read PDF: {exc}")
    if st.button("Analyze Introduction", type="primary"):
        sentences = split_sentences(text)
        if not sentences:
            st.warning("Please provide readable Introduction text.")
        else:
            result = pd.DataFrame({
                "Sentence": sentences,
                "Predicted Move": [demo_classify(s) for s in sentences],
                "Classifier": ["Rule-based demonstration"] * len(sentences)
            })
            st.session_state["analysis_results"] = result
            save_history(result)
    if "analysis_results" in st.session_state:
        result = st.session_state["analysis_results"]
        st.subheader("Sentence-level results")
        counts = result["Predicted Move"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total sentences", len(result))
        c2.metric("Move 1", int(counts.get(MOVE_NAMES["Move 1"], 0)))
        c3.metric("Move 2", int(counts.get(MOVE_NAMES["Move 2"], 0)))
        c4.metric("Move 3", int(counts.get(MOVE_NAMES["Move 3"], 0)))
        st.dataframe(result, use_container_width=True, hide_index=True)
        chart_df = counts.rename_axis("Rhetorical Move").reset_index(name="Sentence Count")
        fig = px.bar(chart_df, x="Rhetorical Move", y="Sentence Count", title="Rhetorical Move Distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            "Download CSV results",
            result.to_csv(index=False).encode("utf-8"),
            file_name="rhetorai_analysis.csv",
            mime="text/csv"
        )

elif page == "Analysis History":
    st.header("Analysis History")
    with sqlite3.connect(DB_PATH) as con:
        history = pd.read_sql_query("SELECT * FROM analyses ORDER BY id DESC", con)
    if history.empty:
        st.info("No analyses have been saved yet. Analyze some text first.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)
        st.download_button("Export history CSV", history.to_csv(index=False).encode("utf-8"),
                           file_name="rhetorai_history.csv", mime="text/csv")

else:
    st.header("Model and Research Information")
    st.markdown("""
    **Research objective:** Investigate automatic classification of rhetorical moves in computer science research paper introductions.

    **CARS categories**
    - **Move 1 — Establishing a Territory:** Introduces the topic and its importance.
    - **Move 2 — Establishing a Niche:** Identifies a gap, limitation, or unresolved problem.
    - **Move 3 — Occupying the Niche:** Presents the current study's aim, approach, or contribution.

    **Planned modelling workflow**
    1. Prepare a labelled sentence dataset.
    2. Split data by paper into train, validation, and test sets.
    3. Train a TF-IDF + Logistic Regression baseline.
    4. Fine-tune a Transformer such as SciBERT.
    5. Report accuracy, macro precision/recall/F1, confusion matrix, and error analysis.

    **Important limitation:** The current app uses simple rules. The model training scripts are starter templates and require a properly labelled dataset before meaningful training or evaluation.
    """)

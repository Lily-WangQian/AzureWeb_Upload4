from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import re
import string
import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from PyPDF2 import PdfReader
import yake  # NEW: contextual keyword extraction

app = Flask(__name__)

# ------------ Config ------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {".pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ------------ Load standards ------------
# CSV must contain (or be mappable to):
#   Standard | Publication Date | TFIDF Keywords | Contextual Keywords
df = pd.read_csv("standards_keywords.csv")
df.columns = df.columns.str.strip()

# normalize possible variants
if "Standards" in df.columns and "Standard" not in df.columns:
    df.rename(columns={"Standards": "Standard"}, inplace=True)
for c in list(df.columns):
    if c.lower().startswith("publication") and "Publication Date" not in df.columns:
        df.rename(columns={c: "Publication Date"}, inplace=True)

EXPECTED = ["Standard", "Publication Date", "TFIDF Keywords", "Contextual Keywords"]
for col in EXPECTED:
    if col not in df.columns:
        df[col] = ""

standards = (
    df["Standard"].dropna().astype(str).str.strip().sort_values().unique().tolist()
)

# ------------ Stopwords & cleaning ------------
CUSTOM_STOPWORDS = set([
    'shall','among','best','would','like','see','needs','•','their','to','“should”','‘should’',
    'requires','“shall”','within','may','lot','etc','b','with','without','pdfs','shows','tells',
    'e','g','also','always','however','go','–','by','for','that','and','or','0c','meet','includes',
    'could','example','examples','chapter','an','a','on','in','as','box','additionally','particularly',
    'thereafter','please','the','The','there','has','to','have','this','welcome','website','appendix','‘can’',
    'we','re',"we’re",'we’re','we','re','should','be','com','rbc','at','from','ceo','appendices',
    'endnotes','volunteerismappendices','is','ii','of','our'
])

def remove_stopwords(text: str) -> str:
    """Lowercase, remove punctuation, drop stopwords/digits."""
    if not text:
        return ""
    # replace punctuation with spaces
    sentence = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    words = sentence.split()
    filtered = []
    for word in words:
        w = word.lower()
        if not w.isdigit() and w not in CUSTOM_STOPWORDS:
            filtered.append(w)
    return " ".join(filtered)

# ------------ TF-IDF bigram keywords ------------
def extract_tfidf_keywords(text: str, top_n: int = 5) -> list[str]:
    """
    Compute top-N bigram TF-IDF keywords from cleaned text.
    """
    if not text or not text.strip():
        return []
    vectorizer = TfidfVectorizer(ngram_range=(2, 2))
    x = vectorizer.fit_transform([text])
    keywords_df = pd.DataFrame(
        x.toarray(),
        columns=vectorizer.get_feature_names_out()
    ).transpose()
    keywords = (
        keywords_df.sort_values(by=0, ascending=False)
        .head(top_n)
        .index
        .tolist()
    )
    return keywords

# ------------ Contextual keywords with YAKE ------------
def extract_contextual_keywords(text: str, top_n: int = 5) -> list[str]:
    """
    Extract top-N bigram contextual keywords from text using YAKE.
    """
    if not text or not text.strip():
        return []
    kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=top_n)
    keywords = [kw for kw, score in kw_extractor.extract_keywords(text)]
    return keywords

# ------------ PDF helpers ------------
def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT

def extract_text_from_pdf(path: str, max_chars: int = 2000) -> str:
    """
    Quick extraction for preview (first ~max_chars characters).
    Uses pdfplumber.
    """
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            parts.append(txt)
            if sum(len(p) for p in parts) >= max_chars:
                break
    raw = "\n".join(parts)
    clean = re.sub(r"\s+", " ", raw).strip()
    return clean[:max_chars] if clean else "(no text extracted)"

def read_pdf_text(path: str, max_chars: int = 60000) -> str:
    """
    Read more text from the PDF for analysis (bigger limit).
    Uses PyPDF2.
    """
    try:
        reader = PdfReader(path)
        chunks = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt:
                chunks.append(txt)
            if sum(len(c) for c in chunks) >= max_chars:
                break
        return "\n".join(chunks)
    except Exception:
        # fallback: just use the preview extractor
        return extract_text_from_pdf(path, max_chars=max_chars)

_MONTHS = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t)?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"

def detect_publication_date(text: str) -> str:
    """
    Heuristic: look for 'Month YYYY' or standalone 'YYYY' (>= 2000).
    """
    if not text:
        return ""
    m = re.search(fr"{_MONTHS}\s+20\d{{2}}", text, flags=re.IGNORECASE)
    if m:
        return m.group(0)
    y = re.search(r"\b20\d{2}\b", text)
    if y:
        return y.group(0)
    return ""

def lookup_standard(std_name: str) -> dict | None:
    row = df.loc[df["Standard"].astype(str).str.strip() == std_name].head(1)
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "standard": r.get("Standard", ""),
        "pub_date": r.get("Publication Date", ""),
        "tfidf": r.get("TFIDF Keywords", ""),
        "contextual": r.get("Contextual Keywords", "")
    }

# ------------ Routes ------------
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        standards=standards,
        selected=None,
        result=None,
        std_info=None,
        error=None,
        message=None,
    )

@app.route("/analyze", methods=["POST"])
def analyze():
    std = (request.form.get("standard") or "").strip()
    pdf_file = request.files.get("bank_pdf")

    if not std:
        return render_template(
            "index.html",
            standards=standards,
            selected=None,
            result=None,
            std_info=None,
            error="Please select a standard.",
            message=None,
        )

    if not pdf_file or pdf_file.filename == "":
        return render_template(
            "index.html",
            standards=standards,
            selected=std,
            result=None,
            std_info=None,
            error="Please upload a bank ESG report (PDF).",
            message=None,
        )

    if not allowed_file(pdf_file.filename):
        return render_template(
            "index.html",
            standards=standards,
            selected=std,
            result=None,
            std_info=None,
            error="The uploaded file should be a PDF.",
            message=None,
        )

    fname = secure_filename(pdf_file.filename)
    fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    pdf_file.save(fpath)

    # Short preview for display
    preview = extract_text_from_pdf(fpath, max_chars=2000)

    # Full text for analysis (stopwords, date, TF-IDF, contextual)
    full_text = read_pdf_text(fpath, max_chars=60000)
    cleaned_text = remove_stopwords(full_text)

    # Publication date
    bank_pub_date = detect_publication_date(full_text)

    # TF-IDF bigram keywords
    bank_tfidf_list = extract_tfidf_keywords(cleaned_text)
    bank_tfidf_str = ", ".join(bank_tfidf_list) if bank_tfidf_list else ""

    # NEW: contextual keywords with YAKE
    bank_contextual_list = extract_contextual_keywords(cleaned_text, top_n=5)
    bank_contextual_str = ", ".join(bank_contextual_list) if bank_contextual_list else ""

    std_info = lookup_standard(std)

    # Result object passed to the template
    result = {
        "filename": fname,
        "standard": std,
        "preview": preview,
        "bank_pub_date": bank_pub_date,
        "bank_tfidf": bank_tfidf_str,
        "bank_contextual": bank_contextual_str,  # NEW FIELD
    }

    return render_template(
        "index.html",
        standards=standards,
        selected=std,
        result=result,
        std_info=std_info,
        error=None,
        message=None,
    )

if __name__ == "__main__":
    # Azure uses WEBSITES_PORT in production; 8000 is good locally.
    app.run(host="0.0.0.0", port=8000)

import io
import re
from typing import List, Tuple

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text stream from PDF document pages."""
    if pypdf is None:
        raise ImportError("pypdf is not installed. Run: pip install pypdf")
    
    stream = io.BytesIO(file_bytes)
    reader = pypdf.PdfReader(stream)
    pages_text = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages_text.append(text.strip())
    return "\n\n".join(pages_text)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract raw text from Microsoft Word .docx XML structures."""
    if docx is None:
        raise ImportError("python-docx is not installed. Run: pip install python-docx")
    
    stream = io.BytesIO(file_bytes)
    doc = docx.Document(stream)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Universal text ingestion router based on file extension."""
    ext = filename.lower().split('.')[-1]
    if ext == 'pdf':
        return extract_text_from_pdf(file_bytes)
    elif ext == 'docx':
        return extract_text_from_docx(file_bytes)
    elif ext in ('txt', 'md'):
        try:
            return file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            return file_bytes.decode('latin-1', errors='ignore')
    else:
        raise ValueError(f"Unsupported file format: .{ext}. Allowed: .pdf, .docx, .txt")


def segment_sentences(text: str) -> List[str]:
    """
    Robust academic sentence segmentation preserving abbreviations,
    acronyms, references, and mathematical notation.
    """
    if not text or not text.strip():
        return []

    # Protect abbreviations commonly found in academic research papers
    protected = text
    subs = [
        (r'et\s+al\.', 'et_al_dot'),
        (r'i\.e\.', 'i_e_dot'),
        (r'e\.g\.', 'e_g_dot'),
        (r'Dr\.', 'Dr_dot'),
        (r'Prof\.', 'Prof_dot'),
        (r'Fig\.', 'Fig_dot'),
        (r'Tab\.', 'Tab_dot'),
        (r'vs\.', 'vs_dot'),
        (r'Eq\.', 'Eq_dot'),
        (r'p\.', 'p_dot'),
        (r'pp\.', 'pp_dot'),
        (r'(\d+)\.(\d+)', r'\1_dec_\2')
    ]
    for pattern, repl in subs:
        protected = re.sub(pattern, repl, protected, flags=re.IGNORECASE)

    # Split on sentence boundaries (period, question mark, exclamation followed by space & capitalized word)
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"“\'])', protected)

    restored = []
    for s in raw_sentences:
        clean = (s.replace('et_al_dot', 'et al.')
                  .replace('i_e_dot', 'i.e.')
                  .replace('e_g_dot', 'e.g.')
                  .replace('Dr_dot', 'Dr.')
                  .replace('Prof_dot', 'Prof.')
                  .replace('Fig_dot', 'Fig.')
                  .replace('Tab_dot', 'Tab.')
                  .replace('vs_dot', 'vs.')
                  .replace('Eq_dot', 'Eq.')
                  .replace('p_dot', 'p.')
                  .replace('pp_dot', 'pp.')
                  .replace('_dec_', '.')
                  .strip())
        # Filter out negligible noise or stray numbering
        if len(clean) > 8:
            restored.append(clean)

    return restored

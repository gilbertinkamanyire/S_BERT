"""
VeriText Core Forensics Engine (Built from Scratch - Zero External ML Dependencies)
==================================================================================
A multi-layered forensics pipeline implementing:
1. Multi-Gram Statistical Perplexity (Unigram, Bigram, Trigram conditional entropy)
2. Goh & Barabási (2008) Burstiness Cadence (Inter-sentence syntactic dynamics)
3. Lexical Constancy & Diversity (Type-Token Ratio TTR, Simpson Index, Yule's K)
4. AI Tell-Tale Buzzword & Discourse Archetype Classifier (Nucleus sampling markers)
5. Syntactic Symmetry & Clause Uniformity (Punctuation and clause-length CV)
6. Deep Semantic Plagiarism / Paraphrase Checker (Character-Trigram Jaccard + Cosine Space)
"""

import math
import re
from typing import Dict, List, Any, Tuple
from collections import Counter
from . import detector

# High-frequency structural words in natural English
STOPWORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he',
    'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or',
    'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about',
    'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
    'take', 'person', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then',
    'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how',
    'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us'
}

# Common academic and everyday vocabulary (low surprisal in standard English)
COMMON_ACADEMIC_AND_EVERYDAY = {
    'people', 'history', 'world', 'life', 'school', 'student', 'study', 'system', 'water', 'city', 'state',
    'family', 'group', 'problem', 'hand', 'part', 'place', 'case', 'week', 'company', 'program', 'child',
    'large', 'small', 'early', 'late', 'young', 'old', 'different', 'important', 'long', 'great', 'little',
    'tell', 'ask', 'seem', 'feel', 'try', 'leave', 'call', 'find', 'give', 'need', 'become', 'start', 'show',
    'learning', 'digital', 'technology', 'technologies', 'modern', 'tools', 'tool', 'education', 'educational',
    'artificial', 'intelligence', 'transforming', 'process', 'information', 'research', 'academic', 'development',
    'methods', 'method', 'analysis', 'results', 'data', 'approach', 'model', 'models', 'framework', 'future',
    'landscape', 'rapidly', 'evolving', 'deep', 'integration', 'integrate', 'innovation', 'significance',
    'potential', 'crucial', 'essential', 'effective', 'effectively', 'continuous', 'adaptation', 'critical',
    'understanding', 'enhance', 'enhancing', 'insights', 'strategies', 'impact'
}

# Formulaic AI generative buzzwords frequently oversampled by LLMs
AI_HALLMARKS = {
    'delve': 4.0, 'delving': 4.0, 'testament': 3.5, 'tapestry': 4.0, 'beacon': 3.5,
    'pivotal': 3.5, 'paramount': 3.5, 'underscores': 3.0, 'fosters': 3.0, 'foster': 3.0,
    'fostering': 3.0, 'unveils': 2.8, 'demystify': 3.5, 'revolutionize': 3.0,
    'transformative': 3.0, 'multifaceted': 3.5, 'cornerstone': 3.0, 'synergy': 3.5,
    'seamless': 3.0, 'seamlessly': 3.0, 'holistic': 2.8, 'navigating': 3.0,
    'interplay': 3.5, 'realm': 3.2, 'harnessing': 3.0, 'embark': 3.5,
    'reverberate': 3.5, 'plethora': 3.5, 'intricate': 2.5, 'nuanced': 2.5,
    'ever-evolving': 3.5, 'groundbreaking': 2.8, 'paradigms': 3.0, 'paradigm': 3.0,
    'unprecedented': 2.8, 'crucial': 2.0
}

AI_PHRASE_PATTERNS = [
    r'\bplays a pivotal role\b',
    r'\bin conclusion,?\b',
    r'\bit is important to note\b',
    r'\bit is worth noting\b',
    r'\bin today\'?s (?:digital|modern|rapidly evolving|fast-paced)\b',
    r'\ba myriad of\b',
    r'\bserves as a testament\b',
    r'\bfostering innovation\b',
    r'\bdeep dive\b',
    r'\bgame-changer\b',
    r'\bby leveraging the power of\b',
    r'\bseamlessly integrates?\b',
    r'\brapidly evolving digital landscape\b'
]


# =============================================================================
# 1. TEXT TOKENIZATION & SENTENCE SEGMENTATION
# =============================================================================

def segment_into_sentences(raw_text: str) -> List[str]:
    """
    Robust sentence segmenter protecting common abbreviations (Dr., Prof., et al., e.g., i.e.)
    """
    if not raw_text:
        return []

    # Protect abbreviations with rare placeholders
    text = raw_text.strip()
    abbrevs = {
        r'\bet al\.': 'ET__AL',
        r'\be\.g\.': 'E__G',
        r'\bi\.e\.': 'I__E',
        r'\bvs\.': 'V__S',
        r'\bDr\.': 'DR__P',
        r'\bProf\.': 'PROF__P',
        r'\bFig\.': 'FIG__P',
        r'\bTab\.': 'TAB__P',
        r'\bEq\.': 'EQ__P',
        r'(\d+)\.(\d+)': r'\1__DEC__\2'
    }

    protected = text
    for pat, rep in abbrevs.items():
        protected = re.sub(pat, rep, protected, flags=re.IGNORECASE)

    raw_sentences = re.split(r'(?<=[.!?])\s+', protected)
    results = []

    for s in raw_sentences:
        clean = s.strip()
        if not clean:
            continue
        # Restore abbreviations
        clean = clean.replace('ET__AL', 'et al.')
        clean = clean.replace('E__G', 'e.g.')
        clean = clean.replace('I__E', 'i.e.')
        clean = clean.replace('V__S', 'vs.')
        clean = clean.replace('DR__P', 'Dr.')
        clean = clean.replace('PROF__P', 'Prof.')
        clean = clean.replace('FIG__P', 'Fig.')
        clean = clean.replace('TAB__P', 'Tab.')
        clean = clean.replace('EQ__P', 'Eq.')
        clean = re.sub(r'(\d+)__DEC__(\d+)', r'\1.\2', clean)
        if len(clean) > 2:
            results.append(clean)

    return results


def tokenize_words(text: str) -> List[str]:
    """Tokenize clean alphanumeric words."""
    return [w for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower()) if w]


# =============================================================================
# 2. STATISTICAL PERPLEXITY SURPRISAL
# =============================================================================

def compute_word_surprisal(word: str) -> float:
    """Estimated bits of surprisal for English words."""
    if word.isdigit():
        return 5.5
    if word in STOPWORDS:
        return 1.3
    if word in AI_HALLMARKS:
        return 1.4  # Clichés have very high probability under LLM generation
    if word in COMMON_ACADEMIC_AND_EVERYDAY:
        return 2.2

    rare_chars = sum(1 for c in word if c in 'zqxjkv')
    base = 3.4 + min(2.4, len(word) * 0.22) + (rare_chars * 0.45)
    return min(7.2, base)


def compute_sentence_perplexity(sentence: str) -> Dict[str, float]:
    """Calculate token entropy and perplexity (PPL)."""
    tokens = tokenize_words(sentence)
    if not tokens:
        return {'ppl': 20.0, 'entropy': 4.3, 'words': 0}

    total_bits = 0.0
    prev = None

    for w in tokens:
        bits = compute_word_surprisal(w)
        if prev:
            if (prev in STOPWORDS or prev in COMMON_ACADEMIC_AND_EVERYDAY or prev in AI_HALLMARKS) and \
               (w in STOPWORDS or w in COMMON_ACADEMIC_AND_EVERYDAY or w in AI_HALLMARKS):
                bits *= 0.85
            elif prev not in STOPWORDS and w not in STOPWORDS:
                bits *= 1.15
        total_bits += bits
        prev = w

    entropy = total_bits / len(tokens)
    ppl = math.pow(2.0, entropy)
    return {
        'ppl': round(ppl, 1),
        'entropy': round(entropy, 2),
        'words': len(tokens)
    }


# =============================================================================
# 3. BURSTINESS CADENCE (Goh & Barabási Metric)
# =============================================================================

def compute_burstiness(lengths: List[int]) -> float:
    """
    Goh & Barabási parameter B = (sigma - mu) / (sigma + mu) in [-1, +1]
    Human rhythm: B > 0.0
    Robotic uniformity: B < -0.30
    """
    if not lengths or len(lengths) <= 1:
        return 0.05

    n = len(lengths)
    mu = sum(lengths) / n
    var = sum((x - mu) ** 2 for x in lengths) / (n - 1)
    sigma = math.sqrt(var)

    if sigma + mu == 0:
        return 0.0
    return round(max(-1.0, min(1.0, (sigma - mu) / (sigma + mu))), 3)


# =============================================================================
# 4. LEXICAL DIVERSITY & VOCABULARY METRICS
# =============================================================================

def compute_lexical_diversity(tokens: List[str]) -> Dict[str, float]:
    """TTR (Type-Token Ratio) and Yule's K characteristic."""
    n = len(tokens)
    if n == 0:
        return {'ttr': 0.0, 'yules_k': 0.0}

    counts = Counter(tokens)
    types = len(counts)
    ttr = types / n

    freq_spectrum = Counter(counts.values())
    s2 = sum(c * (f ** 2) for f, c in freq_spectrum.items())
    yules_k = 10000.0 * (s2 - n) / (n ** 2) if n > 1 else 0.0

    return {
        'ttr': round(ttr, 3),
        'yules_k': round(max(0.0, yules_k), 2)
    }


# =============================================================================
# 5. AI MARKERS & DISCOURSE CRUTCHES
# =============================================================================

def detect_ai_markers(text: str) -> Dict[str, Any]:
    """Find hallmarks and formulaic templates in text."""
    lower = text.lower()
    found_buzzwords = []
    penalty = 0.0

    for word, weight in AI_HALLMARKS.items():
        matches = len(re.findall(r'\b' + re.escape(word) + r'\b', lower))
        if matches > 0:
            found_buzzwords.append({'word': word, 'count': matches})
            penalty += matches * weight

    found_phrases = []
    for pat in AI_PHRASE_PATTERNS:
        matches = len(re.findall(pat, lower))
        if matches > 0:
            found_phrases.append({'phrase': pat.replace(r'\b', '').replace(r'\?', '?'), 'count': matches})
            penalty += matches * 4.5

    return {
        'buzzwords': found_buzzwords,
        'phrases': found_phrases,
        'marker_count': sum(b['count'] for b in found_buzzwords) + sum(p['count'] for p in found_phrases),
        'penalty': round(penalty, 2)
    }


# =============================================================================
# 6. ENSEMBLE AI DETECTOR
# =============================================================================

def analyze_ai_content(text: str) -> Dict[str, Any]:
    """Run the calibrated document detector and shape its output for the UI."""
    sentences = segment_into_sentences(text)
    result = detector.classify_text_ai_probability(text, sentences)
    if not sentences:
        return {
            "ai_probability": 0.0, "human_probability": 100.0,
            "verdict": "No Text", "verdict_color": "neutral",
            "verdict_desc": "Enter valid text to begin analysis.",
            "metrics": {}, "detected_buzzwords": [], "detected_phrases": [],
            "sentences": []
        }

    metrics = result.get("metrics", {})
    sentence_evals = []
    for evaluation in result.get("sentence_evaluations", []):
        sentence_evals.append({
            "id": evaluation["index"] + 1,
            "text": evaluation["sentence"],
            "words": evaluation["word_count"],
            "ppl": evaluation["perplexity"],
            "entropy": evaluation["entropy"],
            "ai_prob": evaluation["ai_prob"],
            "is_ai": evaluation["flagged_ai"],
            "markers": evaluation.get("markers_found", [])
        })

    return {
        "ai_probability": result["ai_probability"],
        "human_probability": result["human_probability"],
        "verdict": result["verdict"], "verdict_color": result["verdict_color"],
        "verdict_desc": result["verdict_desc"],
        "confidence": result.get("confidence", 0.0),
        "sample_warning": result.get("sample_warning", ""),
        "metrics": {
            "avg_ppl": metrics.get("avg_perplexity", 20.0),
            "burstiness": metrics.get("burstiness", 0.0),
            "lexical_ttr": metrics.get("lexical_ttr", 0.0),
            "yules_k": metrics.get("yules_k", 0.0),
            "marker_count": metrics.get("marker_count", 0),
            "ai_sentence_count": metrics.get("ai_sentence_count", 0),
            "total_sentences": metrics.get("total_sentences", len(sentences)),
            "ai_sentence_ratio": metrics.get("ai_sentence_ratio", 0.0),
            "proposal_ai_score": metrics.get("proposal_ai_score", 0.0),
            "ensemble_ai_score": metrics.get("ensemble_ai_score", 0.0),
            "perplexity_signal": metrics.get("perplexity_signal", 0.0),
            "burstiness_signal": metrics.get("burstiness_signal", 0.0),
            "perplexity_weight": metrics.get("perplexity_weight", 0.55),
            "burstiness_weight": metrics.get("burstiness_weight", 0.45)
        },
        "detected_buzzwords": result.get("detected_buzzwords", []),
        "detected_phrases": result.get("detected_phrases", []),
        "sentences": sentence_evals
    }

# =============================================================================
# 7. DEEP SEMANTIC PLAGIARISM CHECKER (Trigram Jaccard + Dense Subspace)
# =============================================================================

def extract_char_trigrams(text: str) -> set:
    """Extract character trigrams for paraphrase and synonym-swap detection."""
    clean = re.sub(r'[^a-z0-9]', '', text.lower())
    if len(clean) < 3:
        return {clean} if clean else set()
    return {clean[i:i+3] for i in range(len(clean) - 2)}


_sbert_model = None

def get_sbert_model():
    """Lazy-load Sentence-BERT model (all-MiniLM-L6-v2) if installed."""
    global _sbert_model
    if _sbert_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            _sbert_model = False
    return _sbert_model if _sbert_model is not False else None


def compute_dense_vector_384(text: str) -> List[float]:
    """
    Generate normalized 384-dimensional dense vector coordinates v in R^384
    satisfying ||v|| = 1.000 for sentence-transformers/all-MiniLM-L6-v2.
    Uses native SBERT when available, and mathematical continuous projection fallback.
    """
    model = get_sbert_model()
    if model is not None:
        try:
            emb = model.encode(text, normalize_embeddings=True)
            return [float(x) for x in emb]
        except Exception:
            pass

    # High-Dimensional Continuous Manifold Deterministic Projection (R^384)
    vec = [0.0] * 384
    words = re.findall(r'[a-z0-9]+', text.lower())
    if not words:
        return vec

    for word in words:
        h = 2166136261
        for ch in word:
            h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF

        b1 = abs(h) % 384
        b2 = abs(h * 31) % 384
        weight = 1.0 + (0.75 if len(word) > 5 else 0.0)
        vec[b1] += weight
        vec[b2] += weight * 0.45

        # Subword character trigrams for morphological paraphrase proximity
        for i in range(len(word) - 2):
            tri = word[i:i+3]
            th = 0
            for tch in tri:
                th = ((th << 5) - th + ord(tch)) & 0xFFFFFFFF
            tb = abs(th) % 384
            vec[tb] += 0.35

    # Continuous manifold smoothing
    smoothed = [0.0] * 384
    for i in range(384):
        prev_val = vec[(i + 383) % 384]
        curr_val = vec[i]
        next_val = vec[(i + 1) % 384]
        smoothed[i] = curr_val * 0.70 + (prev_val + next_val) * 0.15

    # L2 Normalization (||v|| = 1.000)
    norm_sq = sum(x * x for x in smoothed)
    norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
    return [round(x / norm, 6) for x in smoothed]


def check_plagiarism_against_reference(
    doc_a_sentences: List[str],
    doc_b_sentences: List[str],
    similarity_threshold: float = 0.75
) -> Dict[str, Any]:
    """
    Phase 2: Semantic Vector Analysis Module 
    - Sentence-BERT (all-MiniLM-L6-v2) 384-D dense embeddings
    - Cosine Similarity: (A . B) / (||A|| ||B||)
    - Paraphrase threshold >= 0.75
    - Color-coded tiers:
      * High Similarity (>=85%): Red
      * Moderate Paraphrase (70%-84%): Yellow
      * AI-Generated: Purple/Blue
      * Clean: Green
    """
    if not doc_a_sentences or not doc_b_sentences:
        return {
            'has_reference': False,
            'overall_similarity': 0.0,
            'plagiarism_rate': 0.0,
            'flagged_count': 0,
            'high_count': 0,
            'moderate_count': 0,
            'matches': []
        }

    # Pre-index Reference Corpus (Document B)
    ref_corpus = []
    for s_b in doc_b_sentences:
        s_b_clean = s_b.strip()
        ref_corpus.append({
            'text': s_b_clean,
            'trigrams': extract_char_trigrams(s_b_clean),
            'vec': compute_dense_vector_384(s_b_clean)
        })

    threshold_value = float(similarity_threshold)
    threshold = max(0.0, min(1.0, threshold_value / 100.0 if threshold_value > 1.0 else threshold_value))
    threshold_pct = threshold * 100.0
    flagged = 0
    high_count = 0
    moderate_count = 0
    sim_scores = []
    sentence_matches = []

    for idx, s_a in enumerate(doc_a_sentences):
        s_a_clean = s_a.strip()
        tri_a = extract_char_trigrams(s_a_clean)
        vec_a = compute_dense_vector_384(s_a_clean)

        best_score = 0.0
        best_match_clause = "No matching clause found."

        for item_b in ref_corpus:
            # 1. Character Trigram Jaccard (morphological / verbatim overlap)
            union_len = len(tri_a | item_b['trigrams'])
            jaccard = len(tri_a & item_b['trigrams']) / max(1, union_len)

            # 2. SBERT 384-D Vector Cosine Distance (Jurafsky & Martin, 2024; Reimers & Gurevych, 2019)
            # Cosine = (A . B) / (||A|| ||B||)
            cosine = sum(a * b for a, b in zip(vec_a, item_b['vec']))
            cosine = max(0.0, min(1.0, cosine))

            # Deep Semantic Paraphrase Score:
            # Uses SBERT Vector Cosine as primary semantic measure, boosting for verbatim overlap
            score = max(cosine, (jaccard * 0.35) + (cosine * 0.65))
            if score > best_score:
                best_score = score
                best_match_clause = item_b['text']

        sim_scores.append(best_score)
        pct_score = round(best_score * 100.0, 1)

        # Concept Paper Phase 4 Tiers
        if pct_score >= 85.0 and pct_score >= threshold_pct:
            tier = 'high_similarity'
            color = 'red'
            is_plag = True
            high_count += 1
            flagged += 1
        elif pct_score >= threshold_pct:
            tier = 'moderate_paraphrase'
            color = 'yellow'
            is_plag = True
            moderate_count += 1
            flagged += 1
        else:
            tier = 'clean'
            color = 'green'
            is_plag = False

        sentence_matches.append({
            'id': idx + 1,
            'sentence': s_a_clean,
            'similarity': pct_score,
            'matched_clause': best_match_clause,
            'is_plagiarized': is_plag,
            'tier': tier,
            'color': color
        })

    avg_sim = (sum(sim_scores) / len(sim_scores)) * 100.0 if sim_scores else 0.0
    plagiarism_rate = (flagged / max(1, len(doc_a_sentences))) * 100.0

    return {
        'has_reference': True,
        'overall_similarity': round(avg_sim, 1),
        'plagiarism_rate': round(plagiarism_rate, 1),
        'flagged_count': flagged,
        'high_count': high_count,
        'moderate_count': moderate_count,
        'total_sentences': len(doc_a_sentences),
        'matches': sentence_matches
    }


# =============================================================================
# 7. AI TEXT HUMANIZER (Rule-Based Stylometric Restructuring Engine)
# =============================================================================

HUMANIZER_PHRASE_MAP = [
    (r"\bIn today's rapidly evolving digital landscape,?\b", "In current digital settings,"),
    (r"\bin today's digital landscape,?\b", "in modern technology,"),
    (r"\bit is crucial to delve into\b", "we should examine"),
    (r"\bit is crucial to\b", "it is essential to"),
    (r"\bit is important to note that\b", "notably,"),
    (r"\bit is worth noting that\b", "in particular,"),
    (r"\bplays a pivotal role in\b", "is central to"),
    (r"\bplays a pivotal role\b", "is essential"),
    (r"\bserves as a testament to\b", "reflects"),
    (r"\bserves as a testament\b", "stands as clear evidence"),
    (r"\bmultifaceted tapestry of\b", "complex structure of"),
    (r"\bmultifaceted tapestry\b", "varied dynamics"),
    (r"\bconsequently,\b", "as a result,"),
    (r"\bfurthermore,\b", "also,"),
    (r"\bmoreover,\b", "in addition,"),
    (r"\bharnessing the power of\b", "using"),
    (r"\bharnessing\b", "applying"),
    (r"\bfosters seamless synergy\b", "improves coordination"),
    (r"\bfosters seamless\b", "supports smooth"),
    (r"\bfosters\b", "encourages"),
    (r"\bseamless synergy\b", "effective collaboration"),
    (r"\bseamless\b", "fluid"),
    (r"\bsynergy\b", "teamwork"),
    (r"\brevolutionizes modern society\b", "alters everyday practices"),
    (r"\bcornerstone of innovation\b", "foundation of progress"),
    (r"\bcornerstone\b", "foundation"),
    (r"\bparamount efficiency\b", "practical efficiency"),
    (r"\bparamount\b", "central"),
    (r"\bholistic, nuanced solutions\b", "thorough solutions"),
    (r"\bholistic and nuanced\b", "thorough and balanced"),
    (r"\bholistic\b", "comprehensive"),
    (r"\bnuanced\b", "careful"),
    (r"\btransformative\b", "substantial"),
    (r"\bgroundbreaking paradigms\b", "fresh approaches"),
    (r"\bgroundbreaking\b", "notable"),
    (r"\bembark on a journey toward\b", "work toward"),
    (r"\bembark on a journey\b", "start working"),
    (r"\bembark on\b", "begin"),
    (r"\bdelve into\b", "explore"),
    (r"\bdelving into\b", "looking into"),
    (r"\bdelve\b", "examine"),
    (r"\ba plethora of\b", "a range of"),
    (r"\bplethora\b", "variety"),
    (r"\binterplay between\b", "interaction of"),
    (r"\binterplay\b", "relationship"),
    (r"\bunderscores the pivotal role\b", "highlights the need"),
    (r"\bunderscores\b", "highlights"),
    (r"\bdeep dive\b", "detailed look"),
    (r"\bbeacon of\b", "model for"),
    (r"\btapestry of\b", "blend of"),
    (r"\btapestry\b", "framework"),
    (r"\bdemystify\b", "clarify"),
    (r"\breverberate\b", "echo"),
    (r"\bever-evolving\b", "shifting"),
    (r"\bby leveraging the power of\b", "by using"),
    (r"\bby leveraging\b", "by applying"),
    (r"\bleveraging\b", "using"),
]

def humanize_text(raw_text: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
    """Rewrite formulaic prose locally while preserving its paragraphs and meaning."""
    original = raw_text.strip()
    options = options or {}
    mode = str(options.get('mode', 'balanced')).lower()
    intensity = max(1, min(3, int(options.get('intensity', 2))))
    if not original:
        return {'original': '', 'humanized': '', 'original_ai_prob': 0, 'humanized_ai_prob': 0, 'improvements': []}

    modified = original.replace('\r\n', '\n').replace('\r', '\n')
    replacements_made = []

    if mode == 'casual':
        modified = re.sub(r"\bit is\b", "it's", modified, flags=re.IGNORECASE)
        modified = re.sub(r"\bdo not\b", "don't", modified, flags=re.IGNORECASE)
        modified = re.sub(r"\bcannot\b", "can't", modified, flags=re.IGNORECASE)
        modified = re.sub(r"\bwill not\b", "won't", modified, flags=re.IGNORECASE)
        replacements_made.append('Used natural contractions for a conversational tone')

    replacement_limit = {1: 0.45, 2: 0.75, 3: 1.0}[intensity]
    applicable_phrases = HUMANIZER_PHRASE_MAP if intensity >= 2 else HUMANIZER_PHRASE_MAP[::2]
    for pattern, replacement in applicable_phrases:
        matches = re.findall(pattern, modified, flags=re.IGNORECASE)
        if not matches:
            continue
        replacements_to_apply = max(1, math.ceil(len(matches) * replacement_limit))
        modified = re.sub(pattern, replacement, modified, flags=re.IGNORECASE, count=replacements_to_apply)
        replacements_made.append(f'Rephrased repeated AI-style phrase “{matches[0]}”')

    # Structure-aware cadence edits, keeping paragraph boundaries intact.
    output_paragraphs = []
    for paragraph in modified.split('\n\n'):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        restructured_sents = []
        for sentence in segment_into_sentences(paragraph):
            words = sentence.split()
            if len(words) >= 28 and ',' in sentence:
                parts = re.split(r'\s+and\s+', sentence, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 2 and len(parts[0].split()) >= 10 and len(parts[1].split()) >= 9:
                    first = parts[0].strip().rstrip(',;') + '.'
                    second = parts[1].strip().rstrip('.,;') + '.'
                    second = second[0].upper() + second[1:] if len(second) > 1 else second.upper()
                    restructured_sents.extend([first, second])
                    replacements_made.append('Split an overlong sentence to vary reading cadence')
                    continue
            restructured_sents.append(sentence.strip())
        output_paragraphs.append(' '.join(restructured_sents))

    humanized = '\n\n'.join(output_paragraphs)
    humanized = re.sub(r'\bIn conclusion\b', 'Overall', humanized, flags=re.IGNORECASE)
    humanized = re.sub(r'\bTo summarize\b', 'Taken together', humanized, flags=re.IGNORECASE)
    humanized = re.sub(r'\s+([,.;:?])', r'\1', humanized)
    humanized = re.sub(r',{2,}', ',', humanized)
    humanized = re.sub(r'([.!?])([A-Z])', r'\1 \2', humanized)

    orig_eval = analyze_ai_content(original)
    human_eval = analyze_ai_content(humanized)
    prob_reduction = round(orig_eval['ai_probability'] - human_eval['ai_probability'], 1)

    return {
        'original': original,
        'humanized': humanized,
        'original_ai_prob': orig_eval['ai_probability'],
        'original_verdict': orig_eval['verdict'],
        'humanized_ai_prob': human_eval['ai_probability'],
        'humanized_verdict': human_eval['verdict'],
        'prob_reduction': prob_reduction,
        'replacements_count': len(replacements_made),
        'improvements': list(dict.fromkeys(replacements_made))[:8],
        'word_count': len(humanized.split()),
        'options': {'mode': mode, 'intensity': intensity},
        'note': 'Uses local rule-based rewriting and does not guarantee that third-party AI detectors will classify the result as human-written.'
    }


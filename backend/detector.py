"""
AI Content & Plagiarism Forensics Detector (v3.1 Calibrated Pipeline)
=====================================================================
Multi-dimensional feature extraction with calibrated thresholds for
distinguishing genuine human academic scholarship from LLM generation.
"""

import math
import re
from typing import Dict, List, Any
from collections import Counter

_causal_tokenizer = None
_causal_model = None
_causal_load_attempted = False


def load_local_perplexity_model():
    """Load GPT-2 once and reuse it for proposal-equation perplexity."""
    global _causal_tokenizer, _causal_model, _causal_load_attempted
    if _causal_load_attempted:
        return _causal_tokenizer, _causal_model

    _causal_load_attempted = True
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _causal_tokenizer = AutoTokenizer.from_pretrained('gpt2')
        _causal_model = AutoModelForCausalLM.from_pretrained('gpt2')
        _causal_model.eval()
    except Exception as exc:
        print(f"Local perplexity model unavailable; using calibrated fallback: {exc}")
    return _causal_tokenizer, _causal_model


# =============================================================================
# 1. EXPANDED LEXICAL TIERS & RARITY SCALES
# =============================================================================

# Stopwords & common function words
COMMON_FUNCTION_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he',
    'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or',
    'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about',
    'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
    'take', 'person', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then',
    'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how',
    'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us'
}

# Common everyday words (low surprisal, but standard in English)
COMMON_VOCAB = {
    'people', 'history', 'world', 'life', 'school', 'student', 'study', 'system', 'water', 'city', 'state',
    'family', 'group', 'problem', 'hand', 'part', 'place', 'case', 'week', 'company', 'program', 'child',
    'large', 'small', 'early', 'late', 'young', 'old', 'different', 'important', 'long', 'great', 'little',
    'tell', 'ask', 'seem', 'feel', 'try', 'leave', 'call', 'find', 'give', 'need', 'become', 'start', 'show'
}

# Formulaic AI Buzzwords & High-Frequency LLM Idioms
AI_TELL_TALE_WORDS = {
    'delve': 4.0, 'delving': 4.0, 'testament': 3.5, 'tapestry': 4.0, 'beacon': 3.5,
    'pivotal': 2.8, 'paramount': 2.8, 'underscores': 2.5, 'fosters': 2.5, 'unveils': 2.5,
    'demystify': 3.5, 'revolutionize': 2.5, 'transformative': 2.5, 'multifaceted': 3.0,
    'cornerstone': 2.8, 'synergy': 3.0, 'seamless': 2.8, 'holistic': 2.5,
    'navigating': 2.5, 'interplay': 3.0, 'realm': 3.0, 'harnessing': 2.5, 'embark': 3.5,
    'reverberate': 3.5, 'plethora': 3.0, 'intricate': 2.0, 'nuanced': 2.0
}

AI_PHRASE_MARKERS = [
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
    r'\bseamlessly integrates?\b'
]


def extract_tokens(text: str) -> List[str]:
    """Clean alphanumeric tokens."""
    return [w for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower()) if w]


# =============================================================================
# 2. CALIBRATED N-GRAM SURPRISAL & PERPLEXITY
# =============================================================================

def estimate_word_surprisal(word: str) -> float:
    """
    Calibrated natural surprisal score (bits of information):
    - Function words: ~1.2 - 1.8 bits
    - Frequent English words: ~2.5 - 3.2 bits
    - Specialized, domain-specific, numerical or rare words: ~4.5 - 7.5 bits
    """
    if word.isdigit():
        return 5.2  # Concrete numbers/years are characteristic of human reporting
    if word in COMMON_FUNCTION_WORDS:
        return 1.4
    if word in COMMON_VOCAB:
        return 2.8

    # Word rarity estimation based on length, morphology, and rare characters
    char_len = len(word)
    rare_chars = sum(1 for c in word if c in 'zqxjkv')
    base = 3.8 + min(2.5, char_len * 0.28) + (rare_chars * 0.5)
    return min(7.5, base)


def calculate_sentence_perplexity(sentence: str) -> Dict[str, float]:
    """
    Computes statistical token surprisal across the sentence.
    Natural human sentences typically exceed 30.0 - 55.0 PPL due to specific nouns,
    dates, names, and organic irregular phrasing.
    """
    tokens = extract_tokens(sentence)
    if not tokens:
        return {'perplexity': 22.0, 'entropy': 4.4, 'word_count': 0}

    tokenizer, model = load_local_perplexity_model()
    if tokenizer is not None and model is not None:
        try:
            import torch
            encoded = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                logits = model(**encoded).logits[:, :-1, :]
                targets = encoded['input_ids'][:, 1:]
                log_probs = torch.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
            avg_neg_log_likelihood = -token_log_probs.mean().item()
            ppl = math.exp(min(avg_neg_log_likelihood, 20.0))
            return {
                'perplexity': round(min(ppl, 1000000.0), 1),
                'entropy': round(avg_neg_log_likelihood / math.log(2.0), 2),
                'word_count': len(targets[0])
            }
        except Exception as exc:
            print(f"Conditional perplexity calculation failed; using calibrated fallback: {exc}")

    total_bits = 0.0
    prev_w = None
    for w in tokens:
        bits = estimate_word_surprisal(w)
        if prev_w:
            if prev_w in COMMON_FUNCTION_WORDS and w in COMMON_FUNCTION_WORDS:
                bits *= 0.82
            elif prev_w not in COMMON_FUNCTION_WORDS and w not in COMMON_FUNCTION_WORDS:
                bits *= 1.25
        total_bits += bits
        prev_w = w

    avg_entropy = total_bits / len(tokens)
    return {
        'perplexity': round(math.pow(2.0, avg_entropy), 1),
        'entropy': round(avg_entropy, 2),
        'word_count': len(tokens)
    }


# =============================================================================
# 3. BURSTINESS & STRUCTURAL CADENCE (Goh & Barabási, 2008)
# =============================================================================

def calculate_burstiness(lengths: List[int]) -> float:
    """
    Goh & Barabási burstiness parameter B in [-1, +1]:
    B = (sigma - mu) / (sigma + mu)

    For small samples (< 3 sentences), an unbiased fallback based on clause
    rhythm is returned rather than -1.0 to prevent false-positive penalization.
    """
    if not lengths or len(lengths) <= 1:
        return 0.05  # Neutral default for single sentences

    n = len(lengths)
    mu = sum(lengths) / n
    variance = sum((x - mu) ** 2 for x in lengths) / (n - 1)
    sigma = math.sqrt(variance)

    if sigma + mu == 0:
        return 0.0
    
    b = (sigma - mu) / (sigma + mu)
    return round(max(-1.0, min(1.0, b)), 3)


# =============================================================================
# 4. LEXICAL DIVERSITY & VOCABULARY METRICS
# =============================================================================

def calculate_lexical_metrics(tokens: List[str]) -> Dict[str, float]:
    """
    Type-Token Ratio (TTR) and Yule's K Characteristic.
    """
    n = len(tokens)
    if n == 0:
        return {'ttr': 0.0, 'root_ttr': 0.0, 'yules_k': 0.0}

    counts = Counter(tokens)
    types = len(counts)
    ttr = types / n
    root_ttr = types / math.sqrt(n)

    # Yule's K
    freq_spectrum = Counter(counts.values())
    s2 = sum(count * (freq ** 2) for freq, count in freq_spectrum.items())
    yules_k = 10000.0 * (s2 - n) / (n ** 2) if n > 1 else 0.0

    return {
        'ttr': round(ttr, 3),
        'root_ttr': round(root_ttr, 2),
        'yules_k': round(max(0.0, yules_k), 2)
    }


# =============================================================================
# 5. DISCOURSE MARKERS & FORMULAIC PATTERN DETECTION
# =============================================================================

def detect_ai_markers(text: str) -> Dict[str, Any]:
    """Detects tell-tale LLM lexical buzzwords and formulaic phrases."""
    lower = text.lower()
    found_buzzwords = []
    total_penalty = 0.0

    for word, weight in AI_TELL_TALE_WORDS.items():
        pattern = r'\b' + re.escape(word) + r'\b'
        matches = len(re.findall(pattern, lower))
        if matches > 0:
            found_buzzwords.append({'word': word, 'count': matches, 'weight': weight})
            total_penalty += weight * matches

    found_phrases = []
    for pat in AI_PHRASE_MARKERS:
        matches = len(re.findall(pat, lower))
        if matches > 0:
            found_phrases.append({'phrase': pat.replace(r'\b', '').replace(r'\?', '?'), 'count': matches})
            total_penalty += matches * 4.5

    return {
        'buzzwords': found_buzzwords,
        'phrases': found_phrases,
        'marker_penalty': round(total_penalty, 2),
        'marker_count': sum(b['count'] for b in found_buzzwords) + sum(p['count'] for p in found_phrases)
    }


# =============================================================================
# 6. CALIBRATED ENSEMBLE DETECTOR
# =============================================================================

def classify_text_ai_probability(text: str, sentences: List[str]) -> Dict[str, Any]:
    """
    Ensemble AI Detector:
    Combines:
    1. Sentence Perplexity Spectrum (Average & Minimum surprisal)
    2. Document & Clause Burstiness Cadence
    3. Formulaic AI Buzzword & Phrase Density
    4. Lexical Constancy (TTR & Yule's K)
    5. Specificity / Entity Density (dates, proper nouns, figures)
    """
    if not text.strip() or not sentences:
        return {
            'ai_probability': 0.0,
            'human_probability': 100.0,
            'verdict': 'Insufficient Text',
            'verdict_color': 'neutral',
            'verdict_desc': 'Please submit text or documents with valid sentences.',
            'confidence': 0.0,
            'metrics': {},
            'sentence_evaluations': []
        }

    tokens = extract_tokens(text)
    total_words = len(tokens)
    sent_lengths = [len(s.split()) for s in sentences]

    burstiness = calculate_burstiness(sent_lengths)
    lexical = calculate_lexical_metrics(tokens)
    markers = detect_ai_markers(text)

    # Check for concrete specificity (proper nouns, years, measurements)
    # Humans cite exact years (e.g. 1953), specific places, and proper nouns
    years_and_numbers = len(re.findall(r'\b(?:19|20)\d{2}\b|\b\d+(?:\.\d+)?%?\b', text))
    capitalized_words = len(re.findall(r'\b[A-Z][a-z]+\b', text))
    specificity_score = (years_and_numbers * 1.5) + (capitalized_words * 0.2)

    sentence_evals = []
    ppl_list = []
    ai_sentence_count = 0

    for idx, sent in enumerate(sentences):
        sent_clean = sent.strip()
        p_res = calculate_sentence_perplexity(sent_clean)
        s_ppl = p_res['perplexity']
        ppl_list.append(s_ppl)

        # Check markers in this sentence
        s_lower = sent_clean.lower()
        s_markers = [b['word'] for b in markers['buzzwords'] if b['word'] in s_lower]

        # Sentence-level logistic scoring
        # Average AI sentence has low perplexity (< 22) and formulaic flow
        # Human sentence has high perplexity (> 32)
        sent_z = -1.8  # Human-favored prior for individual sentences
        
        if s_ppl < 16.0:
            sent_z += 2.4
        elif s_ppl < 22.0:
            sent_z += 1.3
        elif s_ppl > 36.0:
            sent_z -= 2.0
        elif s_ppl > 28.0:
            sent_z -= 1.0

        if s_markers:
            sent_z += len(s_markers) * 1.6

        # Check sentence length uniformity
        if len(sent_lengths) >= 3 and burstiness < -0.30:
            sent_z += 0.8
        elif burstiness > 0.0:
            sent_z -= 0.8

        sent_prob = 1.0 / (1.0 + math.exp(-sent_z))
        sent_prob = max(0.01, min(0.99, sent_prob))
        is_ai_sent = sent_prob >= 0.50

        if is_ai_sent:
            ai_sentence_count += 1

        sentence_evals.append({
            'index': idx,
            'sentence': sent_clean,
            'word_count': p_res['word_count'],
            'perplexity': s_ppl,
            'entropy': p_res['entropy'],
            'ai_prob': round(sent_prob * 100.0, 1),
            'flagged_ai': is_ai_sent,
            'markers_found': s_markers
        })

    avg_ppl = sum(ppl_list) / len(ppl_list) if ppl_list else 25.0
    ai_sent_ratio = ai_sentence_count / max(1, len(sentences))

    # Document-Level Calibrated Log-Odds Ensemble
    # Academic prior: default toward human unless strong synthetic features trigger
    log_odds = -1.20

    # 1. Perplexity Impact
    if avg_ppl < 16.0:
        log_odds += 3.2
    elif avg_ppl < 22.0:
        log_odds += 2.1
    elif avg_ppl < 26.0:
        log_odds += 1.2
    elif avg_ppl > 38.0:
        log_odds -= 2.5
    elif avg_ppl > 30.0:
        log_odds -= 1.5

    # 2. Burstiness Impact (only active if 3+ sentences)
    if len(sentences) >= 3:
        if burstiness < -0.35:
            log_odds += 1.8
        elif burstiness < -0.18:
            log_odds += 0.9
        elif burstiness > 0.10:
            log_odds -= 1.6
        elif burstiness > -0.05:
            log_odds -= 0.8

    # 3. Marker & Buzzword Penalty
    marker_density = (markers['marker_penalty'] / max(1, total_words)) * 100.0
    if marker_density > 2.0:
        log_odds += 3.6
    elif marker_density > 1.0:
        log_odds += 2.2
    elif marker_density > 0.3:
        log_odds += 1.1

    # 4. Sentence AI Ratio
    log_odds += (ai_sent_ratio - 0.40) * 3.2

    # 5. Specificity Bonus (authentic details, names, numbers weaken AI odds)
    spec_density = (specificity_score / max(1, total_words)) * 100.0
    if spec_density > 3.0:
        log_odds -= 1.8
    elif spec_density > 1.0:
        log_odds -= 0.8

    # Proposed aggregate equation: AI Score = w1 * f(Perplexity) + w2 * g(Burstiness)
    # Lower perplexity and more uniform rhythm increase AI probability.
    perplexity_signal = 1.0 / (1.0 + math.exp(-(26.0 - avg_ppl) / 5.0))
    burstiness_signal = 1.0 / (1.0 + math.exp(-(burstiness + 0.10) / 0.25))
    proposal_score = (0.55 * perplexity_signal) + (0.45 * burstiness_signal)

    # Preserve the richer ensemble's other evidence, while making the two proposal
    # equations a substantial part of the final score.
    ensemble_ai = 1.0 / (1.0 + math.exp(-log_odds))
    ai_prob = (0.70 * proposal_score) + (0.30 * ensemble_ai)
    ai_prob = max(0.0, min(1.0, ai_prob))
    ai_prob_pct = round(ai_prob * 100.0, 1)
    human_prob_pct = round((1.0 - ai_prob) * 100.0, 1)

    # Classify Verdict
    if ai_prob_pct >= 80.0:
        verdict = "Highly Likely AI-Generated"
        verdict_color = "danger"
        verdict_desc = "The document exhibits strong synthetic signatures: uniform sentence cadence, low perplexity surprisal, and formulaic discourse structures typical of LLMs."
    elif ai_prob_pct >= 55.0:
        verdict = "Likely AI / Paraphrased"
        verdict_color = "warning"
        verdict_desc = "Noticeable AI-assisted or paraphrased patterns detected. Stylometric variability is subdued with clustered predictable transitions."
    elif ai_prob_pct >= 30.0:
        verdict = "Mixed Content (Human & AI-Assisted)"
        verdict_color = "accent"
        verdict_desc = "Displays an authentic human foundation with portions of polished or machine-generated phraseology."
    else:
        verdict = "Human Authored (Original Scholarship)"
        verdict_color = "success"
        verdict_desc = "Organic human prose confirmed. High perplexity variation, dynamic burstiness cadence, and natural syntactic diversity."

    return {
        'ai_probability': ai_prob_pct,
        'human_probability': human_prob_pct,
        'verdict': verdict,
        'verdict_color': verdict_color,
        'verdict_desc': verdict_desc,
        'confidence': round(abs(ai_prob_pct - 50.0) * 2.0, 1),
        'metrics': {
            'avg_perplexity': round(avg_ppl, 1),
            'burstiness': burstiness,
            'lexical_ttr': lexical['ttr'],
            'yules_k': lexical['yules_k'],
            'marker_penalty': markers['marker_penalty'],
            'marker_count': markers['marker_count'],
            'ai_sentence_count': ai_sentence_count,
            'total_sentences': len(sentences),
            'ai_sentence_ratio': round(ai_sent_ratio * 100.0, 1),
            'specificity_score': round(specificity_score, 1),
            'perplexity_signal': round(perplexity_signal, 4),
            'burstiness_signal': round(burstiness_signal, 4),
            'proposal_ai_score': round(proposal_score * 100.0, 1),
            'ensemble_ai_score': round(ensemble_ai * 100.0, 1),
            'perplexity_weight': 0.55,
            'burstiness_weight': 0.45
        },
        'detected_buzzwords': markers['buzzwords'],
        'detected_phrases': markers['phrases'],
        'sentence_evaluations': sentence_evals
    }

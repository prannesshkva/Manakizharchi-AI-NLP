import html
import re
from collections import Counter
from typing import Dict, List, Tuple

import nltk
from nltk.stem import WordNetLemmatizer

from app.core.config import settings
from app.models.schemas import FeatureExtractionResult, POSToken


_NLTK_READY = False
_LEMMATIZER = WordNetLemmatizer()

# Ironic / ambiguous emojis often mis-scored by classical lexicon polarity
IRONIC_OR_SLANG_EMOJIS: Dict[str, str] = {
    "💀": "skull_dead_funny",
    "☠️": "skull_dead_funny",
    "😭": "crying_overwhelmed",
    "🤡": "clown_mocking",
    "🙄": "eye_roll_sarcasm",
    "🙃": "upside_down_irony",
    "🔥": "fire_awesome",
    "💯": "hundred_valid",
    "❤️": "heart_love",
    "♥️": "heart_love",
    "😂": "laughing_joy",
    "🤣": "rofl_joy",
    "😍": "heart_eyes_love",
    "🥳": "party_excited",
    "🚀": "rocket_hype",
    "👍": "thumbs_up",
    "👎": "thumbs_down",
    "😡": "angry_furious",
    "😢": "sad_crying",
    "🙏": "grateful_pleading",
    "✨": "sparkles_vibe",
    "😒": "unimpressed_sideeye",
    "🤔": "skeptical_thinking",
}

IRONIC_EMOJI_KEYS = {"💀", "☠️", "🤡", "🙄", "🙃", "😭", "😒"}

ASCII_EMOTICONS: Dict[str, str] = {
    ":-)": "smiling_happy",
    ":)": "smiling_happy",
    ":-(": "sad_frown",
    ":(": "sad_frown",
    ":-D": "grinning_happy",
    ":D": "grinning_happy",
    ";-)": "winking_tease",
    ";)": "winking_tease",
    ":/": "skeptical_unsure",
    "<3": "heart_love",
    "</3": "broken_heart",
}

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "nothing", "nowhere", "none",
    "hardly", "scarcely", "barely", "dont", "don't", "doesnt", "doesn't",
    "didnt", "didn't", "cant", "can't", "cannot", "wont", "won't", "isnt",
    "isn't", "arent", "aren't", "wasnt", "wasn't", "werent", "weren't",
    "shouldnt", "shouldn't", "couldnt", "couldn't", "wouldnt", "wouldn't",
    "n't",
}

CONTRAST_CONJUNCTIONS = {
    "but", "however", "although", "though", "yet", "nevertheless",
    "nonetheless", "despite", "except", "unfortunately", "whereas",
}

SLANG_SINGLE_TERMS = {
    "mid", "fire", "lit", "goat", "goated", "cap", "dead", "womp", "cooked",
    "slay", "slayed", "slaying", "valid", "based", "cringe", "ratio", "ratioed",
    "lowkey", "highkey", "peak", "banger", "bussin", "delulu", "sus", "flex",
    "salty", "vibing", "vibe", "tbh", "rn", "af", "bruh", "bro",
    # Tamil / Tanglish colloquial sentiment terms (Manakizharchi domain)
    "sema", "mass", "mokkai", "mokke", "kalakkal", "thalaiva", "marana",
    "vere", "vera", "konjam", "overa", "superu", "poda", "mokka", "ghilli",
}

SLANG_MULTIWORD_PHRASES = (
    "no cap",
    "straight fire",
    "im dead",
    "i'm dead",
    "literally crying",
    "womp womp",
    "rent free",
    "touch grass",
    "main character",
    "hits different",
    "vera level",
    "sema mass",
    "let them cook",
    "caught in 4k",
    "its giving",
    "it's giving",
    "cant even",
    "can't even",
)


def ensure_nltk_resources() -> None:
    """Safely verify or download minimal NLTK corpora once per process."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    _NLTK_READY = True

    resources = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("tokenizers/punkt", "punkt"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass


def _wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith("J"):
        return "a"
    if treebank_tag.startswith("V"):
        return "v"
    if treebank_tag.startswith("R"):
        return "r"
    return "n"


def _pos_group(tag: str) -> str:
    if tag.startswith("JJ"):
        return "adjectives"
    if tag.startswith("RB"):
        return "adverbs"
    if tag.startswith("VB"):
        return "verbs"
    if tag.startswith("NN"):
        return "nouns"
    if tag == "UH":
        return "interjections"
    return "other"


def _safe_word_tokenize(text: str) -> List[str]:
    ensure_nltk_resources()
    try:
        return nltk.word_tokenize(text)
    except Exception:
        return re.findall(r"[A-Za-z0-9']+|[^\w\s]", text)


def _safe_sent_tokenize(text: str) -> List[str]:
    ensure_nltk_resources()
    try:
        sents = nltk.sent_tokenize(text)
        return [s.strip() for s in sents if s.strip()]
    except Exception:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _safe_pos_tag(tokens: List[str]) -> List[Tuple[str, str]]:
    if not tokens:
        return []
    ensure_nltk_resources()
    try:
        return nltk.pos_tag(tokens)
    except Exception:
        # Lightweight deterministic fallback tagger
        tagged = []
        for tok in tokens:
            low = tok.lower()
            if low in NEGATION_WORDS:
                tagged.append((tok, "RB"))
            elif low in CONTRAST_CONJUNCTIONS:
                tagged.append((tok, "CC"))
            elif low.endswith("ly") and len(low) > 3:
                tagged.append((tok, "RB"))
            elif low.endswith(("ing", "ed")) and len(low) > 4:
                tagged.append((tok, "VB"))
            elif low.endswith(("ful", "ous", "able", "ible", "less", "ive")):
                tagged.append((tok, "JJ"))
            else:
                tagged.append((tok, "NN"))
        return tagged


def clean_raw_text(raw_text: str) -> Tuple[str, List[str], List[str]]:
    """
    Strips unstructured textual noise (HTML, URLs, mentions, zero-width chars,
    excessive character elongations, whitespace jitter) while preserving linguistic
    emphasis and recording ironic emojis.

    Returns:
      (cleaned_text, noise_tags_removed, ironic_emojis_found)
    """
    text = str(raw_text or "")
    noise_tags: List[str] = []
    ironic_emojis: List[str] = []

    # 1. HTML entities & tags
    unescaped = html.unescape(text)
    if unescaped != text or re.search(r"<[^>]+>", unescaped):
        noise_tags.append("html_markup")
    text = re.sub(r"<[^>]+>", " ", unescaped)

    # 2. Zero-width & control characters
    cleaned_ctrl = re.sub(r"[\u200b-\u200f\ufeff\x00-\x08\x0b-\x1f\x7f]", "", text)
    if cleaned_ctrl != text:
        noise_tags.append("control_chars")
    text = cleaned_ctrl

    # 2b. Bracketed uppercase system/log metadata tags (e.g., [ALERT], [INFO], [TICKET-12])
    cleaned_log = re.sub(r"\[[A-Z][A-Z0-9_-]{1,18}\]\s*", " ", text)
    if cleaned_log != text:
        noise_tags.append("log_metadata_tags")
        text = cleaned_log

    # 3. URLs and email addresses
    if re.search(r"https?://\S+|www\.\S+|\S+@\S+\.\S+", text, flags=re.IGNORECASE):
        noise_tags.append("urls_and_handles")
        text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # 4. Social @mentions and #Hashtag decomposition
    if re.search(r"(?:^|\s)@[A-Za-z0-9_.]+", text):
        if "mentions" not in noise_tags:
            noise_tags.append("mentions")
        text = re.sub(r"(?<!\w)@[A-Za-z0-9_.]+", " ", text)

    if "#" in text:
        def _split_hashtag(match: re.Match) -> str:
            tag_body = match.group(1)
            # Split CamelCase hashtag: #BestDayEver -> Best Day Ever
            spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", tag_body)
            return " " + spaced.replace("_", " ") + " "

        new_text = re.sub(r"#([A-Za-z0-9_]+)", _split_hashtag, text)
        if new_text != text:
            noise_tags.append("hashtag_formatting")
            text = new_text

    # 5. Track ironic / slang emojis before ASCII folding
    for emoji_char, label in IRONIC_OR_SLANG_EMOJIS.items():
        if emoji_char in text:
            if emoji_char in IRONIC_EMOJI_KEYS and emoji_char not in ironic_emojis:
                ironic_emojis.append(emoji_char)

    for emoticon, label in ASCII_EMOTICONS.items():
        if emoticon in text:
            text = text.replace(emoticon, f" {label.replace('_', ' ')} ")

    # 6. Normalize elongated character noise (e.g. "soooooo goooood" -> "soo good")
    elongated = re.sub(r"([A-Za-z])\1{2,}", r"\1\1", text)
    if elongated != text:
        noise_tags.append("char_elongation")
    text = elongated

    # 7. Collapse excessive punctuation runs while retaining emphasis marker (!!!! -> !)
    punct_collapsed = re.sub(r"([!?.,]){2,}", r"\1", text)
    if punct_collapsed != text:
        noise_tags.append("punctuation_clutter")
    text = punct_collapsed

    # 8. Normalize smart quotes & collapse irregular whitespace
    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    collapsed_ws = re.sub(r"\s+", " ", text).strip()
    if len(collapsed_ws) < len(text):
        noise_tags.append("whitespace_jitter")

    return collapsed_ws, noise_tags, ironic_emojis


def semantic_chunk_text(
    clean_text: str,
    max_tokens: int = None,
    overlap_tokens: int = None,
) -> List[str]:
    """
    Sentence- and discourse-clause-aware semantic chunker.
    Splits contrast-bearing clauses (e.g. around 'but', 'however', 'although')
    so mixed-aspect evaluations keep distinct polarities cleanly separated.
    """
    max_toks = max_tokens or settings.max_chunk_tokens
    overlap = overlap_tokens or settings.chunk_overlap_tokens

    if not clean_text or not clean_text.strip():
        return []

    sentences = _safe_sent_tokenize(clean_text)
    clause_pattern = re.compile(
        r"\s*(?:,|;)?\s*\b(but|however|although|though|yet|whereas|while)\b\s+",
        flags=re.IGNORECASE,
    )

    clauses: List[str] = []
    for sent in sentences:
        parts = clause_pattern.split(sent)
        if len(parts) <= 1:
            clauses.append(sent)
            continue

        # Re-attach contrast marker to the second clause so nuance stays intact
        first = parts[0].strip(" ,;")
        if first:
            clauses.append(first)
        idx = 1
        while idx < len(parts):
            conj = parts[idx].strip()
            body = parts[idx + 1].strip(" ,;") if idx + 1 < len(parts) else ""
            combined = f"{conj} {body}".strip()
            if combined:
                clauses.append(combined)
            idx += 2

    chunks: List[str] = []
    for clause in clauses:
        words = clause.split()
        if len(words) <= max_toks:
            chunks.append(clause)
        else:
            step = max(1, max_toks - overlap)
            for start in range(0, len(words), step):
                sub = " ".join(words[start : start + max_toks])
                if sub:
                    chunks.append(sub)
                if start + max_toks >= len(words):
                    break

    return chunks if chunks else [clean_text]


def extract_features(raw_text: str, include_pos_tags: bool = True) -> FeatureExtractionResult:
    """
    Full custom NLP feature extraction pipeline:
      - Noise stripping & canonicalization
      - NLTK POS tagging & WordNet POS-aware lemmatization
      - Linguistic ambiguity cue extraction (negations, slang, contrast conjunctions, emojis)
      - Semantic chunking
    """
    original_len = len(raw_text or "")
    canonical_text, noise_tags, ironic_emojis = clean_raw_text(raw_text)

    # Tokenize words (keeping apostrophes inside contractions like n't)
    word_tokens_raw = _safe_word_tokenize(canonical_text)
    alpha_tokens = [t for t in word_tokens_raw if re.search(r"[A-Za-z0-9]", t)]

    tagged = _safe_pos_tag(alpha_tokens)
    pos_dist: Dict[str, int] = {
        "adjectives": 0,
        "adverbs": 0,
        "verbs": 0,
        "nouns": 0,
        "interjections": 0,
        "other": 0,
    }
    lemmas: List[str] = []
    pos_sample: List[POSToken] = []
    negations: List[str] = []
    slang_found: List[str] = []
    contrasts: List[str] = []

    lower_canonical = canonical_text.lower()
    for phrase in SLANG_MULTIWORD_PHRASES:
        if phrase in lower_canonical:
            slang_found.append(phrase)

    for tok, tag in tagged:
        low = tok.lower()
        group = _pos_group(tag)
        pos_dist[group] += 1

        try:
            lemma = _LEMMATIZER.lemmatize(low, pos=_wordnet_pos(tag))
        except Exception:
            lemma = low
        lemmas.append(lemma)

        if low in NEGATION_WORDS and low not in negations:
            negations.append(low)
        if low in CONTRAST_CONJUNCTIONS and low not in contrasts:
            contrasts.append(low)
        if low in SLANG_SINGLE_TERMS and low not in slang_found:
            slang_found.append(low)

        if include_pos_tags and len(pos_sample) < 40:
            pos_sample.append(
                POSToken(token=tok, tag=tag, pos_group=group, lemma=lemma)
            )

    semantic_chunks = semantic_chunk_text(canonical_text)
    cleaned_len = len(canonical_text)
    reduction_pct = (
        round((max(0, original_len - cleaned_len) / original_len) * 100.0, 2)
        if original_len > 0
        else 0.0
    )

    return FeatureExtractionResult(
        original_length=original_len,
        cleaned_length=cleaned_len,
        noise_reduction_pct=reduction_pct,
        noise_tags_removed=noise_tags,
        canonical_text=canonical_text,
        tokens=[t.lower() for t in alpha_tokens[:80]],
        lemmas=lemmas[:80],
        pos_distribution=pos_dist,
        pos_tags_sample=pos_sample,
        semantic_chunks=semantic_chunks,
        negation_terms=negations,
        slang_markers=slang_found,
        contrast_markers=contrasts,
        ironic_emojis=ironic_emojis,
    )


def measure_inconsistency_reduction(raw_noisy_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
    """
    Quantifies preprocessing inconsistency reduction across pairs of noisily formatted
    equivalent utterances (e.g. HTML tags, URLs, elongations, casing noise vs clean variant).
    Measures normalized token mismatch / edit inconsistency before vs after cleaning.
    """
    if not raw_noisy_pairs:
        return {"raw_inconsistency_rate": 0.0, "cleaned_inconsistency_rate": 0.0, "relative_reduction_pct": 0.0}

    raw_divergences = []
    clean_divergences = []

    for noisy_a, noisy_b in raw_noisy_pairs:
        toks_a_raw = set(noisy_a.split())
        toks_b_raw = set(noisy_b.split())
        union_raw = toks_a_raw | toks_b_raw
        raw_jacc_dist = 1.0 - (len(toks_a_raw & toks_b_raw) / len(union_raw)) if union_raw else 0.0
        raw_divergences.append(raw_jacc_dist)

        c_a, _, _ = clean_raw_text(noisy_a)
        c_b, _, _ = clean_raw_text(noisy_b)
        toks_a_clean = set(c_a.lower().split())
        toks_b_clean = set(c_b.lower().split())
        union_clean = toks_a_clean | toks_b_clean
        clean_jacc_dist = 1.0 - (len(toks_a_clean & toks_b_clean) / len(union_clean)) if union_clean else 0.0
        clean_divergences.append(clean_jacc_dist)

    raw_mean = sum(raw_divergences) / len(raw_divergences)
    clean_mean = sum(clean_divergences) / len(clean_divergences)
    reduction_pct = ((raw_mean - clean_mean) / raw_mean * 100.0) if raw_mean > 1e-8 else 0.0

    return {
        "raw_inconsistency_rate": round(raw_mean * 100.0, 2),
        "cleaned_inconsistency_rate": round(clean_mean * 100.0, 2),
        "relative_reduction_pct": round(max(0.0, reduction_pct), 2),
    }

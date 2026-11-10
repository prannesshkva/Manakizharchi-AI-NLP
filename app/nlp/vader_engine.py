import re
from dataclasses import dataclass
from typing import List

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

from app.core.config import settings
from app.models.schemas import ChunkSentiment, FeatureExtractionResult, SentimentLabel, VaderScores


_ANALYZER = SentimentIntensityAnalyzer()

# Classic sarcasm / irony templates where lexicon polarity flips against intended meaning
SARCASM_REGEX_PATTERNS = [
    re.compile(r"\boh\s+great\b", re.IGNORECASE),
    re.compile(r"\bthanks\s+for\s+nothing\b", re.IGNORECASE),
    re.compile(r"\bjust\s+what\s+i\s+needed\b", re.IGNORECASE),
    re.compile(r"\bbest\s+(?:update|day|decision|service|feature)\s+ever\b", re.IGNORECASE),
    re.compile(r"\byeah\s+(?:right|sure)\b", re.IGNORECASE),
    re.compile(r"\blove\s+(?:waiting|when\s+it\s+crashes|getting\s+ignored|losing)\b", re.IGNORECASE),
    re.compile(r"['\"](?:great|amazing|genius|helpful|awesome)['\"]", re.IGNORECASE),
]

NEGATIVE_EVENT_CONTEXT = {
    "crash", "crashed", "crashes", "bug", "bugs", "broken", "deleted", "waiting",
    "hold", "late", "delay", "delayed", "fail", "failed", "error", "frozen",
    "ignored", "deadline", "overtime", "queue", "reboot",
}


@dataclass
class Stage1Result:
    vader_scores: VaderScores
    initial_label: SentimentLabel
    confidence: float
    ambiguity_reasons: List[str]
    requires_stage2: bool
    chunk_sentiments: List[ChunkSentiment]


def _classify_vader_label(compound: float, pos: float, neg: float) -> SentimentLabel:
    if pos >= 0.16 and neg >= 0.16:
        return SentimentLabel.MIXED
    if compound >= 0.05:
        return SentimentLabel.POSITIVE
    if compound <= -0.05:
        return SentimentLabel.NEGATIVE
    return SentimentLabel.NEUTRAL


def evaluate_stage1_vader(
    raw_text: str,
    features: FeatureExtractionResult,
    confidence_threshold: float = None,
) -> Stage1Result:
    """
    Stage-1 Fast Lexicon Scoring & Confidence-Threshold Gating.
    Evaluates VADER scores on canonicalized text and semantic chunks, identifies
    lexical ambiguity triggers (slang, sarcasm, mixed clauses, weak polarity, OOV gaps),
    and calibrates Stage-1 confidence to decide fast-exit vs Stage-2 escalation.
    """
    threshold = confidence_threshold if confidence_threshold is not None else settings.confidence_threshold

    target_text = features.canonical_text or raw_text
    raw_scores = _ANALYZER.polarity_scores(target_text)
    vader_scores = VaderScores(
        positive=round(raw_scores["pos"], 4),
        neutral=round(raw_scores["neu"], 4),
        negative=round(raw_scores["neg"], 4),
        compound=round(raw_scores["compound"], 4),
    )

    chunk_sentiments: List[ChunkSentiment] = []
    chunk_compounds: List[float] = []
    for idx, chunk in enumerate(features.semantic_chunks):
        c_scores = _ANALYZER.polarity_scores(chunk)
        chunk_sentiments.append(
            ChunkSentiment(
                chunk_index=idx,
                text=chunk,
                compound=round(c_scores["compound"], 4),
                positive=round(c_scores["pos"], 4),
                negative=round(c_scores["neg"], 4),
                neutral=round(c_scores["neu"], 4),
            )
        )
        chunk_compounds.append(c_scores["compound"])

    ambiguity_reasons: List[str] = []
    penalty = 0.0

    # 1. Detect modern slang / code-mixed Tanglish lexicon gaps
    if features.slang_markers or features.ironic_emojis:
        ambiguity_reasons.append("slang_lexicon_gap")
        penalty += 0.34

    # 2. Detect sarcasm or irony patterns
    low_text = target_text.lower()
    sarcasm_pattern_hit = any(p.search(target_text) for p in SARCASM_REGEX_PATTERNS)
    positive_with_negative_event = (
        raw_scores["compound"] >= 0.10
        and any(neg_evt in features.lemmas or neg_evt in low_text for neg_evt in NEGATIVE_EVENT_CONTEXT)
    )
    if sarcasm_pattern_hit or positive_with_negative_event or any(e in {"🙄", "🤡", "😒", "🙃"} for e in features.ironic_emojis):
        ambiguity_reasons.append("sarcasm_or_irony_cue")
        penalty += 0.42

    # 3. Detect mixed-clause / opposing aspect polarity
    spread = (max(chunk_compounds) - min(chunk_compounds)) if len(chunk_compounds) >= 2 else 0.0
    both_poles_present = raw_scores["pos"] >= 0.16 and raw_scores["neg"] >= 0.16
    contrast_clause_conflict = bool(features.contrast_markers) and len(chunk_compounds) >= 2 and (
        spread >= 0.30 or (raw_scores["pos"] >= 0.12 and raw_scores["neg"] >= 0.12)
    )
    if spread >= 0.50 or both_poles_present or contrast_clause_conflict:
        ambiguity_reasons.append("mixed_clause_contrast")
        penalty += 0.32

    # 4. Weak or borderline VADER compound conviction
    abs_comp = abs(raw_scores["compound"])
    if abs_comp < 0.38:
        ambiguity_reasons.append("weak_vader_polarity")
        penalty += 0.16

    # 5. Expressive OOV gap (many adjectives/verbs/interjections yet VADER sees high neutrality)
    expressive_count = (
        features.pos_distribution.get("adjectives", 0)
        + features.pos_distribution.get("adverbs", 0)
        + features.pos_distribution.get("interjections", 0)
    )
    if raw_scores["neu"] >= 0.85 and expressive_count >= 2:
        ambiguity_reasons.append("oov_expressive_gap")
        penalty += 0.25

    # 6. Negation scope on borderline text
    if features.negation_terms and abs_comp < 0.50:
        ambiguity_reasons.append("negation_scope_shift")
        penalty += 0.18

    # Calibrated Stage-1 confidence score in [0.0, 1.0]
    raw_confidence = (
        0.58 * abs_comp
        + 0.28 * abs(raw_scores["pos"] - raw_scores["neg"])
        + 0.14 * (1.0 - raw_scores["neu"])
    )
    calibrated = max(0.0, min(1.0, raw_confidence - penalty))
    # Clear decisive lexicon signal boost when zero ambiguities are detected and |compound| >= 0.50
    if not ambiguity_reasons and abs_comp >= 0.50:
        calibrated = max(calibrated, min(0.98, 0.50 + 0.55 * abs_comp))
    calibrated = round(calibrated, 4)

    initial_label = _classify_vader_label(raw_scores["compound"], raw_scores["pos"], raw_scores["neg"])

    critical_ambiguities = {"slang_lexicon_gap", "sarcasm_or_irony_cue", "mixed_clause_contrast", "oov_expressive_gap"}
    has_critical = any(r in critical_ambiguities for r in ambiguity_reasons)
    requires_stage2 = (calibrated < threshold) or has_critical

    return Stage1Result(
        vader_scores=vader_scores,
        initial_label=initial_label,
        confidence=calibrated,
        ambiguity_reasons=ambiguity_reasons,
        requires_stage2=requires_stage2,
        chunk_sentiments=chunk_sentiments,
    )

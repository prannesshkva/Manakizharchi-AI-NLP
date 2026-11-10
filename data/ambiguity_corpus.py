import random
from typing import Any, Dict, List


SEED_AMBIGUITY_CORPUS: List[Dict[str, str]] = [
    # --- Modern Internet / Gen-Z Slang (Often mis-scored by classical lexicons) ---
    {"text": "this is straight fire tbh", "emotion_label": "Excited & Enthusiastic", "sentiment": "POSITIVE", "category": "slang_praise"},
    {"text": "im literally crying rn", "emotion_label": "Overwhelmed Joy or Hilarity", "sentiment": "POSITIVE", "category": "slang_praise"},
    {"text": "im literally dead lol", "emotion_label": "Extremely Amused / Laughing Hard", "sentiment": "POSITIVE", "category": "slang_praise"},
    {"text": "im dead skull emoji", "emotion_label": "Extremely Amused / Funny", "sentiment": "POSITIVE", "category": "slang_praise"},
    {"text": "no cap this is the best", "emotion_label": "Genuine & Enthusiastic", "sentiment": "POSITIVE", "category": "slang_praise"},
    {"text": "slay queen goated performance", "emotion_label": "Encouraging & Admiring", "sentiment": "POSITIVE", "category": "slang_praise"},
    {"text": "rent free in my head so good", "emotion_label": "Captivated & Fond", "sentiment": "POSITIVE", "category": "slang_praise"},
    {"text": "main character energy all day", "emotion_label": "Confident & Empowered", "sentiment": "POSITIVE", "category": "slang_praise"},
    {"text": "valid take completely agreed", "emotion_label": "Validating & Supportive", "sentiment": "POSITIVE", "category": "slang_praise"},
    {"text": "let them cook absolute banger", "emotion_label": "Impressed & Hyped", "sentiment": "POSITIVE", "category": "slang_praise"},
    {"text": "lowkey peak cinema hits different", "emotion_label": "Deeply Appreciative", "sentiment": "POSITIVE", "category": "slang_praise"},
    {"text": "sick beat insane production bro", "emotion_label": "Amazed & Impressed", "sentiment": "POSITIVE", "category": "slang_praise"},

    # --- Dismissive / Negative Slang ---
    {"text": "this is so mid tbh", "emotion_label": "Unimpressed & Disappointed", "sentiment": "NEGATIVE", "category": "slang_critique"},
    {"text": "womp womp tough luck", "emotion_label": "Dismissive & Mocking", "sentiment": "NEGATIVE", "category": "slang_critique"},
    {"text": "its giving desperate cringe", "emotion_label": "Judgmental & Disapproving", "sentiment": "NEGATIVE", "category": "slang_critique"},
    {"text": "bruh what is this mess", "emotion_label": "Confused & Frustrated", "sentiment": "NEGATIVE", "category": "slang_critique"},
    {"text": "we are cooked completely ruined", "emotion_label": "Anxious & Defeated", "sentiment": "NEGATIVE", "category": "slang_critique"},
    {"text": "touch grass get off the internet", "emotion_label": "Hostile & Dismissive", "sentiment": "NEGATIVE", "category": "slang_critique"},
    {"text": "caught in 4k exposed lying", "emotion_label": "Accusatory & Critical", "sentiment": "NEGATIVE", "category": "slang_critique"},
    {"text": "ratioed total L terrible take", "emotion_label": "Mocking & Negative", "sentiment": "NEGATIVE", "category": "slang_critique"},
    {"text": "delulu thinking not valid", "emotion_label": "Skeptical & Dismissive", "sentiment": "NEGATIVE", "category": "slang_critique"},

    # --- Code-Mixed Tamil / Tanglish (Manakizharchi Affective Domain) ---
    {"text": "sema mass performance bro", "emotion_label": "Thrilled & Celebratory", "sentiment": "POSITIVE", "category": "tanglish"},
    {"text": "vera level editing kalakkal", "emotion_label": "Amazed & Admiring", "sentiment": "POSITIVE", "category": "tanglish"},
    {"text": "ghilli level energy superu", "emotion_label": "Excited & Joyful", "sentiment": "POSITIVE", "category": "tanglish"},
    {"text": "padam full mokkai boring", "emotion_label": "Bored & Disappointed", "sentiment": "NEGATIVE", "category": "tanglish"},
    {"text": "konjam overa iruku honestly", "emotion_label": "Mildly Annoyed / Critical", "sentiment": "NEGATIVE", "category": "tanglish"},
    {"text": "mokka joke zero laugh", "emotion_label": "Unimpressed", "sentiment": "NEGATIVE", "category": "tanglish"},

    # --- Sarcasm & Irony Patterns ---
    {"text": "oh great another crash just what i needed", "emotion_label": "Sarcastic & Frustrated", "sentiment": "SARCASTIC", "category": "sarcasm"},
    {"text": "best update ever deleted all my work", "emotion_label": "Sarcastic & Furious", "sentiment": "SARCASTIC", "category": "sarcasm"},
    {"text": "love waiting 2 hours on hold with support", "emotion_label": "Ironic & Exasperated", "sentiment": "SARCASTIC", "category": "sarcasm"},
    {"text": "thanks for nothing super helpful as always", "emotion_label": "Bitterly Sarcastic", "sentiment": "SARCASTIC", "category": "sarcasm"},
    {"text": "yeah right amazing service after missing delivery", "emotion_label": "Cynical & Sarcastic", "sentiment": "SARCASTIC", "category": "sarcasm"},
    {"text": "genius design making the save button crash", "emotion_label": "Mocking & Sarcastic", "sentiment": "SARCASTIC", "category": "sarcasm"},

    # --- Mixed-Aspect / Contrastive Evaluations ---
    {"text": "great camera quality however battery drains terribly fast", "emotion_label": "Ambivalent / Mixed Critique", "sentiment": "MIXED", "category": "mixed_contrast"},
    {"text": "fast shipping but the packaging arrived damaged", "emotion_label": "Satisfied yet Frustrated", "sentiment": "MIXED", "category": "mixed_contrast"},
    {"text": "brilliant cinematography although the ending felt rushed", "emotion_label": "Appreciative with Reservations", "sentiment": "MIXED", "category": "mixed_contrast"},
    {"text": "nervous yet genuinely excited for tomorrow", "emotion_label": "Anticipatory & Bittersweet", "sentiment": "MIXED", "category": "mixed_contrast"},
    {"text": "UI looks sleek yet core export feature keeps failing", "emotion_label": "Balanced Critique", "sentiment": "MIXED", "category": "mixed_contrast"},
    {"text": "food tasted fantastic even though service took ages", "emotion_label": "Pleasant with Delay Critique", "sentiment": "MIXED", "category": "mixed_contrast"},

    # --- Neutral / Informational Anchors ---
    {"text": "the package will arrive on Tuesday morning", "emotion_label": "Objective / Matter-of-Fact", "sentiment": "NEUTRAL", "category": "neutral_anchor"},
    {"text": "meeting scheduled for 3pm in room 402", "emotion_label": "Informational / Neutral", "sentiment": "NEUTRAL", "category": "neutral_anchor"},
    {"text": "software version 2.4 released changelog notes", "emotion_label": "Neutral Update", "sentiment": "NEUTRAL", "category": "neutral_anchor"},
    {"text": "checking configuration settings on staging node", "emotion_label": "Neutral Operations", "sentiment": "NEUTRAL", "category": "neutral_anchor"},
    {"text": "received specification document for review yesterday", "emotion_label": "Informational / Neutral", "sentiment": "NEUTRAL", "category": "neutral_anchor"},

    # --- General Expressive Product/Software Anchors (VADER OOV coverage) ---
    {"text": "works smoothly and exceeded every single expectation", "emotion_label": "Satisfied & Delighted", "sentiment": "POSITIVE", "category": "product_praise"},
    {"text": "responsive intuitive dependable and quick shipping", "emotion_label": "Confident & Pleased", "sentiment": "POSITIVE", "category": "product_praise"},
    {"text": "hinge feels fragile however fast delivery", "emotion_label": "Balanced / Mixed Feedback", "sentiment": "MIXED", "category": "mixed_contrast"},
    {"text": "crisp screen fast speed yet battery drains quick", "emotion_label": "Ambivalent Critique", "sentiment": "MIXED", "category": "mixed_contrast"},
    {"text": "mobile app lags constantly and freezes repeatedly", "emotion_label": "Frustrated & Critical", "sentiment": "NEGATIVE", "category": "product_critique"},
]


def generate_benchmark_corpus(total_samples: int = 6250, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generates a deterministic, diverse benchmark dataset of 6,200+ unstructured inputs
    blending straightforward sentiment samples (~70%) and lexically ambiguous/noisy samples (~30%)
    across slang, sarcasm, mixed-aspect clauses, HTML/URL noise, and neutral items.
    """
    rng = random.Random(seed)

    clear_positive_templates = [
        "Absolutely loved the {item}! Best decision all year, super responsive and reliable.",
        "Fantastic {item} with incredible build quality and fast shipping!",
        "The {item} works smoothly and exceeded every single expectation I had.",
        "Highly recommend this {item} — intuitive, dependable, and great value.",
        "Wonderful support team and an amazingly crisp {item} experience!",
    ]
    clear_negative_templates = [
        "Terrible {item}. Completely broken out of the box and customer service ignored me.",
        "Worst {item} I have ever wasted money on. Utterly disappointing and buggy.",
        "Awful experience with this {item}, constant errors and horrible build quality.",
        "Avoid this {item} at all costs — slow, frustrating, and unusable.",
        "Miserable failure on the {item}, ruined my afternoon workflow completely.",
    ]
    slang_positive_templates = [
        "im literally dead 💀😂 this {item} is straight fire no cap",
        "lowkey goated {item} tbh, hits different 🔥💯",
        "sema mass {item} bro!! vera level quality kalakkal",
        "im literally crying 😭 this {item} is peak perfection valid af",
    ]
    slang_negative_templates = [
        "this {item} is honestly so mid womp womp",
        "we are cooked bruh 💀 this {item} is full mokkai cringe",
        "its giving broken {item}, konjam overa trash tbh",
    ]
    sarcastic_templates = [
        "Oh great, the {item} crashed again right before my deadline! Just what I needed 🙄",
        "Best {item} update ever — loved losing two hours of work 🤡",
        "Thanks for nothing!! Nothing says 'quality {item}' like freezing every 5 minutes.",
        "Yeah sure, totally 'seamless' {item} after failing three uploads 🙄",
    ]
    mixed_templates = [
        "The {item} design is gorgeous and fast, however the battery drains terribly quick.",
        "I appreciate the helpful {item} onboarding, but the pricing tier feels frustratingly rigid.",
        "Brilliant features in this {item}, although the mobile app lags constantly.",
        "Fast delivery of the {item}, yet the hinge feels worryingly fragile.",
    ]
    neutral_templates = [
        "Received the {item} specification document for Tuesday review.",
        "Updated {item} firmware version 2.1 yesterday afternoon.",
        "Checking the {item} configuration settings on the staging node.",
    ]

    items = [
        "laptop", "headset", "study platform", "camera lens", "API gateway",
        "analytics tool", "smartphone", "keyboard", "dashboard", "course module",
        "wireless mouse", "mechanical switch", "tablet stylus", "espresso maker",
        "mesh router", "cloud console", "microphone", "gaming monitor", "SSD drive",
        "code editor", "video exporter", "smartwatch", "charging dock", "search index",
        "vector database", "mobile banking app", "conference mic", "LED light bar",
        "webcam", "drawing pad",
    ]
    contexts = [
        "",
        "for daily workflow",
        "during semester projects",
        "after Tuesday's release",
        "on my workstation",
        "during remote sessions",
        "for studio recordings",
        "across client demos",
        "while traveling abroad",
    ]
    ambig_edge_templates = [
        ("Not terrible honestly, though the {item} could feel smoother.", "MIXED"),
        ("Lowkey disappointed — expected way better from this {item} ngl.", "NEGATIVE"),
        ("Honestly thrilled, weirdly addictive {item} experience!", "POSITIVE"),
        ("Yeah just what a busy Monday needed, another {item} outage.", "SARCASTIC"),
    ]
    noise_prefixes = ["", "<div>", "  [ALERT] ", "<p class='msg'>"]
    noise_suffixes = ["", " https://t.co/demo123", " @support_team #TechReview", "</div>   "]

    samples: List[Dict[str, Any]] = []
    for i in range(total_samples):
        item = items[i % len(items)]
        ctx = contexts[(i // len(items)) % len(contexts)]
        phrase = f"{item} {ctx}".strip()
        r = rng.random()

        if r < 0.35:
            text = rng.choice(clear_positive_templates).format(item=phrase)
            label = "POSITIVE"
            ambiguous = False
        elif r < 0.68:
            text = rng.choice(clear_negative_templates).format(item=phrase)
            label = "NEGATIVE"
            ambiguous = False
        elif r < 0.74:
            text = rng.choice(slang_positive_templates).format(item=phrase)
            label = "POSITIVE"
            ambiguous = True
        elif r < 0.80:
            text = rng.choice(slang_negative_templates).format(item=phrase)
            label = "NEGATIVE"
            ambiguous = True
        elif r < 0.87:
            text = rng.choice(sarcastic_templates).format(item=phrase)
            label = "SARCASTIC"
            ambiguous = True
        elif r < 0.93:
            text = rng.choice(mixed_templates).format(item=phrase)
            label = "MIXED"
            ambiguous = True
        elif r < 0.97:
            text = rng.choice(neutral_templates).format(item=phrase)
            label = "NEUTRAL"
            ambiguous = True
        else:
            tmpl, label = rng.choice(ambig_edge_templates)
            text = tmpl.format(item=phrase)
            ambiguous = True

        # Inject unstructured noise into 35% of inputs
        if rng.random() < 0.35:
            text = f"{rng.choice(noise_prefixes)}{text}{rng.choice(noise_suffixes)}"

        samples.append({"id": i, "text": text, "expected_label": label, "is_ambiguous": ambiguous})

    return samples

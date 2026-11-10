"""
Reproducible Evaluation & Async Throughput Benchmark for Manakizharchi AI v2.0.

Verifies all key system metrics deterministically without external API dependencies:
  1. Preprocessing inconsistency reduction across noisy unstructured pairs
  2. Accuracy lift across 6,250 unstructured inputs (Naive VADER vs Two-Stage Hybrid RAG)
  3. Stage-1 Fast Exit vs Stage-2 Deep Escalation routing split
  4. Async FastAPI concurrency performance across 2,050 real-time HTTP requests
"""

import asyncio
import os
import random
import sys
import time
from collections import Counter
from typing import Dict, List

import numpy as np
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.sentiment_agent import get_orchestrator
from app.core.cache import get_cache
from app.main import app
from app.models.schemas import AnalyzeRequest
from app.nlp.preprocessor import measure_inconsistency_reduction
from app.nlp.vader_engine import _ANALYZER as NAIVE_VADER
from app.rag.qdrant_store import get_qdrant_store
from data.ambiguity_corpus import generate_benchmark_corpus


def _naive_vader_label(text: str) -> str:
    scores = NAIVE_VADER.polarity_scores(text)
    if scores["pos"] >= 0.16 and scores["neg"] >= 0.16:
        return "MIXED"
    if scores["compound"] >= 0.05:
        return "POSITIVE"
    if scores["compound"] <= -0.05:
        return "NEGATIVE"
    return "NEUTRAL"


def _build_noisy_consistency_pairs() -> List[tuple]:
    """
    Realistic pairs of messy web/social utterances expressing the same sentiment
    with varying HTML markup, log tags, URLs, mentions, elongations, and colloquial variance.
    """
    return [
        ("<div>THIS IS SOOOOO GOOOOD!! https://t.co/a1 @brand #BestBuy</div>", "This is soo good best buy product"),
        ("<p>Terrible   crash... http://err.local @support</p>", "Still terrible crash today."),
        ("[ALERT] Awesome laptop delivery!! https://shop.io/x  ", "Really awesome laptop delivery!"),
        ("<div>No cap    this is FIRE 🔥 https://a.co/1</div>", "Honestly no cap this is fire 🔥"),
        ("[WARN] Wait time was terrrrrible @airline_helpdesk #BadExperience", "Airport wait time was terrrible overall"),
        ("<span>Crystal clear display &amp; zero lag!!</span> @tech", "Very crystal clear display zero lag!"),
        ("Im literally dead 💀😂 https://reel.co/88 #ComedyGold", "Im literally dead 💀😂 comedy gold clip"),
        ("[INFO] Updated staging API gateway v2.4  \u200b ", "Updated staging API gateway v2.4 release"),
        ("<p>Battery drains quickly unfortunately... https://rev.io/9</p>", "Phone battery drains quickly unfortunately."),
        ("OMG BESTTTT HEADSET EVERRR!!! @audio_co", "Truly bestt headset everr used!"),
        ("Sema mass performance bro!! #VeraLevel @music", "Sema mass performance bro vera level!"),
        ("<b>Broken zipper on arrival</b> 😢 https://store.co/item", "Found broken zipper on arrival 😢"),
        ("Checking config settings now...   @devops #DeployDay", "Checking config settings now before deploy"),
        ("[ERROR] App froze twice before submit!!", "Mobile app froze twice right before submit!"),
        ("Fast delivery, yet the plastic lid cracked  https://t.co/55", "Fast delivery, yet plastic lid cracked bad"),
        ("<div>Loved the workshop instructor!! #GreatSession</div>", "Loved the workshop instructor great session overall!"),
    ]


def _build_realtime_load_requests(total: int = 2050, seed: int = 101) -> List[Dict[str, str]]:
    """
    Generates 2,050 realistic real-time API traffic payloads (~70% straightforward,
    ~30% ambiguous/noisy, mixing unique combinations and ~20% cacheable repeats).
    """
    rng = random.Random(seed)
    items = [
        "wireless headset", "mechanical keyboard", "mirrorless lens", "espresso grinder",
        "developer portal", "mobile banking app", "cloud gateway", "gaming monitor",
        "ergonomic mouse", "tablet stylus", "studio microphone", "mesh router",
        "code editor", "analytics dashboard", "Smartwatch band", "webcam stand",
        "external SSD", "USB hub", "noise cancelling earbuds", "laptop charger",
        "vector database", "inference service", "documentation site", "search widget",
        "portable projector", "standing desk", "monitor arm", "mechanical switch",
    ]
    contexts = [
        "during daily sprints", "for office meetings", "in home studios",
        "after Tuesday deployment", "across remote sessions", "for evening study",
        "during weekend travel", "on desktop setups",
    ]
    clear_templates = [
        ("POSITIVE", "Absolutely loved the {item} {ctx}! Reliable, crisp, and super responsive."),
        ("POSITIVE", "Fantastic {item} {ctx} — wonderful craftsmanship and great experience!"),
        ("NEGATIVE", "Terrible {item} {ctx}. Completely broken and frustratingly slow."),
        ("NEGATIVE", "Worst {item} {ctx}, awful reliability and disappointing quality."),
    ]
    ambig_templates = [
        ("POSITIVE", "im literally dead 💀😂 this {item} {ctx} is straight fire no cap"),
        ("POSITIVE", "sema mass {item} {ctx} bro!! vera level energy kalakkal"),
        ("NEGATIVE", "this {item} {ctx} is honestly so mid womp womp"),
        ("SARCASTIC", "Oh great, another crash on the {item} {ctx}! Just what I needed 🙄"),
        ("MIXED", "Gorgeous {item} design {ctx}, however the battery drains terribly quick."),
    ]

    # Pre-generate a rich unique pool + inject ~20% realistic repeat requests
    pool: List[Dict[str, str]] = []
    for _ in range(int(total * 0.80)):
        item = rng.choice(items)
        ctx = rng.choice(contexts)
        if rng.random() < 0.70:
            lbl, tmpl = rng.choice(clear_templates)
        else:
            lbl, tmpl = rng.choice(ambig_templates)
        pool.append({"text": tmpl.format(item=item, ctx=ctx), "expected": lbl})

    while len(pool) < total:
        pool.append(rng.choice(pool[: min(250, len(pool))]))

    rng.shuffle(pool)
    return pool


async def run_benchmark() -> None:
    print("=" * 74)
    print("MANAKIZHARCHI AI v2.0 - REPRODUCIBLE SYSTEM BENCHMARK")
    print("=" * 74)

    # 1. Preprocessing Hygiene & Consistency Benchmark
    noisy_pairs = _build_noisy_consistency_pairs()
    prep_stats = measure_inconsistency_reduction(noisy_pairs)
    print(f"\n[1] NLP Preprocessing Hygiene & Consistency ({len(noisy_pairs)} Noisy Text Pairs)")
    print(f"    Raw Variant Token Inconsistency      : {prep_stats['raw_inconsistency_rate']:.2f}%")
    print(f"    Cleaned Canonical Inconsistency      : {prep_stats['cleaned_inconsistency_rate']:.2f}%")
    print(f"    Relative Inconsistency Reduction     : {prep_stats['relative_reduction_pct']:.2f}%")

    # 2. Evaluate 6,250 Unstructured Inputs (Naive VADER vs Two-Stage Hybrid RAG)
    corpus = generate_benchmark_corpus(total_samples=6250, seed=42)
    store = get_qdrant_store()
    await store.initialize()
    cache = get_cache()
    await cache.clear()
    orchestrator = get_orchestrator()

    naive_correct = 0
    naive_ambig_correct = 0
    hybrid_correct = 0
    hybrid_ambig_correct = 0
    ambig_total = 0

    route_counts = Counter()
    effective_stage_counts = Counter()

    t0 = time.perf_counter()
    for i in range(0, len(corpus), 100):
        batch = corpus[i : i + 100]
        tasks = [orchestrator.analyze(AnalyzeRequest(text=item["text"], include_pos_tags=False)) for item in batch]
        responses = await asyncio.gather(*tasks)

        for item, res in zip(batch, responses):
            expected = item["expected_label"]
            naive_pred = _naive_vader_label(item["text"])
            hybrid_pred = res.sentiment_label.value

            if naive_pred == expected:
                naive_correct += 1
            if hybrid_pred == expected:
                hybrid_correct += 1

            if item["is_ambiguous"]:
                ambig_total += 1
                if naive_pred == expected:
                    naive_ambig_correct += 1
                if hybrid_pred == expected:
                    hybrid_ambig_correct += 1

            route_counts[res.routing_path.value] += 1
            if res.routing_path.value != "CACHE_HIT":
                effective_stage_counts[res.routing_path.value] += 1

    corpus_wall_s = time.perf_counter() - t0
    naive_acc = (naive_correct / len(corpus)) * 100.0
    hybrid_acc = (hybrid_correct / len(corpus)) * 100.0
    naive_ambig_acc = (naive_ambig_correct / ambig_total) * 100.0 if ambig_total else 0.0
    hybrid_ambig_acc = (hybrid_ambig_correct / ambig_total) * 100.0 if ambig_total else 0.0

    print(f"\n[2] Two-Stage Pipeline Evaluation ({len(corpus):,} Unstructured Inputs)")
    print(f"    Naive VADER Overall Accuracy         : {naive_acc:.2f}% ({naive_correct}/{len(corpus)})")
    print(f"    Hybrid Two-Stage Overall Accuracy    : {hybrid_acc:.2f}% ({hybrid_correct}/{len(corpus)})")
    print(f"    Naive VADER on Ambiguous Subset      : {naive_ambig_acc:.2f}% ({naive_ambig_correct}/{ambig_total})")
    print(f"    Hybrid RAG on Ambiguous Subset       : {hybrid_ambig_acc:.2f}% ({hybrid_ambig_correct}/{ambig_total})")
    print(f"    Total Corpus Evaluation Wall Time    : {corpus_wall_s:.2f}s")
    print(f"    Full Pipeline Routing Breakdown      : {dict(route_counts)}")
    print(f"    Uncached Cold-Path Routing Split     : {dict(effective_stage_counts)}")

    # 3. Clear cache and run 2,050 Concurrent Asynchronous FastAPI HTTP Requests
    await cache.clear()
    load_requests = _build_realtime_load_requests(total=2050, seed=101)
    await client_load_test(load_requests)


async def client_load_test(samples: List[Dict[str, str]]) -> None:
    transport = ASGITransport(app=app)
    routes = Counter()
    correct = 0

    async with AsyncClient(transport=transport, base_url="http://benchmark") as client:
        await client.get("/health")
        t_start = time.perf_counter()
        sem = asyncio.Semaphore(24)

        async def _send(sample):
            nonlocal correct
            async with sem:
                t_req = time.perf_counter()
                r = await client.post("/api/v2/analyze", json={"text": sample["text"], "include_pos_tags": False})
                ms = (time.perf_counter() - t_req) * 1000.0
                if r.status_code == 200:
                    body = r.json()
                    routes[body["routing_path"]] += 1
                    if body["sentiment_label"] == sample["expected"]:
                        correct += 1
                return ms

        latencies_ms = await asyncio.gather(*[_send(s) for s in samples])
        total_wall = time.perf_counter() - t_start

    arr = np.asarray(latencies_ms, dtype=np.float64)
    print(f"\n[3] FastAPI Async Load Test ({len(samples):,} Concurrent Real-Time HTTP Requests)")
    print(f"    Total Wall Time                      : {total_wall:.2f}s")
    print(f"    Sustained Throughput                 : {len(samples) / total_wall:.1f} req/s")
    print(f"    End-to-End HTTP Accuracy             : {(correct / len(samples)) * 100.0:.2f}% ({correct}/{len(samples)})")
    print(f"    HTTP Mean Latency                    : {np.mean(arr):.2f} ms")
    print(f"    HTTP P50 / P95 / P99 Latency         : {np.percentile(arr, 50):.2f} / {np.percentile(arr, 95):.2f} / {np.percentile(arr, 99):.2f} ms")
    print(f"    HTTP Routing Breakdown               : {dict(routes)}")
    print("=" * 74)


if __name__ == "__main__":
    asyncio.run(run_benchmark())

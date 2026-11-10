import asyncio
import html
import re
import urllib.parse
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.agents.sentiment_agent import SentimentOrchestrator, get_orchestrator
from app.models.schemas import AnalyzeRequest, SentimentResponse, SocialAnalyzeRequest
from app.services.plotting import render_distribution_async


def _extract_instagram_shortcode(url: str) -> Optional[str]:
    match = re.search(r"instagram\.com/(?:p|reel|tv)/([A-Za-z0-9_-]+)", url or "")
    return match.group(1) if match else None


async def _fetch_public_instagram_metadata(post_url: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Asynchronously retrieves public Instagram post caption and author via public oEmbed
    or OpenGraph tags without using hardcoded credentials or blocking the event loop.
    """
    shortcode = _extract_instagram_shortcode(post_url)
    if not shortcode:
        return None, None, "invalid_url"

    clean_url = f"https://www.instagram.com/p/{shortcode}/"
    caption: Optional[str] = None
    author: Optional[str] = None
    source_note = "unavailable"

    async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
        # 1. Try public Instagram oEmbed endpoint
        try:
            oembed_url = f"https://www.instagram.com/api/v1/oembed/?url={urllib.parse.quote(clean_url)}"
            resp = await client.get(
                oembed_url,
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            )
            if resp.status_code == 200:
                data = resp.json()
                caption = data.get("title") or None
                author = data.get("author_name") or None
                if caption:
                    source_note = "instagram_oembed"
        except Exception:
            pass

        # 2. Fallback to public OpenGraph metatags
        if not caption:
            try:
                page = await client.get(
                    clean_url,
                    headers={"User-Agent": "Twitterbot/1.0", "Accept": "text/html"},
                )
                if page.status_code == 200:
                    og_desc = re.search(
                        r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']+)["\']',
                        page.text,
                    )
                    if og_desc:
                        caption = html.unescape(og_desc.group(1))
                        source_note = "opengraph_metadata"
                    og_title = re.search(
                        r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\']([^"\']+)["\']',
                        page.text,
                    )
                    if og_title and not author:
                        author = html.unescape(og_title.group(1))
            except Exception:
                pass

    return caption, author, source_note


async def analyze_social_engagement(
    req: SocialAnalyzeRequest,
    orchestrator: Optional[SentimentOrchestrator] = None,
) -> Dict[str, Any]:
    orch = orchestrator or get_orchestrator()
    caption = (req.caption or "").strip()
    author = "Creator"
    fetch_source = "user_provided_caption"

    if req.post_url and not caption:
        fetched_cap, fetched_auth, src = await _fetch_public_instagram_metadata(req.post_url)
        if fetched_cap:
            caption = fetched_cap
            fetch_source = src
        else:
            caption = (
                f"Social post engagement check for {req.post_url.strip()}. "
                "Public scrape restricted by platform anti-bot policy; provide caption or comments for full thread analysis."
            )
            fetch_source = "platform_restricted_fallback"
        if fetched_auth:
            author = fetched_auth

    # Analyze Caption
    caption_result = await orch.analyze(
        AnalyzeRequest(text=caption or "Social post engagement", include_pos_tags=False, include_plot=False)
    )

    # Concurrently analyze all provided audience comments through the two-stage pipeline
    raw_comments = [c.strip() for c in (req.comments or []) if c and c.strip()]
    comment_results: List[SentimentResponse] = []
    if raw_comments:
        sem = asyncio.Semaphore(24)

        async def _eval_comment(c_text: str) -> SentimentResponse:
            async with sem:
                return await orch.analyze(AnalyzeRequest(text=c_text, include_pos_tags=False, include_plot=False))

        comment_results = await asyncio.gather(*[_eval_comment(c) for c in raw_comments])

    all_evals = comment_results if comment_results else [caption_result]
    compounds = [r.vader_scores.compound for r in all_evals]
    avg_compound = sum(compounds) / len(compounds)

    label_counts = Counter(r.sentiment_label.value for r in all_evals)
    emotion_counts = Counter(em for r in all_evals for em in r.emotion_categories)
    stage2_used = sum(1 for r in all_evals if r.routing_path.value == "STAGE2_RAG_LLM_DEEP")

    top_emotions = [em for em, _ in emotion_counts.most_common(3)] or caption_result.emotion_categories
    dominant_sentiment = label_counts.most_common(1)[0][0]

    interpretation = (
        f"Audited {len(comment_results)} audience comment(s) + post caption ({fetch_source}). "
        f"Dominant sentiment is {dominant_sentiment} ({', '.join(top_emotions)}). "
        f"Stage-2 RAG escalated {stage2_used}/{len(all_evals)} ambiguous items."
        if comment_results
        else f"Analyzed post caption ({fetch_source}): {caption_result.interpretation}"
    )

    plot_b64 = await render_distribution_async(compounds) if req.include_plot else None

    return {
        "caption": caption,
        "author": author,
        "source_mode": fetch_source,
        "comments_analyzed": len(comment_results),
        "avg_compound_score": round(avg_compound, 4),
        "dominant_sentiment": dominant_sentiment,
        "emotion_categories": top_emotions,
        "sentiment_distribution": dict(label_counts),
        "stage2_escalations": stage2_used,
        "interpretation": interpretation,
        "caption_analysis": caption_result.model_dump(mode="json"),
        "comment_highlights": [
            {
                "text": req.comments[i],
                "sentiment": comment_results[i].sentiment_label.value,
                "routing": comment_results[i].routing_path.value,
                "confidence": comment_results[i].confidence,
                "interpretation": comment_results[i].interpretation,
            }
            for i in range(min(15, len(comment_results)))
        ],
        "plot": plot_b64,
    }

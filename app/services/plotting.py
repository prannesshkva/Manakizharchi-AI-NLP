import asyncio
import base64
import io
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.models.schemas import VaderScores


def _render_sentiment_bar_sync(scores: VaderScores, title: str = "Sentiment Breakdown") -> str:
    fig, ax = plt.subplots(figsize=(6.8, 3.6), facecolor="#0b0d13")
    ax.set_facecolor("#11141d")

    categories = ["Positive", "Neutral", "Negative"]
    values = [scores.positive * 100.0, scores.neutral * 100.0, scores.negative * 100.0]
    colors = ["#22c55e", "#f59e0b", "#ef4444"]

    bars = ax.bar(categories, values, color=colors, width=0.52, edgecolor="#1f2937", linewidth=1)
    ax.set_ylabel("Lexicon Share (%)", color="#cbd5e1", fontsize=10)
    ax.set_title(title, color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, 108)
    ax.tick_params(colors="#94a3b8", labelsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#334155")
    ax.grid(axis="y", color="#1e293b", linestyle="--", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 2.0,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            color="#f8fafc",
            fontsize=9.5,
            fontweight="bold",
        )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _render_distribution_sync(scores: List[float], title: str = "Audience Compound Sentiment Distribution") -> str:
    fig, ax = plt.subplots(figsize=(7.6, 4.0), facecolor="#0b0d13")
    ax.set_facecolor("#11141d")

    vals = scores if scores else [0.0]
    ax.hist(vals, bins=16, range=(-1.0, 1.0), color="#6366f1", edgecolor="#a5b4fc", alpha=0.82)
    avg = sum(vals) / len(vals)
    ax.axvline(avg, color="#f43f5e", linestyle="--", linewidth=2, label=f"Mean Compound: {avg:+.3f}")

    ax.set_title(title, color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Compound Polarity (-1.0 Negative -> +1.0 Positive)", color="#cbd5e1", fontsize=10)
    ax.set_ylabel("Count", color="#cbd5e1", fontsize=10)
    ax.tick_params(colors="#94a3b8")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#334155")
    ax.grid(axis="y", color="#1e293b", linestyle="--", alpha=0.6)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


async def render_sentiment_bar_async(scores: VaderScores, title: str = "Sentiment Breakdown") -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: _render_sentiment_bar_sync(scores, title))


async def render_distribution_async(scores: List[float], title: str = "Audience Sentiment Distribution") -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: _render_distribution_sync(scores, title))

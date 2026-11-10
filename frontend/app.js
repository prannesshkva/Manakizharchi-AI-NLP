document.addEventListener("DOMContentLoaded", () => {
  // Tab Navigation
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-tab");
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabPanels.forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${target}`).classList.add("active");
      if (target === "kb") {
        refreshMetrics();
        loadCorpus();
      }
    });
  });

  // Slider label
  const thresholdSlider = document.getElementById("optThreshold");
  const thresholdVal = document.getElementById("thresholdVal");
  thresholdSlider.addEventListener("input", () => {
    thresholdVal.textContent = parseFloat(thresholdSlider.value).toFixed(2);
  });

  const concurrencySlider = document.getElementById("optConcurrency");
  const concurrencyVal = document.getElementById("concurrencyVal");
  concurrencySlider.addEventListener("input", () => {
    concurrencyVal.textContent = concurrencySlider.value;
  });

  // Preset Chips
  document.querySelectorAll(".preset-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.getElementById("analyzeInput").value = chip.getAttribute("data-text");
      runSingleAnalysis();
    });
  });

  // Buttons
  document.getElementById("runAnalyzeBtn").addEventListener("click", runSingleAnalysis);
  document.getElementById("loadSampleBatchBtn").addEventListener("click", loadSampleBatch);
  document.getElementById("runBatchBtn").addEventListener("click", runBatchAnalysis);
  document.getElementById("loadSocialDemoBtn").addEventListener("click", loadSocialDemo);
  document.getElementById("runSocialBtn").addEventListener("click", runSocialAnalysis);
  document.getElementById("runIngestBtn").addEventListener("click", runIngest);
  document.getElementById("refreshMetricsBtn").addEventListener("click", refreshMetrics);

  refreshHealth();
});

function showToast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 3000);
}

async function refreshHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    document.getElementById("pillStatus").innerHTML = `<span class="dot pulse"></span> ${data.langgraph_enabled ? "LangGraph + FastAPI" : "FastAPI"} Ready`;
    document.getElementById("pillQdrant").textContent = `Qdrant: ${data.qdrant_points} pts (${data.embedding_dim}d)`;
    document.getElementById("pillCache").textContent = `Cache Hit: ${data.cache.hit_rate_pct}%`;
  } catch (e) {
    document.getElementById("pillStatus").textContent = "Offline";
  }
}

async function runSingleAnalysis() {
  const text = document.getElementById("analyzeInput").value.trim();
  if (!text) return showToast("Please enter text to analyze.");

  const btn = document.getElementById("runAnalyzeBtn");
  btn.disabled = true;

  try {
    const res = await fetch("/api/v2/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        force_stage2: document.getElementById("optForceStage2").checked,
        include_pos_tags: document.getElementById("optIncludePos").checked,
        confidence_threshold: parseFloat(document.getElementById("optThreshold").value),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    renderSingleAnalysis(data);
    refreshHealth();
  } catch (err) {
    showToast(`Error: ${err.message}`);
  } finally {
    btn.disabled = false;
  }
}

function renderSingleAnalysis(data) {
  document.getElementById("analyzeResults").classList.remove("hidden");

  const labelEl = document.getElementById("resLabel");
  labelEl.textContent = data.sentiment_label;
  labelEl.className = `verdict-label ${data.sentiment_label}`;
  document.getElementById("resConfidence").textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;

  const emotionsWrap = document.getElementById("resEmotions");
  emotionsWrap.innerHTML = data.emotion_categories.map((em) => `<span class="chip">${em}</span>`).join("");

  const routeBadge = document.getElementById("resRoutingPath");
  routeBadge.textContent = data.routing_path;
  routeBadge.className = `telemetry-badge ${data.routing_path}`;

  document.getElementById("resLatency").textContent = data.latency_ms.toFixed(2);
  document.getElementById("resProvider").textContent = data.llm_provider_used;

  const ambWrap = document.getElementById("resAmbiguityFlags");
  ambWrap.innerHTML = data.ambiguity_reasons.length
    ? data.ambiguity_reasons.map((r) => `<span class="mono-pill">${r}</span>`).join(" ")
    : `<span class="mono-pill">unambiguous</span>`;

  document.getElementById("resInterpretation").textContent = data.interpretation;

  // VADER bars
  const posPct = (data.vader_scores.positive * 100).toFixed(1) + "%";
  const neuPct = (data.vader_scores.neutral * 100).toFixed(1) + "%";
  const negPct = (data.vader_scores.negative * 100).toFixed(1) + "%";
  document.getElementById("resCompound").textContent = `compound: ${data.vader_scores.compound >= 0 ? "+" : ""}${data.vader_scores.compound.toFixed(3)}`;
  document.getElementById("barPosVal").textContent = posPct;
  document.getElementById("barNeuVal").textContent = neuPct;
  document.getElementById("barNegVal").textContent = negPct;
  document.getElementById("barPos").style.width = posPct;
  document.getElementById("barNeu").style.width = neuPct;
  document.getElementById("barNeg").style.width = negPct;

  // Chunks
  const chunkList = document.getElementById("chunkList");
  chunkList.innerHTML = (data.chunk_sentiments || [])
    .map(
      (c) => `<div class="chunk-item">
        <span>${escapeHtml(c.text)}</span>
        <span class="mono-pill">${c.compound >= 0 ? "+" : ""}${c.compound.toFixed(2)}</span>
      </div>`
    )
    .join("");

  // Preprocessing
  const f = data.features;
  document.getElementById("resNoisePct").textContent = `${f.noise_reduction_pct}% noise stripped`;
  document.getElementById("resCanonical").textContent = f.canonical_text;
  document.getElementById("resNoiseTags").innerHTML = f.noise_tags_removed.length
    ? f.noise_tags_removed.map((t) => `<span class="mono-pill cyan">stripped: ${t}</span>`).join("")
    : `<span class="mono-pill">clean input</span>`;

  document.getElementById("resSlang").innerHTML = [...f.slang_markers, ...f.ironic_emojis].length
    ? [...f.slang_markers, ...f.ironic_emojis].map((s) => `<span class="chip">${escapeHtml(s)}</span>`).join("")
    : `<span class="hint">None</span>`;
  document.getElementById("resNegations").innerHTML = f.negation_terms.length
    ? f.negation_terms.map((n) => `<span class="mono-pill">${escapeHtml(n)}</span>`).join("")
    : `<span class="hint">None</span>`;
  document.getElementById("resContrasts").innerHTML = f.contrast_markers.length
    ? f.contrast_markers.map((c) => `<span class="mono-pill">${escapeHtml(c)}</span>`).join("")
    : `<span class="hint">None</span>`;

  // POS tags
  const posWrap = document.getElementById("posTokensWrap");
  if (f.pos_tags_sample && f.pos_tags_sample.length) {
    posWrap.innerHTML = f.pos_tags_sample
      .map((p) => `<span class="pos-pill">${escapeHtml(p.token)}<span class="tag">${p.tag}</span></span>`)
      .join("");
  } else {
    posWrap.innerHTML = `<span class="hint">POS tags disabled or none extracted.</span>`;
  }

  // Qdrant RAG Evidence
  const ragCard = document.getElementById("ragEvidenceCard");
  const ragList = document.getElementById("ragMatchesList");
  if (data.retrieved_context && data.retrieved_context.length) {
    ragCard.classList.remove("hidden");
    ragList.innerHTML = data.retrieved_context
      .map(
        (m) => `<div class="rag-card">
          <div class="rag-card-top">
            <span class="chip">${m.sentiment}</span>
            <span class="mono-pill">${m.category}</span>
          </div>
          <div class="rag-quote">"${escapeHtml(m.text)}"</div>
          <div class="hint">${escapeHtml(m.emotion_label)}</div>
          <div class="rag-scores mt-2">RRF: ${m.rrf_score.toFixed(4)} | Qdrant Cos: ${m.dense_score.toFixed(3)} | BM25: ${m.sparse_score.toFixed(2)}</div>
        </div>`
      )
      .join("");
  } else {
    ragCard.classList.add("hidden");
  }
}

function loadSampleBatch() {
  const sampleLines = [
    "Absolutely loved the mechanical keyboard! Tactile, sturdy, and zero latency.",
    "Terrible customer support, waited 4 days with no reply and broken hardware.",
    "im literally dead 💀😂 this beat drop is straight fire no cap",
    "Oh great, another corrupted save file!! Best update ever 🙄",
    "The camera resolution is stunning, however the battery drains terribly quick.",
    "Bro sema mass delivery speed!! Vera level quality kalakkal",
    "this sequel is honestly so mid womp womp",
    "<div>LOVE THIS!! https://t.co/xyz @brand #AwesomeProduct</div>",
    "Worst software crash right before my presentation, complete waste.",
    "Incredible battery endurance and crystal clear OLED display!",
    "we are cooked bruh 💀 system deleted all staging configs",
    "Yeah sure 'seamless integration' after failing five times in a row 🤡",
    "Fast shipping arrived Monday morning in neat packaging.",
    "Brilliant screenplay although the final act felt noticeably rushed.",
    "No cap this study platform actually fixed my exam prep workflow!",
    "Disappointing plastic hinge, broke after two days of gentle use.",
  ];
  document.getElementById("batchInput").value = sampleLines.join("\n");
}

async function runBatchAnalysis() {
  const lines = document
    .getElementById("batchInput")
    .value.split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  if (!lines.length) return showToast("Enter at least one line for batch evaluation.");

  const btn = document.getElementById("runBatchBtn");
  btn.disabled = true;

  try {
    const res = await fetch("/api/v2/analyze/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        texts: lines,
        concurrency: parseInt(document.getElementById("optConcurrency").value, 10),
        include_pos_tags: false,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Batch error");

    document.getElementById("batchResults").classList.remove("hidden");
    document.getElementById("bTotal").textContent = data.total_inputs;
    document.getElementById("bFastPct").textContent = `${data.stage1_fast_pct}%`;
    document.getElementById("bAmbiguities").textContent = data.ambiguities_resolved;
    document.getElementById("bLatency").textContent = `${data.mean_latency_ms.toFixed(1)} / ${data.p95_latency_ms.toFixed(1)} ms`;
    document.getElementById("bWallTime").textContent = `${data.total_wall_time_ms.toFixed(1)} ms`;

    const tbody = document.getElementById("batchTableBody");
    tbody.innerHTML = data.items
      .map(
        (item, idx) => `<tr>
          <td>${idx + 1}</td>
          <td>${escapeHtml(item.features.canonical_text.slice(0, 65))}</td>
          <td><strong>${item.sentiment_label}</strong></td>
          <td class="mono">${(item.confidence * 100).toFixed(0)}%</td>
          <td><span class="telemetry-badge ${item.routing_path}">${item.routing_path}</span></td>
          <td>${item.ambiguity_reasons.join(", ") || "-"}</td>
          <td>${item.latency_ms.toFixed(2)} ms</td>
        </tr>`
      )
      .join("");
    refreshHealth();
  } catch (err) {
    showToast(`Batch failed: ${err.message}`);
  } finally {
    btn.disabled = false;
  }
}

function loadSocialDemo() {
  document.getElementById("socialUrl").value = "";
  document.getElementById("socialCaption").value =
    "Launched Manakizharchi AI v2.0 tonight — real-time confidence-gated sentiment intelligence built with FastAPI & Qdrant! 🚀";
  document.getElementById("socialComments").value = [
    "Bro sema mass launch!! Vera level latency 🔥",
    "im literally dead 💀😂 the sarcasm detector caught my boss's email",
    "Awesome throughput numbers, however setup docs could show Docker examples too",
    "Oh great another benchmark claim 🙄 let me actually test it... wait it's fast tbh",
    "Clean UI and super clear API responses, congratulations!",
    "this is straight fire no cap 💯",
  ].join("\n");
}

async function runSocialAnalysis() {
  const postUrl = document.getElementById("socialUrl").value.trim() || null;
  const caption = document.getElementById("socialCaption").value.trim() || null;
  const comments = document
    .getElementById("socialComments")
    .value.split("\n")
    .map((s) => s.trim())
    .filter(Boolean);

  if (!postUrl && !caption && !comments.length) {
    return showToast("Provide a post URL, caption, or audience comments.");
  }

  const btn = document.getElementById("runSocialBtn");
  btn.disabled = true;

  try {
    const res = await fetch("/api/v2/social/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        post_url: postUrl,
        caption,
        comments: comments.length ? comments : null,
        include_plot: true,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Social analysis failed");

    document.getElementById("socialResults").classList.remove("hidden");
    document.getElementById("sDominant").textContent = data.dominant_sentiment;
    document.getElementById("sCompound").textContent = `${data.avg_compound_score >= 0 ? "+" : ""}${data.avg_compound_score.toFixed(3)}`;
    document.getElementById("sCount").textContent = data.comments_analyzed;
    document.getElementById("sEscalations").textContent = data.stage2_escalations;
    document.getElementById("sInterpretation").textContent = data.interpretation;
    document.getElementById("sCaptionOut").textContent = data.caption;

    if (data.plot) {
      document.getElementById("sPlotImg").src = `data:image/png;base64,${data.plot}`;
    }

    document.getElementById("sCommentHighlights").innerHTML = (data.comment_highlights || [])
      .map(
        (c) => `<div class="highlight-row">
          <div>
            <div>"${escapeHtml(c.text)}"</div>
            <div class="hint">${escapeHtml(c.interpretation)}</div>
          </div>
          <div style="text-align:right; flex-shrink:0;">
            <span class="chip">${c.sentiment}</span>
            <div class="hint mt-2">${c.routing}</div>
          </div>
        </div>`
      )
      .join("");
    refreshHealth();
  } catch (err) {
    showToast(`Error: ${err.message}`);
  } finally {
    btn.disabled = false;
  }
}

async function runIngest() {
  const text = document.getElementById("ingestText").value.trim();
  const emotion_label = document.getElementById("ingestEmotion").value.trim() || "Custom Emotion";
  const sentiment = document.getElementById("ingestSentiment").value;
  const category = document.getElementById("ingestCategory").value.trim() || "custom_domain";

  if (!text) return showToast("Enter text to index into Qdrant.");

  try {
    const res = await fetch("/api/v2/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, emotion_label, sentiment, category }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Ingest failed");

    showToast(`Indexed ${data.chunks_indexed} chunk(s) into Qdrant! Total: ${data.total_corpus_size}`);
    document.getElementById("ingestText").value = "";
    loadCorpus();
    refreshHealth();
  } catch (err) {
    showToast(`Ingest error: ${err.message}`);
  }
}

async function loadCorpus() {
  try {
    const res = await fetch("/api/v2/corpus?limit=100");
    const data = await res.json();
    document.getElementById("corpusTotalBadge").textContent = `${data.total_indexed} vectors indexed`;
    document.getElementById("corpusList").innerHTML = data.items
      .map(
        (item) => `<div class="corpus-item">
          <div class="rag-card-top">
            <span class="chip">${item.sentiment}</span>
            <span class="mono-pill">${item.category}</span>
          </div>
          <div class="rag-quote">"${escapeHtml(item.text)}"</div>
          <div class="hint">${escapeHtml(item.emotion_label)}</div>
        </div>`
      )
      .join("");
  } catch (e) {}
}

async function refreshMetrics() {
  try {
    const res = await fetch("/api/v2/metrics");
    const data = await res.json();
    const t = data.telemetry;
    const c = data.cache;

    document.getElementById("liveMetricsBody").innerHTML = `
      <div class="tele-box"><span class="small-label">Total Requests</span><strong>${t.total_requests}</strong></div>
      <div class="tele-box"><span class="small-label">Cache Hit Rate</span><strong>${c.hit_rate_pct}% (${c.hits} hits)</strong></div>
      <div class="tele-box"><span class="small-label">Stage-1 Fast Share</span><strong>${t.routing_counts.stage1_fast_share_pct}%</strong></div>
      <div class="tele-box"><span class="small-label">Ambiguities Resolved</span><strong>${t.ambiguities_resolved}</strong></div>
      <div class="tele-box"><span class="small-label">Mean / P95 Latency</span><strong>${t.latency_ms.mean} / ${t.latency_ms.p95} ms</strong></div>
      <div class="tele-box"><span class="small-label">Noise Stripped Avg</span><strong>${t.preprocessing.avg_noise_reduction_pct}%</strong></div>
    `;
  } catch (e) {}
}

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

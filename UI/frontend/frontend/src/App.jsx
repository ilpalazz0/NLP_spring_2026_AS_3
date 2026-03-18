import { useState, useEffect, useRef } from "react";

const API = "http://localhost:5000";

const FEATURES  = ["CountVectorizer", "TF-IDF", "PMI"];
const MODELS    = ["RNN", "Bidirectional RNN", "LSTM"];
const FEAT_SHORT = { "CountVectorizer": "BoW", "TF-IDF": "TF-IDF", "PMI": "PMI" };

const PALETTE = {
  "CountVectorizer": "#6366f1",
  "TF-IDF":          "#06b6d4",
  "PMI":             "#f59e0b",
};

const MODEL_COLOR = {
  "RNN":               "#818cf8",
  "Bidirectional RNN": "#34d399",
  "LSTM":              "#fb923c",
};

/* ── helpers ── */
function getCell(results, model, feature, metric) {
  const r = results.find(x => x.model === model && x.feature === feature);
  return r ? r[metric] : null;
}

function heatColor(val) {
  if (val === null) return "#1e293b";
  const t = (val - 50) / 50;
  const r = Math.round(30  + t * (16  - 30));
  const g = Math.round(41  + t * (185 - 41));
  const b = Math.round(59  + t * (129 - 59));
  return `rgb(${r},${g},${b})`;
}

/* ── mini bar ── */
function Bar({ value, max = 100, color }) {
  return (
    <div style={{ background: "#0f172a", borderRadius: 4, overflow: "hidden", height: 6, width: "100%" }}>
      <div style={{
        width: `${(value / max) * 100}%`, height: "100%",
        background: color, borderRadius: 4,
        transition: "width .6s cubic-bezier(.4,0,.2,1)"
      }} />
    </div>
  );
}

/* ── confidence pill ── */
function Pill({ label, confidence }) {
  const pos = label === "Positive";
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 6,
      padding: "3px 10px", borderRadius: 99, fontSize: 12, fontWeight: 700,
      background: pos ? "rgba(16,185,129,.15)" : "rgba(239,68,68,.15)",
      color: pos ? "#34d399" : "#f87171",
      border: `1px solid ${pos ? "#34d399" : "#f87171"}33`,
    }}>
      <span>{pos ? "▲" : "▼"}</span>{label} {confidence}%
    </span>
  );
}

/* ══════════════════════════════════════════════ */
export default function App() {
  const [stats,       setStats]       = useState([]);
  const [metric,      setMetric]      = useState("accuracy");
  const [inputText,   setInputText]   = useState("");
  const [predictions, setPredictions] = useState([]);
  const [loading,     setLoading]     = useState(false);
  const [statsLoaded, setStatsLoaded] = useState(false);
  const [error,       setError]       = useState(null);
  const textRef = useRef(null);

  /* fetch pre-computed stats on mount */
  useEffect(() => {
    fetch(`${API}/stats`)
      .then(r => r.json())
      .then(d => { setStats(d.results); setStatsLoaded(true); })
      .catch(() => setError("Cannot reach backend. Is Flask running on port 5000?"));
  }, []);

  async function handlePredict() {
    if (!inputText.trim()) return;
    setLoading(true);
    setPredictions([]);
    try {
      const res  = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: inputText }),
      });
      const data = await res.json();
      setPredictions(data.predictions || []);
    } catch {
      setError("Prediction failed. Is Flask running?");
    } finally {
      setLoading(false);
    }
  }

  /* group predictions by feature */
  const byFeature = FEATURES.reduce((acc, f) => {
    acc[f] = predictions.filter(p => p.feature === f);
    return acc;
  }, {});

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0a0f1e",
      color: "#e2e8f0",
      fontFamily: "'IBM Plex Mono', 'Fira Code', monospace",
      padding: "40px 24px",
    }}>
      {/* google font */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&display=swap');
        * { box-sizing: border-box; }
        ::selection { background: #6366f144; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0f172a; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
        textarea:focus { outline: none; }
        button:hover { filter: brightness(1.15); }
        .fade-in { animation: fadeIn .4s ease; }
        @keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:none; } }
      `}</style>

      <div style={{ maxWidth: 1100, margin: "0 auto" }}>

        {/* ── Header ── */}
        <div style={{ marginBottom: 48 }}>
          <div style={{ fontSize: 11, letterSpacing: 4, color: "#6366f1", marginBottom: 8, textTransform: "uppercase" }}>
            Sentiment Analysis
          </div>
          <h1 style={{ fontSize: 32, fontWeight: 700, margin: 0, lineHeight: 1.2, letterSpacing: -1 }}>
            Model Performance Dashboard
          </h1>
          <p style={{ color: "#64748b", marginTop: 8, fontSize: 13 }}>
            RNN · Bidirectional RNN · LSTM  ×  CountVectorizer · TF-IDF · PMI · Sequence
          </p>
        </div>

        {error && (
          <div style={{
            background: "#7f1d1d22", border: "1px solid #f8717144",
            borderRadius: 8, padding: "12px 16px", marginBottom: 24, color: "#f87171", fontSize: 13
          }}>{error}</div>
        )}

        {/* ── Stats heatmap ── */}
        <section style={{ marginBottom: 48 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
            <h2 style={{ fontSize: 14, fontWeight: 700, margin: 0, textTransform: "uppercase", letterSpacing: 2 }}>
              {statsLoaded ? "Pre-trained Results" : "Loading statistics…"}
            </h2>
            <div style={{ display: "flex", gap: 8 }}>
              {["accuracy", "f1"].map(m => (
                <button key={m} onClick={() => setMetric(m)} style={{
                  padding: "5px 14px", borderRadius: 6, fontSize: 11, fontWeight: 600,
                  cursor: "pointer", border: "1px solid",
                  background:   metric === m ? "#6366f1" : "transparent",
                  borderColor:  metric === m ? "#6366f1" : "#334155",
                  color:        metric === m ? "#fff" : "#94a3b8",
                  textTransform: "uppercase", letterSpacing: 1,
                }}>
                  {m === "accuracy" ? "Accuracy" : "F1 Score"}
                </button>
              ))}
            </div>
          </div>

          {!statsLoaded ? (
            <div style={{ color: "#475569", fontSize: 13, padding: 32, textAlign: "center" }}>
              Connecting to backend…
            </div>
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table style={{ borderCollapse: "collapse", width: "100%", minWidth: 600 }}>
                <thead>
                  <tr>
                    <th style={{ width: 160, textAlign: "left", padding: "8px 12px", fontSize: 11, color: "#475569" }}>Model</th>
                    {FEATURES.map(f => (
                      <th key={f} style={{ textAlign: "center", padding: "8px 16px", fontSize: 11, color: PALETTE[f] }}>
                        {f}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {MODELS.map(model => (
                    <tr key={model}>
                      <td style={{
                        padding: "8px 12px", fontSize: 12, fontWeight: 600, color: MODEL_COLOR[model],
                        borderRight: "1px solid #1e293b",
                      }}>{model}</td>
                      {FEATURES.map(feat => {
                        const val = getCell(stats, model, feat, metric);
                        return (
                          <td key={feat} style={{
                            padding: "10px 16px", textAlign: "center", fontSize: 14, fontWeight: 700,
                            background: heatColor(val), color: "#f8fafc",
                            border: "2px solid #0a0f1e", borderRadius: 6,
                          }}>
                            {val !== null ? `${val}%` : "—"}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Legend */}
          {statsLoaded && (
            <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
              {[50,60,70,80,90,100].map(v => (
                <div key={v} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 10, color: "#475569" }}>
                  <div style={{ width: 12, height: 12, borderRadius: 3, background: heatColor(v) }} />
                  {v}%
                </div>
              ))}
            </div>
          )}
        </section>

        {/* ── Bar chart comparison ── */}
        {statsLoaded && (
          <section style={{ marginBottom: 48 }}>
            <h2 style={{ fontSize: 14, fontWeight: 700, marginBottom: 20, textTransform: "uppercase", letterSpacing: 2 }}>
              Feature Comparison by Model
            </h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16 }}>
              {MODELS.map(model => (
                <div key={model} style={{
                  background: "#0f172a", borderRadius: 12, padding: "20px 24px",
                  border: `1px solid ${MODEL_COLOR[model]}22`,
                }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: MODEL_COLOR[model], marginBottom: 16, letterSpacing: 1 }}>
                    {model}
                  </div>
                  {FEATURES.map(feat => {
                    const val = getCell(stats, model, feat, metric);
                    return (
                      <div key={feat} style={{ marginBottom: 12 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4, color: "#94a3b8" }}>
                          <span style={{ color: PALETTE[feat] }}>{FEAT_SHORT[feat]}</span>
                          <span style={{ fontWeight: 700, color: "#e2e8f0" }}>{val ?? "—"}%</span>
                        </div>
                        <Bar value={val ?? 0} color={PALETTE[feat]} />
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </section>
        )}

        {/* ── Text input ── */}
        <section style={{ marginBottom: 48 }}>
          <h2 style={{ fontSize: 14, fontWeight: 700, marginBottom: 16, textTransform: "uppercase", letterSpacing: 2 }}>
            Live Prediction
          </h2>
          <div style={{
            background: "#0f172a", borderRadius: 12, padding: 24,
            border: "1px solid #1e293b",
          }}>
            <textarea
              ref={textRef}
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter" && e.metaKey) handlePredict(); }}
              placeholder="Type a review or sentence to classify…"
              rows={4}
              style={{
                width: "100%", background: "#0a0f1e", border: "1px solid #334155",
                borderRadius: 8, padding: "14px 16px", color: "#e2e8f0",
                fontSize: 13, fontFamily: "inherit", resize: "vertical",
                lineHeight: 1.7,
              }}
            />
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 12 }}>
              <span style={{ fontSize: 11, color: "#475569" }}>⌘ + Enter to run</span>
              <button
                onClick={handlePredict}
                disabled={loading || !inputText.trim()}
                style={{
                  padding: "10px 28px", background: "#6366f1", border: "none",
                  borderRadius: 8, color: "#fff", fontFamily: "inherit",
                  fontSize: 12, fontWeight: 700, cursor: "pointer",
                  letterSpacing: 1, textTransform: "uppercase",
                  opacity: loading || !inputText.trim() ? 0.5 : 1,
                }}
              >
                {loading ? "Running…" : "Classify →"}
              </button>
            </div>
          </div>
        </section>

        {/* ── Prediction results ── */}
        {predictions.length > 0 && (
          <section className="fade-in">
            <h2 style={{ fontSize: 14, fontWeight: 700, marginBottom: 20, textTransform: "uppercase", letterSpacing: 2 }}>
              Prediction Results
            </h2>

            {/* consensus banner */}
            {(() => {
              const pos   = predictions.filter(p => p.label === "Positive").length;
              const total = predictions.length;
              const pct   = Math.round((pos / total) * 100);
              const isPos = pct >= 50;
              return (
                <div style={{
                  background: isPos ? "rgba(16,185,129,.08)" : "rgba(239,68,68,.08)",
                  border: `1px solid ${isPos ? "#34d39944" : "#f8717144"}`,
                  borderRadius: 10, padding: "16px 24px", marginBottom: 24,
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                }}>
                  <div>
                    <div style={{ fontSize: 11, color: "#64748b", marginBottom: 4, textTransform: "uppercase", letterSpacing: 2 }}>
                      Model Consensus
                    </div>
                    <div style={{ fontSize: 22, fontWeight: 700, color: isPos ? "#34d399" : "#f87171" }}>
                      {isPos ? "Positive" : "Negative"} Sentiment
                    </div>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <div style={{ fontSize: 32, fontWeight: 700, color: isPos ? "#34d399" : "#f87171" }}>{pct}%</div>
                    <div style={{ fontSize: 11, color: "#475569" }}>{pos}/{total} models agree</div>
                  </div>
                </div>
              );
            })()}

            {/* per-feature breakdown */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 16 }}>
              {FEATURES.map(feat => (
                <div key={feat} style={{
                  background: "#0f172a", borderRadius: 12, padding: "20px 24px",
                  border: `1px solid ${PALETTE[feat]}33`,
                }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: PALETTE[feat], marginBottom: 16, letterSpacing: 2, textTransform: "uppercase" }}>
                    {feat}
                  </div>
                  {(byFeature[feat] || []).map(p => (
                    <div key={p.model} style={{
                      marginBottom: 14, paddingBottom: 14,
                      borderBottom: "1px solid #1e293b",
                    }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                        <span style={{ fontSize: 11, color: MODEL_COLOR[p.model], fontWeight: 600 }}>{p.model}</span>
                        <Pill label={p.label} confidence={p.confidence} />
                      </div>
                      {/* prob bars */}
                      {Object.entries(p.probs).map(([cls, prob]) => (
                        <div key={cls} style={{ marginBottom: 4 }}>
                          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#475569", marginBottom: 2 }}>
                            <span>{cls === "1" ? "Positive" : "Negative"}</span>
                            <span>{prob}%</span>
                          </div>
                          <Bar value={prob} color={cls === "1" ? "#34d399" : "#f87171"} />
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </section>
        )}

        <footer style={{ marginTop: 64, borderTop: "1px solid #1e293b", paddingTop: 24, fontSize: 11, color: "#334155", textAlign: "center" }}>
          Models pre-trained · Flask API on :5000 · React on :3000
        </footer>
      </div>
    </div>
  );
}
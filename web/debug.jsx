const { useState, useEffect, useMemo, useCallback } = React;

function formatMs(ms) {
  if (ms === undefined || ms === null) return "0.0 ms";
  return `${Number(ms).toFixed(1)} ms`;
}

function formatTimestamp(isoStr) {
  if (!isoStr) return "";
  try {
    const d = new Date(isoStr);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }) + '.' + String(d.getMilliseconds()).padStart(3, '0');
  } catch (e) {
    return isoStr;
  }
}

function StatusBadge({ status }) {
  const s = (status || "").toLowerCase();
  if (s === "success") return <span className="badge badge-success">✓ Success</span>;
  if (s === "running") return <span className="badge badge-running">● Running</span>;
  if (s === "error") return <span className="badge badge-error">✕ Error</span>;
  return <span className="badge badge-direct">{status}</span>;
}

function RouteBadge({ route }) {
  const r = (route || "direct").toLowerCase();
  if (r === "legal") return <span className="badge badge-legal">§ Legal RAG</span>;
  if (r === "rate") return <span className="badge badge-rate">🏷 Rate Lookup</span>;
  if (r === "mixed") return <span className="badge badge-route">⚡ Mixed (Legal+Rate)</span>;
  if (r === "clarification") return <span className="badge badge-calc">❓ Clarification</span>;
  return <span className="badge badge-direct">⚙ Direct / Calc</span>;
}

// -----------------------------------------------------------------------------
// Main Application Component
// -----------------------------------------------------------------------------
function DebugApp() {
  const [traces, setTraces] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [selectedTrace, setSelectedTrace] = useState(null);
  const [activeTab, setActiveTab] = useState("waterfall");
  const [legalSubTab, setLegalSubTab] = useState("final");
  const [searchQuery, setSearchQuery] = useState("");
  const [routeFilter, setRouteFilter] = useState("all");
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  // Fetch list of recent traces
  const fetchTraces = useCallback(async () => {
    try {
      const res = await fetch("/debug/executions?limit=100");
      if (res.ok) {
        const data = await res.json();
        setTraces(data);
        if (data.length > 0 && !selectedId) {
          setSelectedId(data[0].execution_id);
        }
      }
    } catch (err) {
      console.error("Failed to fetch traces:", err);
    }
  }, [selectedId]);

  // Fetch full trace details when selectedId changes
  const fetchTraceDetail = useCallback(async (id) => {
    if (!id) return;
    setLoading(true);
    try {
      const res = await fetch(`/debug/executions/${id}`);
      if (res.ok) {
        const data = await res.json();
        setSelectedTrace(data);
      }
    } catch (err) {
      console.error("Failed to fetch trace detail:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchTraces();
  }, [fetchTraces]);

  useEffect(() => {
    if (selectedId) {
      fetchTraceDetail(selectedId);
    }
  }, [selectedId, fetchTraceDetail]);

  // Auto-refresh interval
  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(() => {
      fetchTraces();
      if (selectedId) {
        fetchTraceDetail(selectedId);
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [autoRefresh, selectedId, fetchTraces, fetchTraceDetail]);

  const handleClear = async () => {
    if (!window.confirm("Clear all recorded debug traces?")) return;
    try {
      await fetch("/debug/clear", { method: "POST" });
      setTraces([]);
      setSelectedId(null);
      setSelectedTrace(null);
    } catch (e) {
      console.error(e);
    }
  };

  const copyExecutionId = (id) => {
    navigator.clipboard.writeText(id);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const filteredTraces = useMemo(() => {
    return traces.filter((t) => {
      const matchesSearch =
        !searchQuery ||
        (t.query && t.query.toLowerCase().includes(searchQuery.toLowerCase())) ||
        (t.execution_id && t.execution_id.toLowerCase().includes(searchQuery.toLowerCase()));
      const matchesRoute =
        routeFilter === "all" ||
        (t.route && t.route.toLowerCase() === routeFilter.toLowerCase());
      return matchesSearch && matchesRoute;
    });
  }, [traces, searchQuery, routeFilter]);

  return (
    <div className="debug-layout">
      {/* 1. Sidebar */}
      <aside className="debug-sidebar">
        <div className="debug-sidebar-header">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 18 }}>🔍</span>
              <strong style={{ fontSize: 15, color: "#f1f7f9", letterSpacing: "-0.2px" }}>GST Bot Traces</strong>
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              <button
                onClick={fetchTraces}
                className="sub-tab-btn"
                title="Refresh list"
              >
                🔄
              </button>
              <button
                onClick={handleClear}
                className="sub-tab-btn"
                title="Clear all traces"
                style={{ color: "#f87171" }}
              >
                🗑
              </button>
            </div>
          </div>

          <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
            <input
              type="text"
              placeholder="Filter by query or trace ID..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                flex: 1,
                padding: "6px 10px",
                background: "#090d10",
                border: "1px solid #233039",
                borderRadius: 4,
                color: "#e6edf0",
                fontSize: 12,
              }}
            />
          </div>

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <select
              value={routeFilter}
              onChange={(e) => setRouteFilter(e.target.value)}
              style={{
                background: "#090d10",
                border: "1px solid #233039",
                borderRadius: 4,
                color: "#8e9fa5",
                fontSize: 11,
                padding: "4px 8px",
              }}
            >
              <option value="all">All Routes ({traces.length})</option>
              <option value="legal">Legal RAG</option>
              <option value="rate">Rate Lookup</option>
              <option value="mixed">Mixed</option>
              <option value="direct">Direct / Calc</option>
            </select>

            <label style={{ fontSize: 11, color: "#8e9fa5", display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
              Auto-refresh
            </label>
          </div>
        </div>

        <div className="debug-sidebar-list">
          {filteredTraces.length === 0 ? (
            <div style={{ padding: 24, textAlign: "center", color: "#64748b", fontSize: 13 }}>
              No execution traces recorded yet.<br />
              <small style={{ marginTop: 6, display: "block" }}>Run a query in the <a href="/" style={{ color: "#38bdf8" }}>Chat UI</a> to generate a trace.</small>
            </div>
          ) : (
            filteredTraces.map((t) => {
              const isSelected = t.execution_id === selectedId;
              return (
                <div
                  key={t.execution_id}
                  onClick={() => setSelectedId(t.execution_id)}
                  className={`trace-item ${isSelected ? "selected" : ""}`}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                    <RouteBadge route={t.route} />
                    <span style={{ fontSize: 11, color: "#8e9fa5", fontFamily: "var(--mono-font)" }}>
                      {formatTimestamp(t.timestamp)}
                    </span>
                  </div>
                  <div style={{ fontSize: 13, fontWeight: 500, color: "#f1f7f9", marginBottom: 6, lineHeight: 1.3, wordBreak: "break-word" }}>
                    {t.query || "(Empty query)"}
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11 }}>
                    <StatusBadge status={t.status} />
                    <span style={{ color: "#38bdf8", fontFamily: "var(--mono-font)", fontWeight: 500 }}>
                      {formatMs(t.total_timing_ms)}
                    </span>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </aside>

      {/* 2. Main Content Area */}
      <main className="debug-main">
        {selectedTrace ? (
          <>
            {/* Main Header */}
            <div className="debug-main-header">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
                <div>
                  <h2 style={{ margin: "0 0 6px 0", fontSize: 18, color: "#f8fafc", fontWeight: 600, letterSpacing: "-0.3px" }}>
                    {selectedTrace.query}
                  </h2>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 12, color: "#8e9fa5" }}>
                    <span>ID: <code style={{ color: "#cbd5e1", fontFamily: "var(--mono-font)" }}>{selectedTrace.execution_id}</code></span>
                    <button
                      onClick={() => copyExecutionId(selectedTrace.execution_id)}
                      className="sub-tab-btn"
                      style={{ padding: "2px 6px", fontSize: 11 }}
                    >
                      {copied ? "✓ Copied" : "Copy ID"}
                    </button>
                    <span>•</span>
                    <span>{formatTimestamp(selectedTrace.timestamp)}</span>
                    <span>•</span>
                    <a href="/" target="_blank" rel="noreferrer" style={{ color: "#38bdf8", textDecoration: "none", fontSize: 12 }}>
                      Open Chatbot ↗
                    </a>
                  </div>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <RouteBadge route={selectedTrace.planner?.clean_subqueries?.route || "direct"} />
                  <StatusBadge status={selectedTrace.status} />
                  <div style={{ textAlign: "right", padding: "6px 12px", background: "#161e24", border: "1px solid #233039", borderRadius: 6 }}>
                    <div style={{ fontSize: 10, textTransform: "uppercase", color: "#8e9fa5", fontWeight: 600 }}>Total Latency</div>
                    <div style={{ fontSize: 15, fontFamily: "var(--mono-font)", color: "#38bdf8", fontWeight: 600 }}>
                      {formatMs(selectedTrace.timings?.total_request_ms)}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Navigation Tabs */}
            <div className="debug-tabs-bar">
              <button
                className={`debug-tab-btn ${activeTab === "waterfall" ? "active" : ""}`}
                onClick={() => setActiveTab("waterfall")}
              >
                📊 Waterfall & Flow
              </button>
              <button
                className={`debug-tab-btn ${activeTab === "planner" ? "active" : ""}`}
                onClick={() => setActiveTab("planner")}
              >
                🧠 Planner Decisions
              </button>
              <button
                className={`debug-tab-btn ${activeTab === "graph" ? "active" : ""}`}
                onClick={() => setActiveTab("graph")}
              >
                🕸 LangGraph Execution
              </button>
              <button
                className={`debug-tab-btn ${activeTab === "legal" ? "active" : ""}`}
                onClick={() => setActiveTab("legal")}
              >
                § Legal Retrieval ({selectedTrace.legal_retrieval?.final_chunks?.length || 0})
              </button>
              <button
                className={`debug-tab-btn ${activeTab === "rate" ? "active" : ""}`}
                onClick={() => setActiveTab("rate")}
              >
                🏷 Rate Lookup ({selectedTrace.rate_retrieval?.returned_candidates?.length || 0})
              </button>
              <button
                className={`debug-tab-btn ${activeTab === "notifications" ? "active" : ""}`}
                onClick={() => setActiveTab("notifications")}
              >
                📜 Gazette Notifications ({selectedTrace.notification_retrieval?.returned_chunks?.length || 0})
              </button>
              <button
                className={`debug-tab-btn ${activeTab === "reasoning" ? "active" : ""}`}
                onClick={() => setActiveTab("reasoning")}
              >
                ⚖️ Grounded Reasoning {selectedTrace.reasoning?.grounded_reasoning_output ? "✓" : ""}
              </button>
              <button
                className={`debug-tab-btn ${activeTab === "calc" ? "active" : ""}`}
                onClick={() => setActiveTab("calc")}
              >
                🧮 Calculation {selectedTrace.calculation?.deterministic_result ? "✓" : ""}
              </button>
              <button
                className={`debug-tab-btn ${activeTab === "synthesis" ? "active" : ""}`}
                onClick={() => setActiveTab("synthesis")}
              >
                📝 Synthesis & Sources
              </button>
            </div>

            {/* Tab Contents */}
            <div className="debug-content-area">
              {activeTab === "waterfall" && (
                <WaterfallView trace={selectedTrace} />
              )}
              {activeTab === "planner" && (
                <PlannerView trace={selectedTrace} />
              )}
              {activeTab === "graph" && (
                <GraphExecutionView trace={selectedTrace} />
              )}
              {activeTab === "legal" && (
                <LegalRetrievalView trace={selectedTrace} subTab={legalSubTab} setSubTab={setLegalSubTab} />
              )}
              {activeTab === "rate" && (
                <RateRetrievalView trace={selectedTrace} />
              )}
              {activeTab === "notifications" && (
                <NotificationRetrievalView trace={selectedTrace} />
              )}
              {activeTab === "reasoning" && (
                <ReasoningView trace={selectedTrace} />
              )}
              {activeTab === "calc" && (
                <CalculationView trace={selectedTrace} />
              )}
              {activeTab === "synthesis" && (
                <SynthesisView trace={selectedTrace} />
              )}
            </div>
          </>
        ) : (
          <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "#64748b" }}>
            Select an execution trace from the sidebar to inspect its lifecycle.
          </div>
        )}
      </main>
    </div>
  );
}

// -----------------------------------------------------------------------------
// 1. Waterfall & Flow Tab View
// -----------------------------------------------------------------------------
function WaterfallView({ trace }) {
  const timings = trace.timings || {};
  const totalMs = timings.total_request_ms || 1.0;

  const steps = [
    { key: "planner_ms", label: "Planner", val: timings.planner_ms || 0, color: "#c084fc" },
    { key: "dense_ms", label: "Dense Retrieval (BGE-M3)", val: timings.dense_ms || 0, color: "#38bdf8" },
    { key: "bm25_ms", label: "BM25 Retrieval (pg_textsearch)", val: timings.bm25_ms || 0, color: "#60a5fa" },
    { key: "rrf_ms", label: "RRF Score Fusion", val: timings.rrf_ms || 0, color: "#818cf8" },
    { key: "reranker_ms", label: "Cross-Encoder Reranker", val: timings.reranker_ms || 0, color: "#a78bfa" },
    { key: "rate_lookup_ms", label: "Tariff Rate Lookup", val: timings.rate_lookup_ms || 0, color: "#fb923c" },
    { key: "notification_support_ms", label: "Supporting Gazette Notifications", val: timings.notification_support_ms || trace.graph_execution?.node_timings_ms?.["notification_support"] || 0, color: "#eab308" },
    { key: "grounded_reasoning_ms", label: "Grounded Reasoning", val: timings.grounded_reasoning_ms || 0, color: "#f472b6" },
    { key: "direct_reasoning_ms", label: "Direct Reasoning", val: timings.direct_reasoning_ms || 0, color: "#e879f9" },
    { key: "calculation_ms", label: "Deterministic Calculator", val: timings.calculation_ms || 0, color: "#facc15" },
    { key: "synthesis_ms", label: "LLM Grounded Synthesis", val: timings.synthesis_ms || 0, color: "#34d399" },
  ];

  const executedNodes = trace.graph_execution?.actual_executed_nodes || [];
  const nodeStatuses = trace.graph_execution?.node_status || {};
  const nodeTimings = trace.graph_execution?.node_timings_ms || {};

  return (
    <div>
      {/* Visual Execution Flow Pipeline */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">LangGraph Execution Flow Pipeline</div>
          <span style={{ fontSize: 12, color: "#8e9fa5" }}>
            Single-pass execution observed live (no rerun)
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12, overflowX: "auto", padding: "10px 0" }}>
          <FlowStepNode
            title="START"
            executed={true}
            status="success"
            ms={0}
          />
          <ArrowRight />
          <FlowStepNode
            title="planner"
            executed={executedNodes.includes("planner")}
            status={nodeStatuses["planner"]}
            ms={nodeTimings["planner"]}
          />
          <ArrowRight />
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <FlowStepNode
              title="legal_retrieval"
              executed={executedNodes.includes("legal_retrieval")}
              status={nodeStatuses["legal_retrieval"]}
              ms={nodeTimings["legal_retrieval"]}
            />
            <FlowStepNode
              title="rate_lookup"
              executed={executedNodes.includes("rate_lookup")}
              status={nodeStatuses["rate_lookup"]}
              ms={nodeTimings["rate_lookup"]}
            />
          </div>
          <ArrowRight />
          <FlowStepNode
            title="notification_support"
            executed={executedNodes.includes("notification_support")}
            status={nodeStatuses["notification_support"]}
            ms={nodeTimings["notification_support"]}
          />
          <ArrowRight />
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <FlowStepNode
              title="grounded_reasoning"
              executed={executedNodes.includes("grounded_reasoning")}
              status={nodeStatuses["grounded_reasoning"]}
              ms={nodeTimings["grounded_reasoning"]}
            />
            <FlowStepNode
              title="direct_reasoning"
              executed={executedNodes.includes("direct_reasoning")}
              status={nodeStatuses["direct_reasoning"]}
              ms={nodeTimings["direct_reasoning"]}
            />
          </div>
          <ArrowRight />
          <FlowStepNode
            title="calculation"
            executed={executedNodes.includes("calculation")}
            status={nodeStatuses["calculation"]}
            ms={nodeTimings["calculation"]}
          />
          <ArrowRight />
          <FlowStepNode
            title="synthesis"
            executed={executedNodes.includes("synthesis")}
            status={nodeStatuses["synthesis"]}
            ms={nodeTimings["synthesis"]}
          />
          <ArrowRight />
          <FlowStepNode
            title="END"
            executed={trace.status === "success"}
            status={trace.status}
            ms={0}
          />
        </div>
      </div>

      {/* Waterfall Timings Breakdown */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">Stage Latency Waterfall Breakdown</div>
          <span style={{ fontFamily: "var(--mono-font)", color: "#38bdf8", fontWeight: 600 }}>
            Total: {formatMs(totalMs)}
          </span>
        </div>

        <div style={{ padding: "8px 0" }}>
          {steps.map((s) => {
            const pct = Math.max(0, Math.min(100, (s.val / Math.max(totalMs, 1)) * 100));
            return (
              <div key={s.key} className="waterfall-bar-container">
                <div className="waterfall-bar-label">{s.label}</div>
                <div className="waterfall-bar-track">
                  <div
                    className="waterfall-bar-fill"
                    style={{
                      width: `${pct}%`,
                      background: s.color,
                      opacity: s.val > 0 ? 0.85 : 0.2,
                    }}
                  />
                </div>
                <div className="waterfall-bar-value">{formatMs(s.val)}</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function FlowStepNode({ title, executed, status, ms }) {
  const isOk = status === "success";
  const isErr = status === "error";
  const isSkip = status === "skipped" || (!executed && title !== "START" && title !== "END");

  let borderColor = "#2a3942";
  let bg = "#141b20";
  let textColor = "#8e9fa5";

  if (executed) {
    if (isOk) {
      borderColor = "#34d399";
      bg = "rgba(52, 211, 153, 0.08)";
      textColor = "#e6edf0";
    } else if (isErr) {
      borderColor = "#f87171";
      bg = "rgba(248, 113, 113, 0.08)";
      textColor = "#f87171";
    } else {
      borderColor = "#38bdf8";
      bg = "rgba(56, 189, 248, 0.08)";
      textColor = "#e6edf0";
    }
  } else if (isSkip) {
    borderColor = "#1e2830";
    bg = "#0d1114";
    textColor = "#475569";
  }

  return (
    <div
      style={{
        padding: "8px 14px",
        borderRadius: 6,
        border: `1px solid ${borderColor}`,
        background: bg,
        minWidth: 110,
        textAlign: "center",
      }}
    >
      <div style={{ fontSize: 12, fontWeight: 600, color: textColor }}>{title}</div>
      <div style={{ fontSize: 10, fontFamily: "var(--mono-font)", marginTop: 4, color: executed ? "#38bdf8" : "#475569" }}>
        {executed ? (ms ? formatMs(ms) : "active") : "bypassed"}
      </div>
    </div>
  );
}

function ArrowRight() {
  return <div style={{ color: "#334155", fontSize: 16, userSelect: "none" }}>➔</div>;
}

// -----------------------------------------------------------------------------
// 2. Planner Tab View
// -----------------------------------------------------------------------------
function PlannerView({ trace }) {
  const p = trace.planner || {};
  const flags = p.capability_flags || {};
  const subqueries = p.clean_subqueries || {};
  const premises = p.extracted_user_premises || {};
  const clarification = p.clarification_decision || {};

  return (
    <div>
      <div className="card">
        <div className="card-header">
          <div className="card-title">Planner Capability Decision Flags</div>
          <span style={{ fontSize: 12, color: "#8e9fa5" }}>
            Latency: {formatMs(p.timing_ms)}
          </span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12 }}>
          <FlagCard label="needs_legal" value={flags.needs_legal} desc="Statutory/Legal RAG retrieval" />
          <FlagCard label="needs_rate" value={flags.needs_rate} desc="Structured GST tariff lookup" />
          <FlagCard label="needs_grounded_reasoning" value={flags.needs_grounded_reasoning} desc="Reasons over retrieved evidence" />
          <FlagCard label="needs_direct_reasoning" value={flags.needs_direct_reasoning} desc="Direct calculation / arithmetic" />
          <FlagCard label="needs_calculation" value={flags.needs_calculation} desc="Mathematical tax / discount formula" />
          <FlagCard label="needs_clarification" value={flags.needs_clarification} desc="Query requires user clarification" />
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <div className="card-title">Clean Subqueries & Resolved Route</div>
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th style={{ width: 180 }}>Target Dimension</th>
              <th>Clean Extracted Subquery</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Resolved Route</strong></td>
              <td><RouteBadge route={subqueries.route} /></td>
            </tr>
            <tr>
              <td><strong>Legal Subquery</strong></td>
              <td><code style={{ color: "#38bdf8" }}>{subqueries.legal_query || "None"}</code></td>
            </tr>
            <tr>
              <td><strong>Rate Subquery</strong></td>
              <td><code style={{ color: "#fb923c" }}>{subqueries.rate_query || "None"}</code></td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="card">
        <div className="card-header">
          <div className="card-title">Extracted User Premises (Values & Assumptions)</div>
        </div>
        {Object.keys(premises).length === 0 ? (
          <div style={{ color: "#8e9fa5", fontSize: 13, padding: 8 }}>
            No explicit numerical premises extracted from query.
          </div>
        ) : (
          <pre className="code-block">{JSON.stringify(premises, null, 2)}</pre>
        )}
      </div>

      {clarification.needs_clarification && (
        <div className="card" style={{ borderColor: "#facc15" }}>
          <div className="card-header">
            <div className="card-title" style={{ color: "#facc15" }}>Clarification Required</div>
          </div>
          <div style={{ fontSize: 13, color: "#fef08a" }}>
            {clarification.clarification_prompt || "Query requires additional context before execution."}
          </div>
        </div>
      )}
    </div>
  );
}

function FlagCard({ label, value, desc }) {
  const isTrue = Boolean(value);
  return (
    <div
      style={{
        padding: 12,
        borderRadius: 6,
        background: isTrue ? "rgba(56, 189, 248, 0.08)" : "#0e1418",
        border: `1px solid ${isTrue ? "#38bdf8" : "#1f2b33"}`,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <strong style={{ fontSize: 13, color: isTrue ? "#38bdf8" : "#94a3b8" }}>{label}</strong>
        <span className={`badge ${isTrue ? "badge-success" : "badge-direct"}`}>
          {isTrue ? "TRUE" : "FALSE"}
        </span>
      </div>
      <div style={{ fontSize: 11, color: "#8e9fa5" }}>{desc}</div>
    </div>
  );
}

// -----------------------------------------------------------------------------
// 3. LangGraph Execution View
// -----------------------------------------------------------------------------
function GraphExecutionView({ trace }) {
  const g = trace.graph_execution || {};
  const selectedNodes = g.selected_nodes || [];
  const actualNodes = g.actual_executed_nodes || [];
  const order = g.execution_order || [];
  const statusMap = g.node_status || {};
  const timingMap = g.node_timings_ms || {};

  return (
    <div>
      <div className="card">
        <div className="card-header">
          <div className="card-title">Node Execution Summary</div>
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Node Name</th>
              <th>Selected by Planner</th>
              <th>Actually Executed</th>
              <th>Status</th>
              <th>Execution Latency</th>
            </tr>
          </thead>
          <tbody>
            {["planner", "rate_lookup", "legal_retrieval", "notification_support", "grounded_reasoning", "direct_reasoning", "calculation", "synthesis"].map((node) => {
              const isSelected = selectedNodes.includes(node) || node === "planner";
              const isExecuted = actualNodes.includes(node);
              const status = statusMap[node] || (isExecuted ? "success" : "skipped");
              const ms = timingMap[node];

              return (
                <tr key={node} className={isExecuted ? "highlight-row" : ""}>
                  <td><strong>{node}</strong></td>
                  <td>{isSelected ? <span style={{ color: "#34d399" }}>✓ Yes</span> : <span style={{ color: "#64748b" }}>– No</span>}</td>
                  <td>{isExecuted ? <span style={{ color: "#38bdf8" }}>✓ Yes</span> : <span style={{ color: "#64748b" }}>– Skipped</span>}</td>
                  <td><StatusBadge status={status} /></td>
                  <td style={{ fontFamily: "var(--mono-font)" }}>{isExecuted && ms !== undefined ? formatMs(ms) : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="card">
        <div className="card-header">
          <div className="card-title">Execution Order Sequence</div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          {order.map((node, i) => (
            <React.Fragment key={i}>
              <span className="badge badge-route" style={{ fontSize: 12, padding: "4px 10px" }}>
                {i + 1}. {node}
              </span>
              {i < order.length - 1 && <span style={{ color: "#475569" }}>➔</span>}
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
}

// -----------------------------------------------------------------------------
// 4. Legal Retrieval View (Hybrid Inspection)
// -----------------------------------------------------------------------------
function LegalRetrievalView({ trace, subTab, setSubTab }) {
  const l = trace.legal_retrieval || {};
  const query = l.retrieval_query || "";
  const finalChunks = l.final_chunks || [];
  const dense = l.dense_candidates || [];
  const bm25 = l.bm25_candidates || [];
  const rrf = l.rrf_candidates || [];
  const reranker = l.reranker_results || [];
  const timings = l.stages_timings_ms || {};

  return (
    <div>
      <div className="card">
        <div className="card-header">
          <div className="card-title">Legal Retrieval Stage Overview</div>
          <span style={{ fontSize: 12, color: "#8e9fa5" }}>
            Total Retrieval Latency: {formatMs(l.timing_ms)}
          </span>
        </div>
        <div style={{ marginBottom: 12 }}>
          <span style={{ color: "#8e9fa5", fontSize: 12 }}>Retrieval Query: </span>
          <code style={{ color: "#38bdf8", fontWeight: 600 }}>{query || "(None)"}</code>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 10 }}>
          <div style={{ padding: 10, background: "#0e1418", border: "1px solid #1f2b33", borderRadius: 4 }}>
            <div style={{ fontSize: 11, color: "#8e9fa5" }}>Dense (BGE-M3)</div>
            <div style={{ fontSize: 14, color: "#38bdf8", fontFamily: "var(--mono-font)" }}>{formatMs(timings.dense)}</div>
          </div>
          <div style={{ padding: 10, background: "#0e1418", border: "1px solid #1f2b33", borderRadius: 4 }}>
            <div style={{ fontSize: 11, color: "#8e9fa5" }}>BM25 Lexical</div>
            <div style={{ fontSize: 14, color: "#60a5fa", fontFamily: "var(--mono-font)" }}>{formatMs(timings.bm25)}</div>
          </div>
          <div style={{ padding: 10, background: "#0e1418", border: "1px solid #1f2b33", borderRadius: 4 }}>
            <div style={{ fontSize: 11, color: "#8e9fa5" }}>RRF Fusion</div>
            <div style={{ fontSize: 14, color: "#818cf8", fontFamily: "var(--mono-font)" }}>{formatMs(timings.rrf)}</div>
          </div>
          <div style={{ padding: 10, background: "#0e1418", border: "1px solid #1f2b33", borderRadius: 4 }}>
            <div style={{ fontSize: 11, color: "#8e9fa5" }}>Reranker</div>
            <div style={{ fontSize: 14, color: "#a78bfa", fontFamily: "var(--mono-font)" }}>{formatMs(timings.reranker)}</div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="sub-tabs">
          <button
            className={`sub-tab-btn ${subTab === "final" ? "active" : ""}`}
            onClick={() => setSubTab("final")}
          >
            Final Returned Chunks ({finalChunks.length})
          </button>
          <button
            className={`sub-tab-btn ${subTab === "dense" ? "active" : ""}`}
            onClick={() => setSubTab("dense")}
          >
            Dense Candidates ({dense.length})
          </button>
          <button
            className={`sub-tab-btn ${subTab === "bm25" ? "active" : ""}`}
            onClick={() => setSubTab("bm25")}
          >
            BM25 Candidates ({bm25.length})
          </button>
          <button
            className={`sub-tab-btn ${subTab === "rrf" ? "active" : ""}`}
            onClick={() => setSubTab("rrf")}
          >
            RRF Fused Candidates ({rrf.length})
          </button>
          <button
            className={`sub-tab-btn ${subTab === "reranker" ? "active" : ""}`}
            onClick={() => setSubTab("reranker")}
          >
            Reranker Results ({reranker.length})
          </button>
        </div>

        {subTab === "final" && <ChunksTable chunks={finalChunks} showScore={true} />}
        {subTab === "dense" && <CandidateList items={dense} scoreKey="score" rankKey="rank" label="Dense Similarity Score" />}
        {subTab === "bm25" && <CandidateList items={bm25} scoreKey="bm25_score" rankKey="bm25_rank" label="BM25 Lexical Score" />}
        {subTab === "rrf" && <CandidateList items={rrf} scoreKey="rrf_score" rankKey="rrf_rank" label="RRF Fused Score" />}
        {subTab === "reranker" && <CandidateList items={reranker} scoreKey="reranker_score" rankKey="reranker_rank" label="Cross-Encoder Score" />}
      </div>
    </div>
  );
}

function ChunksTable({ chunks, showScore }) {
  if (!chunks || chunks.length === 0) {
    return <div style={{ padding: 14, color: "#8e9fa5" }}>No chunks returned.</div>;
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {chunks.map((c, i) => {
        const docType = c.doc_type || c.document_type || "Legal";
        const docRef = c.doc_reference || c.section_number || c.rule_number || c.reference || "";
        const score = c.reranker_score !== undefined ? c.reranker_score : c.score;
        return (
          <div key={i} style={{ padding: 14, background: "#0e1418", border: "1px solid #1f2b33", borderRadius: 6 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span className="badge badge-legal">{docType}</span>
                <strong style={{ fontSize: 13, color: "#f1f7f9" }}>{docRef}</strong>
                {c.title && <span style={{ fontSize: 12, color: "#8e9fa5" }}>— {c.title}</span>}
              </div>
              {showScore && score !== undefined && (
                <span style={{ fontSize: 11, fontFamily: "var(--mono-font)", color: "#38bdf8" }}>
                  Score: {Number(score).toFixed(4)}
                </span>
              )}
            </div>
            <div style={{ fontSize: 12, lineHeight: 1.6, color: "#cbd5e1", whiteSpace: "pre-wrap", maxHeight: 200, overflowY: "auto" }}>
              {c.content || c.chunk_text || c.text}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function CandidateList({ items, scoreKey, rankKey, label }) {
  if (!items || items.length === 0) {
    return <div style={{ padding: 14, color: "#8e9fa5" }}>No candidates available.</div>;
  }
  return (
    <table className="data-table">
      <thead>
        <tr>
          <th style={{ width: 60 }}>Rank</th>
          <th style={{ width: 180 }}>Document Ref</th>
          <th>Content Snippet</th>
          <th style={{ width: 140 }}>{label}</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item, idx) => {
          const rank = item[rankKey] !== undefined ? item[rankKey] : idx + 1;
          const score = item[scoreKey] !== undefined ? item[scoreKey] : item.score;
          const ref = item.doc_reference || item.reference || item.section_number || item.id || `Doc #${idx + 1}`;
          const snippet = item.content || item.chunk_text || item.text || "";
          return (
            <tr key={idx}>
              <td style={{ fontFamily: "var(--mono-font)", color: "#8e9fa5" }}>#{rank}</td>
              <td><strong>{ref}</strong></td>
              <td style={{ fontSize: 12, color: "#94a3b8" }}>
                {snippet.slice(0, 140)}...
              </td>
              <td style={{ fontFamily: "var(--mono-font)", color: "#38bdf8" }}>
                {score !== undefined ? Number(score).toFixed(4) : "—"}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

// -----------------------------------------------------------------------------
// 5. Rate Retrieval View
// -----------------------------------------------------------------------------
function RateRetrievalView({ trace }) {
  const r = trace.rate_retrieval || {};
  const query = r.lookup_query || "";
  const candidates = r.returned_candidates || [];
  const selected = r.selected_rate_used_downstream;

  return (
    <div>
      <div className="card">
        <div className="card-header">
          <div className="card-title">Structured Tariff Rate Lookup</div>
          <span style={{ fontSize: 12, color: "#8e9fa5" }}>
            Lookup Latency: {formatMs(r.timing_ms)}
          </span>
        </div>
        <div style={{ marginBottom: 12 }}>
          <span style={{ color: "#8e9fa5", fontSize: 12 }}>Tariff Query: </span>
          <code style={{ color: "#fb923c", fontWeight: 600 }}>{query || "(None)"}</code>
        </div>
      </div>

      {selected && (
        <div className="card" style={{ borderColor: "#fb923c", background: "rgba(251, 146, 60, 0.05)" }}>
          <div className="card-header">
            <div className="card-title" style={{ color: "#fb923c" }}>
              ★ Selected Rate Used Downstream in Synthesis & Reasoning
            </div>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12 }}>
            <div>
              <div style={{ fontSize: 11, color: "#8e9fa5" }}>HSN / SAC Code</div>
              <strong style={{ fontSize: 15, color: "#f8fafc", fontFamily: "var(--mono-font)" }}>{selected.code || selected.hsn_code || "—"}</strong>
            </div>
            <div>
              <div style={{ fontSize: 11, color: "#8e9fa5" }}>Description</div>
              <strong style={{ fontSize: 13, color: "#f8fafc" }}>{selected.description || "—"}</strong>
            </div>
            <div>
              <div style={{ fontSize: 11, color: "#8e9fa5" }}>IGST Rate</div>
              <strong style={{ fontSize: 15, color: "#38bdf8", fontFamily: "var(--mono-font)" }}>{selected.igst_rate_pct ?? selected.rate ?? "—"}%</strong>
            </div>
            <div>
              <div style={{ fontSize: 11, color: "#8e9fa5" }}>CGST / SGST</div>
              <strong style={{ fontSize: 13, color: "#94a3b8", fontFamily: "var(--mono-font)" }}>
                {selected.cgst_rate_pct ?? (selected.rate ? selected.rate / 2 : "—")}% + {selected.sgst_rate_pct ?? (selected.rate ? selected.rate / 2 : "—")}%
              </strong>
            </div>
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-header">
          <div className="card-title">Returned Tariff Candidates Table</div>
        </div>
        {candidates.length === 0 ? (
          <div style={{ padding: 14, color: "#8e9fa5" }}>No structured rate records found.</div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>HSN / SAC</th>
                <th>Description</th>
                <th>IGST</th>
                <th>CGST</th>
                <th>SGST</th>
                <th>Cess</th>
                <th>Notification / Condition</th>
              </tr>
            </thead>
            <tbody>
              {candidates.map((item, idx) => {
                const isSel = selected && (selected.code === item.code || selected.id === item.id);
                return (
                  <tr key={idx} className={isSel ? "highlight-row" : ""}>
                    <td>
                      <strong style={{ fontFamily: "var(--mono-font)", color: isSel ? "#fb923c" : "#e6edf0" }}>
                        {item.code || item.hsn_code}
                      </strong>
                      {isSel && <span style={{ marginLeft: 6, color: "#fb923c" }}>★</span>}
                    </td>
                    <td style={{ maxWidth: 260 }}>{item.description}</td>
                    <td style={{ fontFamily: "var(--mono-font)", color: "#38bdf8" }}>{item.igst_rate_pct ?? item.rate ?? "—"}%</td>
                    <td style={{ fontFamily: "var(--mono-font)" }}>{item.cgst_rate_pct ?? "—"}%</td>
                    <td style={{ fontFamily: "var(--mono-font)" }}>{item.sgst_rate_pct ?? "—"}%</td>
                    <td style={{ fontFamily: "var(--mono-font)" }}>{item.cess ?? "0%"}</td>
                    <td style={{ fontSize: 11, color: "#8e9fa5" }}>{item.condition || item.notification || "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

// -----------------------------------------------------------------------------
// 5b. Supporting Notification Retrieval View (Gazette Records)
// -----------------------------------------------------------------------------
function NotificationRetrievalView({ trace }) {
  const n = trace.notification_retrieval || {};
  const query = n.semantic_query || "";
  const chunks = n.returned_chunks || [];
  const meta = n.support_metadata || {};
  const ms = n.timing_ms || trace.timings?.notification_support_ms || trace.graph_execution?.node_timings_ms?.["notification_support"] || 0;

  return (
    <div>
      <div className="card">
        <div className="card-header">
          <div className="card-title">Supporting Gazette Notification Vector Retrieval</div>
          <span style={{ fontSize: 12, color: "#8e9fa5" }}>
            Retrieval Latency: {formatMs(ms)}
          </span>
        </div>
        <div style={{ marginBottom: 12 }}>
          <span style={{ color: "#8e9fa5", fontSize: 12 }}>Semantic Gazette Query: </span>
          <code style={{ color: "#eab308", fontWeight: 600 }}>{query || "(None)"}</code>
        </div>
        {Object.keys(meta).length > 0 && (
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 8 }}>
            {meta.target_notification && (
              <span className="badge badge-route">Target: {meta.target_notification}</span>
            )}
            {meta.hsn_code && (
              <span className="badge badge-rate">HSN: {meta.hsn_code}</span>
            )}
            {meta.serial_no && (
              <span className="badge badge-calc">Entry S.No: {meta.serial_no}</span>
            )}
            {meta.schedule && (
              <span className="badge badge-legal">{meta.schedule}</span>
            )}
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-header">
          <div className="card-title">
            Retrieved Gazette Notification Chunks ({chunks.length})
          </div>
        </div>
        {chunks.length === 0 ? (
          <div style={{ padding: 14, color: "#8e9fa5" }}>No supporting notification chunks retrieved or deemed material.</div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {chunks.map((c, i) => {
              const notifNo = c.notification_number || c.source_metadata?.notification_number || "";
              const targetNotif = c.source_metadata?.target_notification;
              const opType = c.source_metadata?.operation_type;
              const taxTreat = c.source_metadata?.tax_treatment;
              const effDate = c.source_metadata?.effective_date;
              const score = c.reranker_score !== undefined ? c.reranker_score : c.score;

              return (
                <div key={i} style={{ padding: 16, background: "#0e1418", border: "1px solid #1f2b33", borderRadius: 6 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10, flexWrap: "wrap", gap: 8 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                      <span className="badge badge-rate">#{c.rank || i + 1}</span>
                      <strong style={{ fontSize: 14, color: "#f1f7f9" }}>{notifNo}</strong>
                      {targetNotif && (
                        <span className="badge badge-route" style={{ fontSize: 11 }}>Amends {targetNotif}</span>
                      )}
                      {opType && (
                        <span className="badge badge-legal" style={{ fontSize: 11 }}>{opType}</span>
                      )}
                      {taxTreat && (
                        <span className="badge badge-calc" style={{ fontSize: 11 }}>{taxTreat}</span>
                      )}
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                      {effDate && (
                        <span style={{ fontSize: 11, color: "#34d399", fontFamily: "var(--mono-font)" }}>
                          Effective: {effDate}
                        </span>
                      )}
                      {score !== undefined && (
                        <span style={{ fontSize: 12, fontFamily: "var(--mono-font)", color: "#eab308", fontWeight: 600 }}>
                          Score: {Number(score).toFixed(4)}
                        </span>
                      )}
                    </div>
                  </div>
                  <div style={{ fontSize: 11, color: "#8e9fa5", marginBottom: 8 }}>
                    Reference: <code>{c.reference || c.title || "Gazette Notification"}</code>
                    {c.chunk_id && <span style={{ marginLeft: 10, color: "#475569" }}>({c.chunk_id})</span>}
                  </div>
                  <div style={{ fontSize: 12, lineHeight: 1.6, color: "#cbd5e1", whiteSpace: "pre-wrap", maxHeight: 220, overflowY: "auto", background: "#080b0e", padding: 10, borderRadius: 4, border: "1px solid #141c22" }}>
                    {c.content || c.snippet}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// -----------------------------------------------------------------------------
// 6. Grounded Reasoning View
// -----------------------------------------------------------------------------
function ReasoningView({ trace }) {
  const r = trace.reasoning || {};
  const groundedOutput = r.grounded_reasoning_output;
  const directOutput = r.direct_reasoning_output;
  const calcInputs = r.calculation_inputs;

  return (
    <div>
      <div className="card">
        <div className="card-header">
          <div className="card-title">Grounded Evidence Reasoning (Statutory & Policy Grounding)</div>
          <span style={{ fontSize: 12, color: "#8e9fa5" }}>
            Latency: {formatMs(r.grounded_timing_ms)}
          </span>
        </div>
        <div style={{ fontSize: 12, color: "#8e9fa5", marginBottom: 12 }}>
          Analyzes retrieved legal provisions (e.g. ITC utilization under Sec 49 / Rule 88A) and verified tariff rates to determine scenario applicability and prepare structured parameters for the deterministic calculator. <em>Does not perform arithmetic itself.</em>
        </div>

        {groundedOutput ? (
          <div style={{ padding: 14, background: "#0e1418", border: "1px solid #1f2b33", borderRadius: 6, marginBottom: 14 }}>
            <div style={{ fontSize: 11, textTransform: "uppercase", color: "#f472b6", fontWeight: 600, marginBottom: 6 }}>
              Statutory Application & Policy Grounding
            </div>
            <div style={{ fontSize: 13, lineHeight: 1.7, color: "#e6edf0", whiteSpace: "pre-wrap" }}>
              {groundedOutput}
            </div>
          </div>
        ) : (
          <div style={{ padding: 14, color: "#8e9fa5", background: "#0e1418", border: "1px solid #1f2b33", borderRadius: 6, marginBottom: 14 }}>
            No grounded reasoning was required for this query.
          </div>
        )}

        {calcInputs && (
          <div>
            <div style={{ fontSize: 12, fontWeight: 600, color: "#facc15", marginBottom: 6 }}>
              Structured Calculation Inputs Prepared for Calculation Node:
            </div>
            <pre className="code-block">{JSON.stringify(calcInputs, null, 2)}</pre>
          </div>
        )}
      </div>

      {directOutput && (
        <div className="card">
          <div className="card-header">
            <div className="card-title">Retrieval-Free Direct Reasoning Output</div>
            <span style={{ fontSize: 12, color: "#8e9fa5" }}>
              Latency: {formatMs(r.direct_timing_ms)}
            </span>
          </div>
          <div style={{ fontSize: 13, lineHeight: 1.7, color: "#e6edf0", whiteSpace: "pre-wrap" }}>
            {directOutput}
          </div>
        </div>
      )}
    </div>
  );
}

// -----------------------------------------------------------------------------
// 7. Calculation View
// -----------------------------------------------------------------------------
function CalculationView({ trace }) {
  const c = trace.calculation || {};
  const res = c.deterministic_result;
  const inputs = c.calculation_inputs;

  return (
    <div>
      <div className="card">
        <div className="card-header">
          <div className="card-title">Deterministic Arithmetic & Reasoning</div>
          <span style={{ fontSize: 12, color: "#8e9fa5" }}>
            Latency: {formatMs(c.timing_ms)}
          </span>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 14, marginBottom: 14 }}>
          <div style={{ padding: 12, background: "#0e1418", border: "1px solid #1f2b33", borderRadius: 6 }}>
            <div style={{ fontSize: 11, color: "#8e9fa5" }}>Operation</div>
            <strong style={{ fontSize: 14, color: "#facc15" }}>{c.operation || "(None / Skipped)"}</strong>
          </div>
          <div style={{ padding: 12, background: "#0e1418", border: "1px solid #1f2b33", borderRadius: 6 }}>
            <div style={{ fontSize: 11, color: "#8e9fa5" }}>Tax Rate Applied</div>
            <strong style={{ fontSize: 14, color: "#38bdf8", fontFamily: "var(--mono-font)" }}>
              {c.rate_used !== null && c.rate_used !== undefined ? `${c.rate_used}%` : "None"}
            </strong>
          </div>
          <div style={{ padding: 12, background: "#0e1418", border: "1px solid #1f2b33", borderRadius: 6 }}>
            <div style={{ fontSize: 11, color: "#8e9fa5" }}>Source of Rate</div>
            <strong style={{ fontSize: 12, color: "#34d399" }}>{c.source_of_rate || "N/A"}</strong>
          </div>
        </div>
      </div>

      {res ? (
        <div className="card">
          <div className="card-header">
            <div className="card-title">Verified Calculation Result</div>
            <span className="badge badge-success">Deterministic Verifiable</span>
          </div>

          <div style={{ marginBottom: 16, padding: 14, background: "#162026", borderRadius: 6, border: "1px solid #233039" }}>
            <div style={{ fontSize: 12, color: "#8e9fa5" }}>Verified Final Result Value</div>
            <div style={{ fontSize: 24, fontWeight: 700, color: "#38bdf8", fontFamily: "var(--mono-font)", marginTop: 4 }}>
              ₹{Number(res.result_value || 0).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
          </div>

          {res.steps && res.steps.length > 0 && (
            <div style={{ marginBottom: 14 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#cbd5e1", marginBottom: 8 }}>Calculation Steps:</div>
              <ul style={{ margin: 0, paddingLeft: 20, color: "#94a3b8", fontSize: 12, lineHeight: 1.7 }}>
                {res.steps.map((step, idx) => (
                  <li key={idx} style={{ fontFamily: "var(--mono-font)" }}>{step}</li>
                ))}
              </ul>
            </div>
          )}

          {inputs && (
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#cbd5e1", marginBottom: 6 }}>Calculation Inputs:</div>
              <pre className="code-block">{JSON.stringify(inputs, null, 2)}</pre>
            </div>
          )}
        </div>
      ) : (
        <div className="card">
          <div style={{ color: "#8e9fa5", fontSize: 13 }}>
            No deterministic calculation was triggered for this query (pure legal or tariff inquiry).
          </div>
        </div>
      )}
    </div>
  );
}

// -----------------------------------------------------------------------------
// 7. Synthesis View
// -----------------------------------------------------------------------------
function SynthesisView({ trace }) {
  const s = trace.synthesis || {};
  const sources = s.sources_provided || [];

  return (
    <div>
      <div className="card">
        <div className="card-header">
          <div className="card-title">Grounded Synthesis Parameters</div>
          <span style={{ fontSize: 12, color: "#8e9fa5" }}>
            Generation Latency: {formatMs(s.generation_timing_ms)}
          </span>
        </div>
        <div style={{ display: "flex", gap: 18, alignItems: "center" }}>
          <div>
            <span style={{ fontSize: 11, color: "#8e9fa5" }}>Synthesis Model: </span>
            <strong style={{ fontSize: 13, color: "#38bdf8", fontFamily: "var(--mono-font)" }}>{s.model_name || "Deterministic Fallback"}</strong>
          </div>
          <div>
            <span style={{ fontSize: 11, color: "#8e9fa5" }}>Sources Provided: </span>
            <strong style={{ fontSize: 13, color: "#f8fafc" }}>{sources.length}</strong>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <div className="card-title">Final Grounded Answer</div>
        </div>
        <div style={{ fontSize: 13, lineHeight: 1.7, color: "#e2e8f0", whiteSpace: "pre-wrap" }}>
          {s.final_answer || "(No answer produced)"}
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <div className="card-title">Citations & Grounded Sources Provided to Model</div>
        </div>
        {sources.length === 0 ? (
          <div style={{ color: "#8e9fa5", fontSize: 13 }}>No sources injected.</div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {sources.map((src, i) => (
              <div key={i} style={{ padding: 10, background: "#0e1418", border: "1px solid #1f2b33", borderRadius: 4, fontSize: 12 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                  <span className="badge badge-route">{src.source_type || "Source"}</span>
                  <strong style={{ color: "#f1f7f9" }}>{src.title || src.reference || `Source #${i + 1}`}</strong>
                </div>
                {src.snippet && (
                  <div style={{ color: "#94a3b8", marginTop: 4, lineHeight: 1.4 }}>
                    {src.snippet}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// Mount the React Application
const rootEl = document.getElementById("root");
if (rootEl) {
  const root = ReactDOM.createRoot(rootEl);
  root.render(<DebugApp />);
}

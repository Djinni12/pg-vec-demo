const { useMemo, useState, useRef, useEffect } = React;

const SUGGESTIONS = [
  "What is the GST rate on computer monitors (HSN 8471)?",
  "How can GST registration be cancelled under Section 29?",
  "What is the GST rate for restaurant and outdoor catering services?",
  "What is the tax rate on milk and how to claim input tax credit under Section 16?",
  "What are the grounds for cancellation under Rule 21?",
];

function MarkdownRenderer({ content, isStreaming }) {
  if (!content) return isStreaming ? <span className="streaming-cursor">▊</span> : null;
  const lines = content.split("\n");
  const elements = [];
  let listItems = [];
  let inList = false;
  let listType = "ul";

  function flushList() {
    if (!inList) return;
    if (listType === "ol") {
      elements.push(
        <ol key={`ol-${elements.length}`} className="ans-list">
          {listItems.map((item, i) => (
            <li key={i}>{formatInline(item)}</li>
          ))}
        </ol>
      );
    } else {
      elements.push(
        <ul key={`ul-${elements.length}`} className="ans-list">
          {listItems.map((item, i) => (
            <li key={i}>{formatInline(item)}</li>
          ))}
        </ul>
      );
    }
    listItems = [];
    inList = false;
  }

  function formatInline(text) {
    const parts = text.split(/(\*\*.*?\*\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return <strong key={i}>{part.slice(2, -2)}</strong>;
      }
      return part;
    });
  }

  lines.forEach((line, idx) => {
    const trimmed = line.trim();
    if (!trimmed) {
      flushList();
      return;
    }

    if (trimmed.startsWith("### ")) {
      flushList();
      elements.push(<h4 key={idx} className="ans-h4">{formatInline(trimmed.slice(4))}</h4>);
    } else if (trimmed.startsWith("## ")) {
      flushList();
      elements.push(<h3 key={idx} className="ans-h3">{formatInline(trimmed.slice(3))}</h3>);
    } else if (trimmed.startsWith("# ")) {
      flushList();
      elements.push(<h2 key={idx} className="ans-h2">{formatInline(trimmed.slice(2))}</h2>);
    } else if (/^\d+\.\s/.test(trimmed)) {
      if (inList && listType !== "ol") flushList();
      inList = true;
      listType = "ol";
      listItems.push(trimmed.replace(/^\d+\.\s/, ""));
    } else if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
      if (inList && listType !== "ul") flushList();
      inList = true;
      listType = "ul";
      listItems.push(trimmed.slice(2));
    } else {
      flushList();
      elements.push(<p key={idx} className="ans-p">{formatInline(trimmed)}</p>);
    }
  });
  flushList();

  return (
    <div className="answer-text">
      {elements}
      {isStreaming && <span className="streaming-cursor">▊</span>}
    </div>
  );
}

function ChatMessage({ message }) {
  if (message.sender === "user") {
    return (
      <div className="message-row user-row">
        <div className="user-bubble">
          <p>{message.text}</p>
          <span className="msg-time">{message.time}</span>
        </div>
      </div>
    );
  }

  const { data, text, isStreaming } = message;

  return (
    <div className="message-row bot-row">
      <div className="bot-avatar">⚖️</div>
      <div className="bot-bubble">
        <div className="bot-header">
          <span className="bot-name">GST Bot</span>
          <span className="msg-time">{message.time}</span>
        </div>

        {/* Clean, user-facing grounded legal answer */}
        <div className="bot-content">
          <MarkdownRenderer content={text || data?.answer || ""} isStreaming={isStreaming} />
        </div>
      </div>
    </div>
  );
}

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [threadId, setThreadId] = useState(null);
  const threadIdRef = useRef(null);
  const messagesEndRef = useRef(null);
  const typingTimerRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "auto" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  useEffect(() => {
    return () => {
      if (typingTimerRef.current) {
        clearInterval(typingTimerRef.current);
      }
    };
  }, []);

  function handleClearChat() {
    if (typingTimerRef.current) {
      clearInterval(typingTimerRef.current);
      typingTimerRef.current = null;
    }
    setMessages([]);
    setThreadId(null);
    threadIdRef.current = null;
    setLoading(false);
  }

  async function sendMessage(textToSend) {
    const query = (textToSend || input).trim();
    if (!query || loading) return;

    if (typingTimerRef.current) {
      clearInterval(typingTimerRef.current);
      typingTimerRef.current = null;
    }

    const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const userMsg = { id: Date.now(), sender: "user", text: query, time };
    const botMsgId = Date.now() + 1;
    const initialBotMsg = {
      id: botMsgId,
      sender: "bot",
      text: "",
      isStreaming: true,
      data: null,
      time,
    };

    setMessages((prev) => [...prev, userMsg, initialBotMsg]);
    setInput("");
    setLoading(true);
    setError("");

    let targetText = "";
    let displayedText = "";
    let messageData = null;
    let sseDone = false;

    // Smooth typewriter pacing ticker (cadence inspired by ChatGPT and Claude)
    typingTimerRef.current = setInterval(() => {
      if (displayedText.length < targetText.length) {
        const remaining = targetText.length - displayedText.length;
        // Adaptive step size:
        // - Single character cadence for short deltas (~50 chars/sec, human-like typing)
        // - Smooth acceleration for longer bursts to maintain responsive flow
        let step = 1;
        if (remaining > 180) {
          step = Math.min(Math.ceil(remaining / 20), 12);
        } else if (remaining > 80) {
          step = 4;
        } else if (remaining > 25) {
          step = 2;
        } else {
          step = 1;
        }

        displayedText = targetText.slice(0, displayedText.length + step);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === botMsgId
              ? { ...m, text: displayedText, data: messageData, isStreaming: true }
              : m
          )
        );
      } else if (sseDone) {
        // Entire stream has been smoothly typed out to the end
        if (typingTimerRef.current) {
          clearInterval(typingTimerRef.current);
          typingTimerRef.current = null;
        }
        setMessages((prev) =>
          prev.map((m) =>
            m.id === botMsgId
              ? {
                  ...m,
                  text: targetText,
                  data: messageData,
                  isStreaming: false,
                }
              : m
          )
        );
        setLoading(false);
      }
    }, 20);

    try {
      const payload = { query, top_k: 10 };
      if (threadIdRef.current) {
        payload.thread_id = threadIdRef.current;
      }

      // Stream tokens in real-time over SSE
      const response = await fetch("/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `Server returned ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data: ")) continue;
          const jsonStr = trimmed.slice(6);
          try {
            const event = jsonStr ? JSON.parse(jsonStr) : null;
            if (!event) continue;

            if (event.thread_id && !threadIdRef.current) {
              threadIdRef.current = event.thread_id;
              setThreadId(event.thread_id);
            }

            if (event.type === "meta") {
              messageData = { ...event };
              setMessages((prev) =>
                prev.map((m) => (m.id === botMsgId ? { ...m, data: messageData } : m))
              );
            } else if (event.type === "token") {
              targetText += event.delta;
            } else if (event.type === "done") {
              targetText = event.answer || targetText;
              messageData = {
                ...(messageData || {}),
                generation_timing: event.generation_timing,
                total_timing: event.total_timing,
                timings_ms: event.timings_ms,
                answer: targetText,
              };
              sseDone = true;
            } else if (event.type === "error") {
              throw new Error(event.error);
            }
          } catch (e) {
            console.error("SSE parse error:", e);
          }
        }
      }

      sseDone = true;
    } catch (err) {
      if (typingTimerRef.current) {
        clearInterval(typingTimerRef.current);
        typingTimerRef.current = null;
      }
      setError(err.message);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === botMsgId
            ? {
                ...m,
                isStreaming: false,
                text: `⚠️ Error: ${err.message}`,
              }
            : m
        )
      );
      setLoading(false);
    } finally {
      if (!typingTimerRef.current) {
        setLoading(false);
      }
    }
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="chat-app-container">
      {/* Top Navigation Bar */}
      <header className="chat-navbar">
        <div className="brand-group">
          <div className="brand-icon">🏛️</div>
          <div>
            <h1>GST Legal Bot</h1>
            <p className="status-sub">
              <span className="online-indicator"></span> CGST Act, Rules, Forms & Rate Schedules
            </p>
          </div>
        </div>
        {messages.length > 0 && (
          <button type="button" className="btn-clear" onClick={handleClearChat}>
            Clear Chat
          </button>
        )}
      </header>

      {/* Main Chat Stream */}
      <section className="chat-stream">
        {messages.length === 0 ? (
          <div className="welcome-hero">
            <div className="hero-badge">AI LEGAL ASSISTANT</div>
            <h2>What would you like to know about GST?</h2>
            <p>
              Ask any question regarding GST rates, cancellation, registration, rules, forms, or procedures.
              Answers are grounded with citations from official GST legal and tariff records.
            </p>
            <div className="suggestion-chips">
              {SUGGESTIONS.map((sug, i) => (
                <button
                  key={i}
                  type="button"
                  className="suggestion-chip"
                  onClick={() => sendMessage(sug)}
                >
                  {sug}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="messages-list">
            {messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} />
            ))}
          </div>
        )}

        <div ref={messagesEndRef} />
      </section>

      {/* Sticky Bottom Input Bar */}
      <footer className="chat-composer-container">
        {error && <div className="composer-error">{error}</div>}
        <div className="chat-composer">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a GST law or procedure question... (Press Enter to send)"
            rows={1}
            disabled={loading}
          />
          <button
            type="button"
            className="btn-send"
            onClick={() => sendMessage()}
            disabled={loading || !input.trim()}
          >
            {loading ? "..." : "Send ↑"}
          </button>
        </div>
      </footer>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

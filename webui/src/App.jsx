import { useEffect, useState, useMemo } from "react";

const API_BASE = "http://localhost:8000";

function MessageBubble({ role, text, meta, onFeedback }) {
  return (
    <div className={`bubble ${role}`}>
      <div>{text}</div>
      {meta && (
        <div className="meta">
          {meta.prompt_id ? `Prompt: ${meta.prompt_id}` : ""}{" "}
          {meta.search_type ? `| Search: ${meta.search_type}` : ""}
        </div>
      )}
      {role === "assistant" && onFeedback && (
        <div className="feedback">
          <button onClick={() => onFeedback(true)}>👍</button>
          <button onClick={() => onFeedback(false)}>👎</button>
        </div>
      )}
    </div>
  );
}

function App() {
  const [prompts, setPrompts] = useState([]);
  const [promptId, setPromptId] = useState("");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/api/prompts`)
      .then((res) => res.json())
      .then((data) => {
        setPrompts(data.prompts || []);
        if (data.prompts && data.prompts.length > 0) {
          setPromptId(data.prompts[0].id);
        }
      })
      .catch(() => {});
  }, []);

  const promptOptions = useMemo(
    () =>
      prompts.map((p) => (
        <option key={p.id} value={p.id}>
          {p.label || p.id}
        </option>
      )),
    [prompts]
  );

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMsg = { role: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    setInput("");

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMsg.text, prompt_id: promptId || null }),
      });
      const data = await res.json();
      if (res.ok) {
        const assistantMsg = {
          role: "assistant",
          text: data.answer,
          meta: { prompt_id: data.prompt_id, search_type: data.search_type },
        };
        setMessages((prev) => [...prev, assistantMsg]);
      } else {
        const assistantMsg = {
          role: "assistant",
          text: data.detail || "Error from server",
          meta: { prompt_id: promptId },
        };
        setMessages((prev) => [...prev, assistantMsg]);
      }
    } catch (err) {
      const assistantMsg = {
        role: "assistant",
        text: "Network error",
        meta: { prompt_id: promptId },
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } finally {
      setLoading(false);
    }
  };

  const sendFeedback = async (index, helpful) => {
    const msg = messages[index - 1]; // user msg
    const assistant = messages[index]; // assistant msg
    if (!assistant || assistant.role !== "assistant") return;

    try {
      await fetch(`${API_BASE}/api/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: msg?.text || "",
          prompt_id: assistant.meta?.prompt_id || promptId,
          search_type: assistant.meta?.search_type,
          helpful,
        }),
      });
    } catch (err) {
      // ignore
    }
  };

  return (
    <div className="app">
      <div className="card">
        <div className="header">
          <div className="title">RAG Web UI</div>
          <div className="prompt-select">
            <span className="small">Prompt</span>
            <select value={promptId} onChange={(e) => setPromptId(e.target.value)}>
              {promptOptions}
            </select>
          </div>
        </div>

        <div className="chat">
          {messages.map((m, idx) => (
            <MessageBubble
              key={idx}
              role={m.role}
              text={m.text}
              meta={m.meta}
              onFeedback={
                m.role === "assistant" ? (val) => sendFeedback(idx, val) : undefined
              }
            />
          ))}
          {messages.length === 0 && (
            <div className="meta">Ask a question to search your knowledge base.</div>
          )}
        </div>

        <div className="input-row">
          <input
            type="text"
            value={input}
            placeholder="Ask something..."
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
          />
          <button onClick={sendMessage} disabled={loading}>
            {loading ? "..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;


import { useEffect, useRef, useState } from "react";

const API_URL = "http://localhost:8000";

function Icon({ name, size = 20 }) {
  const paths = {
    plus: <path d="M12 5v14M5 12h14" />,
    send: (
      <>
        <path d="m22 2-7 20-4-9-9-4Z" />
        <path d="M22 2 11 13" />
      </>
    ),
    mic: (
      <>
        <rect x="9" y="2" width="6" height="12" rx="3" />
        <path d="M5 10a7 7 0 0 0 14 0M12 17v5M8 22h8" />
      </>
    ),
    bot: (
      <>
        <rect x="3" y="7" width="18" height="14" rx="4" />
        <path d="M12 3v4M8 13h.01M16 13h.01M8 17h8" />
      </>
    ),
    panel: (
      <>
        <rect x="3" y="4" width="18" height="16" rx="2" />
        <path d="M9 4v16" />
      </>
    ),
    chevron: <path d="m7 10 5 5 5-5" />,
    spark: (
      <path d="m12 3 2.2 6.8L21 12l-6.8 2.2L12 21l-2.2-6.8L3 12l6.8-2.2Z" />
    ),
    logout: (
      <>
        <path d="M10 17l5-5-5-5" />
        <path d="M15 12H3" />
        <path d="M12 3h6a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-6" />
      </>
    ),
  };

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name] || paths.spark}
    </svg>
  );
}

async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    credentials: "include",
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...options.headers,
    },
  });

  const data =
    response.status === 204 ? null : await response.json();

  if (!response.ok) {
    const error = new Error(
      data?.detail || "The request could not be completed."
    );
    error.status = response.status;
    throw error;
  }

  return data;
}

export default function ChatApp({ onSessionExpired }) {
  const [mode, setMode] = useState("agentic");
  const [input, setInput] = useState("");
  const [chats, setChats] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [listening, setListening] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [modeOpen, setModeOpen] = useState(false);

  const recognitionRef = useRef(null);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);
  const sessionExpiredRef = useRef(false);

  const activeChat = chats.find((chat) => chat.id === activeId);
  const messages = activeChat?.messages ?? [];

  // Handle expired sessions from inside the component.
  async function request(path, options = {}) {
    try {
      return await apiRequest(path, options);
    } catch (err) {
      if (err.status === 401 && !sessionExpiredRef.current) {
        sessionExpiredRef.current = true;
        onSessionExpired?.();
      }
      throw err;
    }
  }

  // Load this user's saved conversations.
  useEffect(() => {
    let cancelled = false;

    async function loadConversations() {
      try {
        const data = await request("/conversations");

        if (cancelled) return;

        const savedChats = data.conversations.map((conversation) => ({
          id: conversation.id,
          title: conversation.title,
          assistant_mode: conversation.assistant_mode,
          messages: [],
        }));

        setChats(savedChats);

        if (savedChats.length > 0) {
          const mostRecent = savedChats[0];
          setActiveId(mostRecent.id);
          setMode(mostRecent.assistant_mode || "agentic");

          const detail = await request(
            `/conversations/${mostRecent.id}`
          );

          if (cancelled) return;

          setChats((previous) =>
            previous.map((chat) =>
              chat.id === mostRecent.id
                ? { ...chat, messages: detail.messages }
                : chat
            )
          );
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setHistoryLoading(false);
      }
    }

    loadConversations();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    return () => recognitionRef.current?.stop();
  }, []);

  function newChat() {
    if (loading) return;
    setActiveId(null);
    setInput("");
    setError("");
    inputRef.current?.focus();
  }

  async function openChat(chat) {
    if (loading) return;

    setActiveId(chat.id);
    setMode(chat.assistant_mode || "agentic");
    setInput("");
    setError("");

    try {
      const data = await request(`/conversations/${chat.id}`);

      setChats((previous) =>
        previous.map((item) =>
          item.id === chat.id
            ? { ...item, messages: data.messages }
            : item
        )
      );
    } catch (err) {
      setError(err.message);
    }
  }

  function startVoiceInput() {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setError(
        "Voice input is not supported in this browser. Try Chrome or Edge."
      );
      return;
    }

    if (listening) {
      recognitionRef.current?.stop();
      setListening(false);
      return;
    }

    setError("");

    const recognition = new SpeechRecognition();
    recognition.lang = "en-IN";
    recognition.interimResults = true;
    recognition.continuous = false;
    recognitionRef.current = recognition;

    recognition.onstart = () => setListening(true);
    recognition.onend = () => setListening(false);

    recognition.onerror = (event) => {
      setListening(false);
      setError(
        event.error === "not-allowed"
          ? "Allow microphone access in your browser to use voice input."
          : `Voice input error: ${event.error}`
      );
    };

    recognition.onresult = (event) => {
      let transcript = "";
      for (let i = 0; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      setInput(transcript);
    };

    recognition.start();
  }

  async function sendMessage(text = input) {
    const question = text.trim();
    if (!question || loading) return;

    setInput("");
    setError("");
    setLoading(true);

    let chatId = activeId;
    let conversationMode = mode;

    try {
      if (!chatId) {
        const created = await request("/conversations", {
          method: "POST",
          body: JSON.stringify({
            title: question.slice(0, 60) || "New chat",
            assistant_mode: mode,
          }),
        });

        chatId = created.conversation.id;
        conversationMode = created.conversation.assistant_mode;

        setActiveId(chatId);
        setChats((previous) => [
          { ...created.conversation, messages: [] },
          ...previous,
        ]);
      }

      const savedUserMessage = await request(
        `/conversations/${chatId}/messages`,
        {
          method: "POST",
          body: JSON.stringify({
            role: "user",
            content: question,
          }),
        }
      );

      setChats((previous) =>
        previous.map((chat) =>
          chat.id === chatId
            ? {
                ...chat,
                title: savedUserMessage.conversation.title,
                messages: [...chat.messages, savedUserMessage.message],
              }
            : chat
        )
      );

      const endpoint =
        conversationMode === "agentic" ? "/agentic-ai" : "/chat";

      const result = await request(endpoint, {
        method: "POST",
        body: JSON.stringify({ text: question }),
      });

      const answer =
        result.response ??
        result.answer ??
        result.reply ??
        result.message ??
        JSON.stringify(result, null, 2);

      const savedAssistantMessage = await request(
        `/conversations/${chatId}/messages`,
        {
          method: "POST",
          body: JSON.stringify({
            role: "assistant",
            content: String(answer),
          }),
        }
      );

      setChats((previous) =>
        previous.map((chat) =>
          chat.id === chatId
            ? {
                ...chat,
                messages: [
                  ...chat.messages,
                  savedAssistantMessage.message,
                ],
              }
            : chat
        )
      );
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  async function logout() {
    if (loading) return;

    try {
      await request("/auth/logout", { method: "POST" });
      window.location.reload();
    } catch (err) {
      setError(err.message);
    }
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="flex h-dvh overflow-hidden bg-black text-[#ececec]">
      {sidebarOpen && (
        <aside className="flex w-[260px] shrink-0 flex-col border-r border-white/[0.08] bg-[#080808] p-3 max-md:absolute max-md:inset-y-0 max-md:left-0 max-md:z-30">
          <div className="flex items-center justify-between px-2 py-2">
            <span className="text-[17px] font-semibold tracking-tight">
              botplayer
            </span>
            <button
              onClick={() => setSidebarOpen(false)}
              aria-label="Close sidebar"
              className="rounded-lg p-2 text-[#aaa] transition hover:bg-white/[0.08]"
            >
              <Icon name="panel" size={18} />
            </button>
          </div>

          <button
            onClick={newChat}
            className="mt-5 flex items-center gap-3 rounded-xl border border-white/[0.12] px-3 py-3 text-[13px] text-[#e4e4e7] transition hover:bg-white/[0.06]"
          >
            <Icon name="plus" size={17} />
            New chat
          </button>

          <p className="mb-2 mt-7 px-2 text-[11px] font-medium text-[#858585]">
            Recent conversations
          </p>

          <div className="flex-1 space-y-1 overflow-y-auto">
            {historyLoading ? (
              <p className="px-2 py-2 text-xs text-[#747474]">
                Loading your chats...
              </p>
            ) : (
              <>
                {chats.map((chat) => (
                  <button
                    key={chat.id}
                    onClick={() => openChat(chat)}
                    className={`w-full truncate rounded-lg px-3 py-2.5 text-left text-[13px] transition ${
                      chat.id === activeId
                        ? "bg-[#252525] text-white"
                        : "text-[#b4b4b4] hover:bg-white/[0.06]"
                    }`}
                  >
                    {chat.title}
                  </button>
                ))}
                {chats.length === 0 && (
                  <p className="px-2 py-2 text-xs leading-5 text-[#747474]">
                    Your chats will appear here.
                  </p>
                )}
              </>
            )}
          </div>

          <div className="relative border-t border-white/[0.08] pt-3">
            <button
              onClick={() => setModeOpen(!modeOpen)}
              className="flex w-full items-center justify-between rounded-xl px-3 py-3 text-left transition hover:bg-white/[0.06]"
            >
              <div className="flex items-center gap-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#202020] text-[#c7c7c7]">
                  <Icon name="spark" size={16} />
                </div>
                <div>
                  <p className="text-xs font-medium text-[#e5e5e5]">
                    {mode === "agentic" ? "Agentic AI" : "Support Chatbot"}
                  </p>
                  <p className="mt-1 text-[10px] text-[#888]">
                    Choose assistant
                  </p>
                </div>
              </div>
              <Icon name="chevron" size={16} />
            </button>

            {modeOpen && (
              <div className="absolute bottom-[68px] left-0 z-40 w-full rounded-xl border border-white/10 bg-[#161616] p-1.5 shadow-xl">
                <button
                  onClick={() => {
                    setMode("agentic");
                    setModeOpen(false);
                  }}
                  className="w-full rounded-lg px-3 py-2.5 text-left text-xs hover:bg-white/[0.08]"
                >
                  Agentic AI Assistant
                </button>
                <button
                  onClick={() => {
                    setMode("support");
                    setModeOpen(false);
                  }}
                  className="w-full rounded-lg px-3 py-2.5 text-left text-xs hover:bg-white/[0.08]"
                >
                  Support Chatbot
                </button>
              </div>
            )}

            <button
              onClick={logout}
              className="mt-2 flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-xs text-[#b4b4b4] transition hover:bg-white/[0.06] hover:text-white"
            >
              <Icon name="logout" size={17} />
              Log out
            </button>
          </div>
        </aside>
      )}

      <main className="flex min-w-0 flex-1 flex-col bg-black">
        <header className="flex h-14 shrink-0 items-center gap-3 px-4 sm:px-6">
          {!sidebarOpen && (
            <button
              onClick={() => setSidebarOpen(true)}
              aria-label="Open sidebar"
              className="rounded-lg p-2 text-[#aaa] hover:bg-white/[0.08]"
            >
              <Icon name="panel" size={18} />
            </button>
          )}
          <span className="text-sm font-medium">botplayer</span>
          <span className="text-xs text-[#929292]">
            {mode === "agentic" ? "Agentic AI" : "Support"}
          </span>
        </header>

        <section className="flex min-h-0 flex-1 flex-col">
          {messages.length === 0 ? (
            <div className="flex flex-1 flex-col items-center justify-center px-5 pb-16">
              <div className="mb-5 flex h-11 w-11 items-center justify-center rounded-2xl border border-white/[0.1] bg-[#141414] text-[#e1e1e1]">
                <Icon name="spark" size={22} />
              </div>
              <h1 className="text-center text-[25px] font-medium tracking-tight text-[#ececec] sm:text-[30px]">
                What can I help with?
              </h1>
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto">
              <div className="mx-auto max-w-[760px] space-y-7 px-5 py-8 sm:px-8 sm:py-10">
                {messages.map((message, index) => (
                  <div
                    key={message.id ?? `${activeId}-${index}`}
                    className="flex gap-3.5 sm:gap-4"
                  >
                    <div
                      className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
                        message.role === "user"
                          ? "bg-[#303030] text-white"
                          : "bg-[#171717] text-[#dedede]"
                      }`}
                    >
                      <Icon
                        name={message.role === "user" ? "bot" : "spark"}
                        size={16}
                      />
                    </div>
                    <div className="min-w-0 flex-1 whitespace-pre-wrap break-words pt-1 text-[14px] leading-[1.85] text-[#e4e4e4]">
                      {message.content}
                    </div>
                  </div>
                ))}
                {loading && (
                  <div className="flex items-center gap-3 pl-1 text-[13px] text-[#a0a0a0]">
                    <Icon name="spark" size={17} />
                    botplayer is thinking...
                  </div>
                )}
                <div ref={bottomRef} />
              </div>
            </div>
          )}

          <div className="mx-auto w-full max-w-[800px] px-4 pb-3 pt-3 sm:px-6 sm:pb-5">
            {error && (
              <div className="mb-3 rounded-xl border border-amber-300/15 bg-amber-300/[0.05] px-3 py-2.5 text-xs leading-5 text-amber-100/90">
                {error}
                <button
                  onClick={() => setError("")}
                  className="ml-2 text-amber-100 underline underline-offset-2"
                >
                  Dismiss
                </button>
              </div>
            )}

            <div className="rounded-[26px] border border-white/[0.12] bg-[#111111] px-3 py-2 shadow-[0_4px_24px_rgba(0,0,0,0.25)] transition focus-within:border-white/[0.2] sm:px-4">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={listening ? "Listening..." : "Message botplayer"}
                rows={2}
                className="max-h-48 min-h-[48px] w-full resize-none bg-transparent px-1 py-2.5 text-[14px] leading-6 text-[#ececec] outline-none placeholder:text-[#999]"
              />
              <div className="flex items-center justify-between pb-1 pt-1">
                <div className="flex items-center gap-2">
                  <button
                    onClick={startVoiceInput}
                    aria-label={listening ? "Stop voice input" : "Start voice input"}
                    title="Voice to text"
                    className={`flex h-8 w-8 items-center justify-center rounded-full transition ${
                      listening
                        ? "bg-red-400/15 text-red-300"
                        : "text-[#b4b4b4] hover:bg-white/[0.08] hover:text-white"
                    }`}
                  >
                    <Icon name="mic" size={18} />
                  </button>
                  <span className="hidden text-[11px] text-[#858585] sm:inline">
                    Enter to send · Shift + Enter for a new line
                  </span>
                </div>
                <button
                  onClick={() => sendMessage()}
                  disabled={!input.trim() || loading}
                  aria-label="Send message"
                  className="flex h-9 w-9 items-center justify-center rounded-full bg-[#ececec] text-[#202020] transition hover:bg-white disabled:cursor-not-allowed disabled:bg-[#555] disabled:text-[#999]"
                >
                  <Icon name="send" size={16} />
                </button>
              </div>
            </div>
            <p className="mt-2.5 text-center text-[10px] text-[#777]">
              botplayer can make mistakes. Verify important information.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}

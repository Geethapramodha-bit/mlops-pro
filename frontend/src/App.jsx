
import { useEffect, useState } from "react";
import ChatApp from "./ChatApp";
import Auth from "./Auth";

const API_URL = "http://localhost:8000";

export default function App() {
  const [entered, setEntered] = useState(false);
  const [user, setUser] = useState(null);
  const [checkingSession, setCheckingSession] = useState(true);

  useEffect(() => {
    let active = true;

    async function checkSession() {
      try {
        const response = await fetch(`${API_URL}/auth/me`, {
          method: "GET",
          credentials: "include",
        });

        if (!response.ok) {
          if (active) {
            setUser(null);
            setEntered(false);
          }
          return;
        }

        const data = await response.json();

        if (active) {
          setUser(data.user);
          setEntered(false);
        }
      } catch {
        if (active) {
          setUser(null);
          setEntered(false);
        }
      } finally {
        if (active) {
          setCheckingSession(false);
        }
      }
    }

    checkSession();

    return () => {
      active = false;
    };
  }, []);

  function handleLogin(loggedInUser) {
    setUser(loggedInUser);
    setEntered(true);
  }

  function handleSessionExpired() {
    setUser(null);
    setEntered(true);
  }

  if (checkingSession) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-black text-white">
        <p className="text-sm text-gray-400">Loading botplayer...</p>
      </main>
    );
  }

  if (!entered) {
    return (
      <main
        onClick={() => setEntered(true)}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            setEntered(true);
          }
        }}
        role="button"
        tabIndex={0}
        aria-label="Enter botplayer"
        className="fixed inset-0 cursor-pointer overflow-hidden bg-black"
      >
        <video
          className="absolute inset-0 h-full w-full object-cover"
          src="/intro.mp4"
          autoPlay
          muted
          loop
          playsInline
          preload="auto"
        />

        <div className="pointer-events-none absolute inset-0 bg-black/25" />

        <div className="pointer-events-none absolute inset-x-0 bottom-12 text-center">
          <h1 className="text-3xl font-semibold tracking-[0.2em] text-white sm:text-5xl">
            botplayer
          </h1>
          <p className="mt-4 text-sm text-white/75 sm:text-base">
            Click anywhere to continue
          </p>
        </div>
      </main>
    );
  }

  if (!user) {
    return <Auth onLogin={handleLogin} />;
  }

  return <ChatApp onSessionExpired={handleSessionExpired} />;
}

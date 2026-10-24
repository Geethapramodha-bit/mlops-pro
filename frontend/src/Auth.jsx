
import { useState } from "react";

const API_URL = "http://localhost:8000";

export default function Auth({ onLogin }) {
  const [mode, setMode] = useState("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setLoading(true);

    try {
      const endpoint =
        mode === "signup" ? "/auth/signup" : "/auth/login";

      const payload =
        mode === "signup"
          ? { name, email, password }
          : { email, password };

      const response = await fetch(`${API_URL}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Authentication failed.");
      }

      onLogin(data.user);
    } catch (err) {
      setError(
        err.message ||
          "Unable to connect to the server. Check that FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-[#0b0d14] px-4 text-white">
      <section className="w-full max-w-md rounded-2xl border border-white/10 bg-[#151720] p-8 shadow-2xl">
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-semibold tracking-wide">
            botplayer
          </h1>
          <p className="mt-3 text-sm text-gray-400">
            {mode === "signup"
              ? "Create your account to get started."
              : "Welcome back. Sign in to continue."}
          </p>
        </div>

        <div className="mb-6 grid grid-cols-2 rounded-lg bg-[#0b0d14] p-1">
          <button
            type="button"
            onClick={() => {
              setMode("login");
              setError("");
            }}
            className={`rounded-md py-2 text-sm transition ${
              mode === "login"
                ? "bg-[#292d3a] text-white"
                : "text-gray-400 hover:text-white"
            }`}
          >
            Log in
          </button>

          <button
            type="button"
            onClick={() => {
              setMode("signup");
              setError("");
            }}
            className={`rounded-md py-2 text-sm transition ${
              mode === "signup"
                ? "bg-[#292d3a] text-white"
                : "text-gray-400 hover:text-white"
            }`}
          >
            Sign up
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {mode === "signup" && (
            <div>
              <label className="mb-2 block text-sm text-gray-300">
                Full name
              </label>
              <input
                required
                minLength={2}
                maxLength={100}
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="Your name"
                autoComplete="name"
                className="w-full rounded-lg border border-white/10 bg-[#0d0f16] px-4 py-3 text-sm outline-none focus:border-gray-500"
              />
            </div>
          )}

          <div>
            <label className="mb-2 block text-sm text-gray-300">
              Email address
            </label>
            <input
              required
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="you@example.com"
              autoComplete="email"
              className="w-full rounded-lg border border-white/10 bg-[#0d0f16] px-4 py-3 text-sm outline-none focus:border-gray-500"
            />
          </div>

          <div>
            <label className="mb-2 block text-sm text-gray-300">
              Password
            </label>
            <input
              required
              type="password"
              minLength={mode === "signup" ? 8 : 1}
              maxLength={72}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder={
                mode === "signup"
                  ? "At least 8 characters"
                  : "Enter your password"
              }
              autoComplete={
                mode === "signup" ? "new-password" : "current-password"
              }
              className="w-full rounded-lg border border-white/10 bg-[#0d0f16] px-4 py-3 text-sm outline-none focus:border-gray-500"
            />
          </div>

          {error && (
            <p
              role="alert"
              className="rounded-lg border border-red-500/20 bg-red-500/10 p-3 text-sm text-red-300"
            >
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full rounded-lg bg-white py-3 font-medium text-black transition hover:bg-gray-200 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading
              ? "Please wait..."
              : mode === "signup"
                ? "Create account"
                : "Log in"}
          </button>
        </form>

        <p className="mt-6 text-center text-xs leading-5 text-gray-500">
          Your account lets you access your botplayer workspace.
        </p>
      </section>
    </main>
  );
}

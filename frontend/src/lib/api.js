export const API_BASE =
  import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(message, { offline = false } = {}) {
    super(message);
    this.name = "ApiError";
    this.offline = offline;
  }
}

const OFFLINE_HINT =
  `Could not reach the backend at ${API_BASE}. Start it with ` +
  `".venv\\Scripts\\activate" then "python -m src.api".`;

export async function fetchHealth() {
  const response = await fetch(`${API_BASE}/health`);
  return response.json();
}

/**
 * Score one reading. Returns the API payload plus the round-trip time, which
 * the dashboard shows so a slow model is visible rather than just "sluggish".
 */
export async function predict({ cycle, sensors }, signal) {
  const startedAt = performance.now();

  let response;
  try {
    response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cycle, sensors }),
      signal,
    });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError(OFFLINE_HINT, { offline: true });
  }

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new ApiError(
      typeof data.detail === "string"
        ? data.detail
        : `Backend returned ${response.status}`
    );
  }

  return { ...data, latencyMs: Math.round(performance.now() - startedAt) };
}

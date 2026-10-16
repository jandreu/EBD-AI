/**
 * api.js
 *
 * Thin fetch wrapper for backend.py's FastAPI service — the React
 * equivalent of api_client.py. Same endpoints, same response shapes.
 * Nothing here talks to rag_core.py or viz_tools.py directly; this file
 * is the ONLY thing the rest of the app talks to, mirroring the
 * frontend/backend split api_client.py's docstring describes.
 */

// The page and the API share one server; the API lives under /api.
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "/api";

export class BackendError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.status = status;
  }
}

// The visitor's workspace (their uploaded rule base, notes and references).
// Every request carries it, so the backend answers with their material.
let workspaceId = null;
export function setWorkspace(id) {
  workspaceId = id;
}

async function request(path, { method = "GET", body, contentType } = {}) {
  const headers = {};
  if (workspaceId) headers["X-Workspace"] = workspaceId;
  if (contentType) headers["Content-Type"] = contentType;
  let resp;
  try {
    resp = await fetch(`${BACKEND_URL}${path}`, { method, headers, body });
  } catch (e) {
    throw new BackendError(`Could not reach backend at ${BACKEND_URL}${path}: ${e.message}`);
  }
  if (resp.status === 401) {
    // The site's password gate answers 401 once a login has ended.
    throw new BackendError("Your login has ended. Reload the page and log in again.", 401);
  }
  if (!resp.ok) {
    let detail = "";
    try {
      detail = (await resp.json()).detail;
    } catch {
      // not JSON
    }
    throw new BackendError(
      typeof detail === "string" && detail ? detail : `Backend returned ${resp.status} for ${path}`,
      resp.status
    );
  }
  return resp.json();
}

const get = (path) => request(path);
const post = (path, body) =>
  request(path, { method: "POST", body: JSON.stringify(body), contentType: "application/json" });

export const health = () => get("/health");

/** Returns {source, target_name, n_rules, rules, cv_performance_estimate, n_seeds}. */
export const getRules = () => get("/rules");

/** Returns the static channel atlas used as the topomap background. */
export const getAtlas = () => get("/atlas");

/**
 * Returns {answer, retrieved, in_scope, used_fallback, latency_ms,
 * visualization, citations, papers}. `visualization` is null or
 * {type, topomap: [...] | null, chord: {...} | null}.
 */
export const ask = (question) => post("/ask", { question });

/** Topomap points across EVERY rule in the model (sidebar overview). */
export const getOverviewTopomap = () => get("/visualize/topomap");

/** Chord {nodes, edges} across EVERY rule in the model (sidebar overview). */
export const getOverviewChord = () => get("/visualize/chord");

// ---- Workspace: the visitor's own rule base, notes and references ----
// Each call returns the workspace summary:
// {id, rules: null | {source, n_rules, classes, n_features}, context, references: [{id, name, n_passages}]}

export const createWorkspace = () => request("/workspace", { method: "POST" });
export const getWorkspace = (id) => get(`/workspace/${id}`);

const upload = (path, file) =>
  request(`${path}?filename=${encodeURIComponent(file.name)}`, {
    method: path.endsWith("/rules") ? "PUT" : "POST",
    body: file,
    contentType: file.type || "application/octet-stream",
  });

export const uploadRules = (id, file) => upload(`/workspace/${id}/rules`, file);
export const resetRules = (id) => request(`/workspace/${id}/rules`, { method: "DELETE" });
export const saveContext = (id, text) =>
  request(`/workspace/${id}/context`, {
    method: "PUT",
    body: JSON.stringify({ text }),
    contentType: "application/json",
  });
export const uploadReference = (id, file) => upload(`/workspace/${id}/references`, file);
export const deleteReference = (id, refId) =>
  request(`/workspace/${id}/references/${encodeURIComponent(refId)}`, { method: "DELETE" });

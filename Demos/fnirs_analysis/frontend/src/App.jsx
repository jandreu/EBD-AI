import { useCallback, useEffect, useState } from "react";
import Sidebar from "./components/Sidebar.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import WorkspaceMenu from "./components/WorkspaceMenu.jsx";
import {
  health,
  getRules,
  getAtlas,
  getOverviewTopomap,
  getOverviewChord,
  getWorkspace,
  setWorkspace,
  BackendError,
} from "./api.js";

// The workspace id survives a reload of this tab, not a new tab.
const STORAGE_KEY = "fnirs-workspace";
function storedWorkspaceId() {
  try {
    return sessionStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}
function storeWorkspaceId(id) {
  try {
    if (id) sessionStorage.setItem(STORAGE_KEY, id);
    else sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    // storage unavailable: the workspace lasts until the page is reloaded
  }
}

/** The rule data the sidebar draws, for the current workspace (or the study). */
async function loadModel() {
  const [data, overviewTopomap, overviewChord] = await Promise.all([
    getRules(),
    getOverviewTopomap(),
    getOverviewChord(),
  ]);
  return { data, overviewTopomap, overviewChord };
}

export default function App() {
  const [state, setState] = useState({ status: "loading" });
  const [workspace, setWorkspaceSummary] = useState(null);

  const forgetWorkspace = useCallback(async () => {
    setWorkspace(null);
    storeWorkspaceId(null);
    setWorkspaceSummary(null);
    try {
      const model = await loadModel();
      setState((s) => ({ ...s, ...model }));
    } catch {
      // keep what is shown
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const savedId = storedWorkspaceId();
        if (savedId) {
          try {
            setWorkspace(savedId);
            const summary = await getWorkspace(savedId);
            if (!cancelled) setWorkspaceSummary(summary);
          } catch {
            setWorkspace(null);
            storeWorkspaceId(null);
          }
        }
        const [h, atlas, model] = await Promise.all([health(), getAtlas(), loadModel()]);
        if (cancelled) return;
        setState({ status: "ready", health: h, atlas, ...model });
      } catch (err) {
        if (cancelled) return;
        const message =
          err instanceof BackendError
            ? err.message
            : `Unexpected error loading the model: ${err.message}`;
        setState({ status: "error", message });
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const onWorkspaceChange = useCallback(async (summary, rulesChanged) => {
    storeWorkspaceId(summary.id);
    setWorkspaceSummary(summary);
    if (rulesChanged) {
      try {
        const model = await loadModel();
        setState((s) => ({ ...s, ...model }));
      } catch (err) {
        if (err instanceof BackendError && err.status === 404) forgetWorkspace();
      }
    }
  }, [forgetWorkspace]);

  if (state.status === "loading") {
    return (
      <div className="main-panel">
        <h1>Explainable Pattern Analysis for fNIRS</h1>
        <p className="caption">Connecting to the backend...</p>
      </div>
    );
  }

  if (state.status === "error") {
    return (
      <div className="main-panel">
        <h1>Explainable Pattern Analysis for fNIRS</h1>
        <div className="error-banner">
          Can't reach the backend API.
          <br />
          {state.message}
          <br />
          Make sure it's running: <code>python Demos/fnirs_analysis/backend.py</code>{" "}
          from the repository root, then reload this page.
        </div>
      </div>
    );
  }

  const degraded = state.health.status !== "ok";

  return (
    <div className="app-layout">
      <Sidebar
        data={state.data}
        atlas={state.atlas}
        overviewTopomap={state.overviewTopomap}
        overviewChord={state.overviewChord}
      />
      <div>
        {degraded && (
          <div className="warning-banner warning-banner-spaced">
            Written explanations are switched off on this server, so each answer lists
            the rules that match your question. The rules, maps and 3D brain on the
            left work as usual.
          </div>
        )}
        <ChatPanel
          atlas={state.atlas}
          menu={
            <WorkspaceMenu
              workspace={workspace}
              onChange={onWorkspaceChange}
              onExpired={forgetWorkspace}
            />
          }
        />
      </div>
    </div>
  );
}

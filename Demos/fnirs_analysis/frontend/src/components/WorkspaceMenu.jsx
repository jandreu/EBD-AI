import { useEffect, useRef, useState } from "react";
import {
  createWorkspace,
  deleteReference,
  resetRules,
  saveContext,
  setWorkspace,
  uploadReference,
  uploadRules,
} from "../api.js";

/**
 * "Your data": a toggle that opens a compact panel where a visitor can use
 * their own Ex-Fuzzy rule base, describe their experiment and add reference
 * documents. Closed by default. Everything goes into the visitor's own
 * workspace on the server; nobody else sees it.
 */
export default function WorkspaceMenu({ workspace, onChange, onExpired }) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState("");
  const [message, setMessage] = useState(null);
  const [notes, setNotes] = useState(workspace?.context || "");
  const rulesInput = useRef(null);
  const refsInput = useRef(null);
  // The id lives in a ref too: several files uploaded in one go must share
  // the workspace the first one created.
  const workspaceIdRef = useRef(workspace?.id || null);

  useEffect(() => {
    workspaceIdRef.current = workspace?.id || null;
    setNotes(workspace?.context || "");
  }, [workspace?.id, workspace?.context]);

  const items =
    (workspace?.rules ? 1 : 0) + (workspace?.context ? 1 : 0) + (workspace?.references?.length || 0);

  async function run(label, action, { rulesChanged = false, done = null } = {}) {
    setBusy(label);
    setMessage(null);
    try {
      let id = workspaceIdRef.current;
      if (!id) {
        const created = await createWorkspace();
        id = created.id;
        workspaceIdRef.current = id;
        setWorkspace(id);
        onChange(created, false);
      }
      const summary = await action(id);
      onChange(summary, rulesChanged);
      if (done) setMessage({ kind: "ok", text: done(summary) });
    } catch (err) {
      if (err.status === 404) {
        workspaceIdRef.current = null;
        onExpired();
      }
      setMessage({ kind: "error", text: err.message });
    } finally {
      setBusy("");
    }
  }

  function pickRules(e) {
    const file = e.target.files[0];
    e.target.value = "";
    if (!file) return;
    run("rules", (id) => uploadRules(id, file), {
      rulesChanged: true,
      done: (s) => `Loaded ${s.rules.n_rules} rules from ${s.rules.source}.`,
    });
  }

  async function pickReferences(e) {
    const files = [...e.target.files];
    e.target.value = "";
    for (const file of files) {
      // One at a time, so each file's problem is reported by name.
      await run("references", (id) => uploadReference(id, file), {
        done: () => `Added ${file.name}.`,
      });
    }
  }

  const rules = workspace?.rules;
  const savedNotes = workspace?.context || "";

  return (
    <div className="workspace">
      <button
        type="button"
        className="workspace-toggle"
        aria-expanded={open}
        aria-controls="workspace-panel"
        onClick={() => setOpen((o) => !o)}
      >
        <span aria-hidden="true">{open ? "▾" : "▸"}</span> Your data
        {items > 0 && <span className="workspace-count">{items}</span>}
      </button>

      {open && (
        <section id="workspace-panel" className="workspace-panel" aria-label="Your rule base, notes and references">
          <div className="workspace-row">
            <div className="workspace-label">Rule base</div>
            <div className="workspace-body">
              <p className="workspace-status">
                {rules ? (
                  <>
                    <strong>{rules.n_rules} rules</strong> from {rules.source} ({rules.classes.join(", ")})
                  </>
                ) : (
                  "The original study's rules."
                )}
              </p>
              <div className="workspace-actions">
                <button type="button" onClick={() => rulesInput.current.click()} disabled={!!busy}>
                  {busy === "rules" ? "Reading…" : "Upload rule base…"}
                </button>
                {rules && (
                  <button
                    type="button"
                    className="secondary"
                    disabled={!!busy}
                    onClick={() =>
                      run("rules", (id) => resetRules(id), {
                        rulesChanged: true,
                        done: () => "Back to the original study's rules.",
                      })
                    }
                  >
                    Use the original
                  </button>
                )}
                <input
                  ref={rulesInput}
                  type="file"
                  accept=".txt,.json,.log,.rules,text/plain,application/json"
                  hidden
                  onChange={pickRules}
                />
              </div>
              <p className="caption">
                The text Ex-Fuzzy prints for a rule base (<code>print(model.rule_base)</code>) or
                this demo's JSON.
              </p>
            </div>
          </div>

          <div className="workspace-row">
            <label className="workspace-label" htmlFor="workspace-notes">
              Experiment
            </label>
            <div className="workspace-body">
              <textarea
                id="workspace-notes"
                rows={3}
                maxLength={20000}
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Task, participants, sensors, what the classes mean…"
              />
              <div className="workspace-actions">
                <button
                  type="button"
                  disabled={!!busy || notes.trim() === savedNotes}
                  onClick={() =>
                    run("notes", (id) => saveContext(id, notes), {
                      done: (s) => (s.context ? "Notes saved." : "Notes cleared."),
                    })
                  }
                >
                  {busy === "notes" ? "Saving…" : "Save notes"}
                </button>
              </div>
            </div>
          </div>

          <div className="workspace-row">
            <div className="workspace-label">References</div>
            <div className="workspace-body">
              {workspace?.references?.length > 0 && (
                <ul className="workspace-files">
                  {workspace.references.map((r) => (
                    <li key={r.id}>
                      <span title={`${r.n_passages} passages`}>{r.name}</span>
                      <button
                        type="button"
                        aria-label={`Remove ${r.name}`}
                        disabled={!!busy}
                        onClick={() => run("references", (id) => deleteReference(id, r.id))}
                      >
                        ×
                      </button>
                    </li>
                  ))}
                </ul>
              )}
              <div className="workspace-actions">
                <button type="button" onClick={() => refsInput.current.click()} disabled={!!busy}>
                  {busy === "references" ? "Reading…" : "Upload references…"}
                </button>
                <input
                  ref={refsInput}
                  type="file"
                  accept=".pdf,.txt,.md,application/pdf,text/plain,text/markdown"
                  multiple
                  hidden
                  onChange={pickReferences}
                />
              </div>
              <p className="caption">PDF or text, up to 8 files. Answers cite them alongside the built-in papers.</p>
            </div>
          </div>

          {message && (
            <p className={`workspace-message ${message.kind}`} role={message.kind === "error" ? "alert" : "status"}>
              {message.text}
            </p>
          )}
          <p className="caption">
            Only you see these. They are kept until you close this tab, or for 3 hours without use.
          </p>
        </section>
      )}
    </div>
  );
}

"""
workspace.py

A visitor's own material: a rule base produced by Ex-Fuzzy (or this toolbox's
ex_fuzzy), notes about their experiment, and reference documents. backend.py
keeps one Workspace per browser tab, in memory only, so uploads change what
that visitor's chat and pictures use and nobody else's. Workspaces expire
after WORKSPACE_IDLE_HOURS without use and vanish when the server restarts.

Accepted rule bases:
- Ex-Fuzzy's printed text (`print(classifier.rule_base)`, `print_rules`,
  `persistence` files): "Rules for consequent: <class>" headers followed by
  "IF <var> IS <label> [(MOD <hedge>)] AND ... [WITH DS <x>, ACC <y>, WGHT <w>]".
- JSON in this demo's format (data/extracted_rules_final_raw.json, written by
  train_model.py) or a bare list of {antecedent, consequent, ...} rules.

Accepted references: PDF, plain text and Markdown files. Their text is split
into passages and searched together with the shipped papers (paper_search.py).
"""

import io
import json
import re
import secrets
import threading
import time
from pathlib import Path

from paper_search import CombinedIndex, PaperIndex

MAX_RULE_FILE = 512 * 1024        # bytes
MAX_RULES = 500
MAX_CONTEXT = 20_000              # characters
MAX_REFERENCE_FILE = 15 * 1024 * 1024
MAX_REFERENCES = 8
MAX_PDF_PAGES = 60
MAX_REFERENCE_CHARS = 300_000
MAX_WORKSPACES = 300
IDLE_SECONDS = 3 * 3600

_HEADER = re.compile(r"^\s*Rules(?:\s+for)?(?:\s+consequent)?\s*:?\s*(.*?)\s*:?\s*$", re.IGNORECASE)
_WITH = re.compile(r"\sWITH\s")
_THEN = re.compile(r"\sTHEN\s")
_AND = re.compile(r"\sAND\s")
_IS = re.compile(r"\sIS\s")
_STAT = {key: re.compile(rf"(?:^|[\s,(]){key}\s+([-+0-9.eE]+|nan)") for key in ("DS", "ACC", "WGHT")}


class UploadError(ValueError):
    """An upload that cannot be used; the message is shown to the visitor."""


def _number(text: str, key: str):
    match = _STAT[key].search(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return None if value != value else value  # NaN -> None


def parse_rule_text(text: str) -> list:
    """Rules from Ex-Fuzzy's printed format, as dicts with antecedent,
    consequent, dominance_score, accuracy (and weight when printed)."""
    rules, consequent = [], None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or set(line) <= {"-", "="}:
            continue
        if line.upper().startswith("IF "):
            parts = _WITH.split(line, maxsplit=1)
            antecedent = _THEN.split(parts[0], maxsplit=1)[0][3:].strip()
            stats = parts[1] if len(parts) > 1 else ""
            terms = [t.strip() for t in _AND.split(antecedent) if t.strip()]
            if not terms or not all(_IS.search(f" {t} ") or " IS " in t for t in terms):
                raise UploadError(f"This rule could not be read: {line[:120]}")
            rule = {"antecedent": " AND ".join(terms),
                    "consequent": consequent if consequent is not None else "class 1",
                    "dominance_score": _number(stats, "DS"), "accuracy": _number(stats, "ACC")}
            weight = _number(stats, "WGHT")
            if weight is not None:
                rule["weight"] = weight
            rules.append(rule)
        elif line.lower().startswith("rules"):
            match = _HEADER.match(line)
            consequent = (match.group(1) if match else "") or f"class {len({r['consequent'] for r in rules}) + 1}"
    return rules


def parse_rule_base(raw: bytes, filename: str = "") -> dict:
    """The upload as the data dict the explainer uses (the shape of
    data/extracted_rules_final_raw.json). Raises UploadError."""
    if len(raw) > MAX_RULE_FILE:
        raise UploadError(f"The rule base file is larger than {MAX_RULE_FILE // 1024} KB.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise UploadError("The rule base must be a text or JSON file (UTF-8).")
    meta = {}
    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            loaded = json.loads(text)
        except ValueError as exc:
            raise UploadError(f"The JSON file could not be read: {exc}")
        listed = loaded.get("rules") if isinstance(loaded, dict) else loaded
        if not isinstance(listed, list):
            raise UploadError('The JSON file needs a "rules" list.')
        meta = loaded if isinstance(loaded, dict) else {}
        rules = []
        for item in listed:
            if not isinstance(item, dict) or not item.get("antecedent") or item.get("consequent") is None:
                raise UploadError('Each rule in the JSON file needs an "antecedent" and a "consequent".')
            antecedent = re.sub(r"^\s*IF\s+", "", str(item["antecedent"]))
            rules.append({"antecedent": antecedent, "consequent": str(item["consequent"]),
                          "dominance_score": _float_or_none(item.get("dominance_score", item.get("score"))),
                          "accuracy": _float_or_none(item.get("accuracy"))})
    else:
        rules = parse_rule_text(text)
    if not rules:
        raise UploadError("No rules were found. Upload the text Ex-Fuzzy prints for a rule base "
                          "(lines starting with IF) or a JSON file with a \"rules\" list.")
    if len(rules) > MAX_RULES:
        raise UploadError(f"The rule base has more than {MAX_RULES} rules.")
    features = []
    for rule in rules:
        for term in _AND.split(f" {rule['antecedent']} "):
            name = _IS.split(f" {term.strip()} ", maxsplit=1)[0].strip()
            if name and name not in features:
                features.append(name)
    for i, rule in enumerate(rules, start=1):
        rule["rule_id"] = i
    cv = meta.get("cv_performance_estimate") if isinstance(meta.get("cv_performance_estimate"), dict) else None
    return {
        "feature_names": meta.get("feature_names") if isinstance(meta.get("feature_names"), list) else features,
        "target_name": str(meta.get("target_name") or "class"),
        "cv_performance_estimate": cv,
        "n_seeds": int(meta.get("n_seeds") or 1) if cv else 1,
        "rule_stability_across_seeds": meta.get("rule_stability_across_seeds") or [],
        "rules": rules,
        "source": Path(filename).name or "uploaded rule base",
    }


def _float_or_none(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _passages(text: str, size: int = 1100) -> list:
    """Text cut into passages of about `size` characters at sentence ends."""
    text = re.sub(r"-\n(?=[a-z])", "", text)            # words hyphenated across lines
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    sentences = []
    for paragraph in paragraphs:
        sentences += re.split(r"(?<=[.!?])\s+(?=[A-Z(])", paragraph)
    out, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > size:
            out.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    if current:
        out.append(current)
    return [p for p in out if len(p) > 80]


def extract_reference(raw: bytes, filename: str) -> dict:
    """A reference document as a paper entry for PaperIndex. Raises UploadError."""
    name = Path(filename).name or "reference"
    if len(raw) > MAX_REFERENCE_FILE:
        raise UploadError(f"{name} is larger than {MAX_REFERENCE_FILE // (1024 * 1024)} MB.")
    title = ""
    if raw[:5] == b"%PDF-" or name.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw))
            pages = []
            for page in reader.pages[:MAX_PDF_PAGES]:
                pages.append(page.extract_text() or "")
                if sum(map(len, pages)) > MAX_REFERENCE_CHARS:
                    break
            text = "\n\n".join(pages)
            title = str((reader.metadata or {}).get("/Title") or "").strip()
        except ImportError:
            raise UploadError("PDF files need the pypdf package on the server (pip install pypdf).")
        except Exception:
            raise UploadError(f"{name} could not be read as a PDF.")
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise UploadError(f"{name} is not a PDF or a UTF-8 text file.")
    text = text[:MAX_REFERENCE_CHARS]
    passages = [{"section": "Uploaded document", "text": p} for p in _passages(text)]
    if not passages:
        raise UploadError(f"No readable text was found in {name} (a scanned PDF has no text layer).")
    label = Path(name).stem[:60]
    return {
        "id": "upload-" + secrets.token_hex(4),
        "citation": f"Uploaded document: {name}" + (f" ({title[:150]})" if title and title.lower() != label.lower() else ""),
        "short": label,
        "doi": "",
        "url": "",
        "licence": "uploaded by the visitor",
        "name": name,
        "passages": passages,
    }


class Workspace:
    def __init__(self, base_index):
        self.id = secrets.token_urlsafe(18)
        self.used = time.time()
        self.rules = None          # data dict from parse_rule_base, or None for the shipped model
        self.context = ""
        self.references = []
        self._base_index = base_index
        self.index = None          # the uploads searched first, then the shipped papers

    def add_reference(self, paper: dict) -> None:
        if len(self.references) >= MAX_REFERENCES:
            raise UploadError(f"At most {MAX_REFERENCES} reference documents can be uploaded.")
        self.references.append(paper)
        self._reindex()

    def remove_reference(self, ref_id: str) -> None:
        self.references = [r for r in self.references if r["id"] != ref_id]
        self._reindex()

    def _reindex(self) -> None:
        self.index = (CombinedIndex(PaperIndex(self.references, background=self._base_index), self._base_index)
                      if self.references else None)

    def summary(self) -> dict:
        return {
            "id": self.id,
            "rules": None if self.rules is None else {
                "source": self.rules["source"], "n_rules": len(self.rules["rules"]),
                "classes": sorted({r["consequent"] for r in self.rules["rules"]}),
                "n_features": len(self.rules["feature_names"]),
            },
            "context": self.context,
            "references": [{"id": r["id"], "name": r["name"], "n_passages": len(r["passages"])}
                           for r in self.references],
        }


class WorkspaceStore:
    """Thread-safe, in-memory workspaces with idle expiry and a size cap."""

    def __init__(self, base_index):
        self._base_index = base_index   # PaperIndex over the shipped papers, or None
        self._items = {}
        self._lock = threading.Lock()

    def create(self) -> Workspace:
        with self._lock:
            self._expire()
            if len(self._items) >= MAX_WORKSPACES:
                oldest = min(self._items.values(), key=lambda w: w.used)
                del self._items[oldest.id]
            workspace = Workspace(self._base_index)
            self._items[workspace.id] = workspace
            return workspace

    def get(self, workspace_id: str):
        if not workspace_id:
            return None
        with self._lock:
            self._expire()
            workspace = self._items.get(workspace_id)
            if workspace is not None:
                workspace.used = time.time()
            return workspace

    def _expire(self) -> None:
        cutoff = time.time() - IDLE_SECONDS
        for key in [k for k, w in self._items.items() if w.used < cutoff]:
            del self._items[key]

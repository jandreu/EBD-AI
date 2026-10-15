"""
build_paper_corpus.py

Downloads the open-access papers listed in PAPERS from Europe PMC and writes
data/papers.json: one entry per paper (reference, link, licence) with its
abstract and its introduction, discussion and conclusion paragraphs. The
explainer (paper_search.py, rag_core.py) retrieves passages from this file as
background for its answers and cites the papers they come from.

Only papers whose Europe PMC licence is CC BY, CC BY-SA or CC0 are kept, so the
passages may be redistributed with attribution (the reference and link travel
with every passage). In-text citations are removed from the passages, so the
model cannot cite a paper it was not given.

Run from the repository root (needs internet access):
    python Demos/fnirs_analysis/build_paper_corpus.py            # writes data/papers.json
    python Demos/fnirs_analysis/build_paper_corpus.py --raw DIR  # also keeps the downloaded XML in DIR
"""

import argparse
import json
import re
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
ALLOWED_LICENCES = {"cc by", "cc by-sa", "cc0"}

# PMCID, and why the paper is here.
PAPERS = [
    ("PMC5810421", "the source dataset: EEG and NIRS during n-back and other tasks"),
    ("PMC12715002", "review of the n-back task: variants, what it measures, uses"),
    ("PMC4274887", "what n-back scores measure compared with other working-memory tasks"),
    ("PMC4683174", "the prefrontal cortex in working memory"),
    ("PMC3930096", "fNIRS sensitivity to cognitive state and n-back load"),
    ("PMC3893598", "n-back workload measured in the prefrontal cortex with fNIRS"),
    ("PMC9971214", "frontal fNIRS activation and performance across n-back sessions"),
    ("PMC3460859", "ageing, cerebral oxygenation and n-back load (fNIRS)"),
    ("PMC8133445", "n-back workload and neurovascular coupling"),
    ("PMC5380755", "working-memory load decoded from high-density fNIRS beyond the forehead"),
    ("PMC10543376", "load-dependent fMRI activation patterns in the n-back task"),
    ("PMC5529418", "classifying mental workload from EEG, fNIRS and physiology (n-back)"),
    ("PMC4309034", "fNIRS brain-computer interfaces: signals, features, classifiers"),
    ("PMC6367070", "fNIRS for cognitive neuroscience: principles, strengths, limits"),
    ("PMC7793571", "best practices for fNIRS: HbO, HbR, physiology, reporting"),
    ("PMC3348501", "systemic physiological artefacts in fNIRS"),
]

# Sections that are mostly procedure or statistics; their paragraphs are left out.
SKIP_SECTION = re.compile(
    r"method|material|participant|subject|procedure|apparatus|statistic|data analys|"
    r"acquisition|preprocess|result|acknowledg|funding|author contribution|conflict|"
    r"competing|supplementar|data availability|ethic|abbreviation", re.IGNORECASE)


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "EBD-Toolbox fNIRS demo corpus builder"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def metadata(pmcid: str) -> dict:
    query = urllib.parse.urlencode({"query": f"PMCID:{pmcid}", "format": "json", "resultType": "core"})
    results = json.loads(fetch(f"{API}/search?{query}"))["resultList"]["result"]
    if not results:
        raise ValueError(f"{pmcid}: not found in Europe PMC")
    return results[0]


def reference(meta: dict) -> tuple:
    """(full reference, short in-text form such as 'Fishburn et al., 2014')."""
    authors = [a for a in meta.get("authorList", {}).get("author", []) if a.get("lastName")]
    names = [f"{a['lastName']}, {a.get('initials', '')}".strip(", ") for a in authors]
    if len(names) > 6:
        names = names[:6] + ["et al."]
    year = meta.get("pubYear", "n.d.")
    journal = meta.get("journalInfo", {})
    venue = journal.get("journal", {}).get("title", "")
    volume = journal.get("volume", "")
    pages = meta.get("pageInfo", "")
    title = re.sub(r"<[^>]+>", "", meta.get("title", "")).rstrip(".")
    full = f"{', '.join(names)} ({year}). {title}. {venue}" + (f", {volume}" if volume else "") + (f", {pages}" if pages else "") + "."
    surnames = [a["lastName"] for a in authors]
    if len(surnames) == 1:
        short = f"{surnames[0]}, {year}"
    elif len(surnames) == 2:
        short = f"{surnames[0]} & {surnames[1]}, {year}"
    else:
        short = f"{surnames[0]} et al., {year}"
    return full, short


def clean_text(element) -> str:
    """Paragraph text without in-text citations, footnote marks or extra spaces."""
    for parent in element.iter():
        for child in list(parent):
            if child.tag == "xref" and child.get("ref-type") in ("bibr", "fn", "table-fn"):
                tail = child.tail or ""
                child.clear()
                child.tail = tail
    text = "".join(element.itertext())
    text = re.sub(r"\s+", " ", text)
    # Brackets left empty, or holding only separators, once the citations are gone.
    for _ in range(3):
        text = re.sub(r"\s*[\(\[](?:\s|[,;–\-]|and\b|e\.g\.,?|see\b|cf\.|for (?:a )?review(?:s)?,?)*[\)\]]", "", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text.strip()


def passages(root) -> list:
    out = []
    for abstract in root.iter("abstract"):
        if abstract.get("abstract-type") in ("graphical", "teaser"):
            continue
        for p in abstract.iter("p"):
            text = clean_text(p)
            if len(text) > 80:
                out.append({"section": "Abstract", "text": text})
        break
    body = root.find(".//body")
    if body is None:
        return out

    def walk(section, heading):
        title_el = section.find("title")
        title = clean_text(title_el) if title_el is not None else ""
        full_heading = f"{heading} > {title}" if heading and title else (title or heading)
        if SKIP_SECTION.search(title or ""):
            return
        for child in section:
            if child.tag == "p":
                text = clean_text(child)
                if len(text) > 120:
                    out.append({"section": full_heading or "Body", "text": text})
            elif child.tag == "sec":
                walk(child, full_heading)

    for child in body:
        if child.tag == "sec":
            walk(child, "")
        elif child.tag == "p":
            text = clean_text(child)
            if len(text) > 120:
                out.append({"section": "Body", "text": text})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/papers.json from open-access papers in Europe PMC.")
    parser.add_argument("--raw", type=Path, help="keep the downloaded XML files in this folder")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "papers.json")
    args = parser.parse_args()
    raw_dir = args.raw or Path(tempfile.mkdtemp(prefix="fnirs-papers-"))
    raw_dir.mkdir(parents=True, exist_ok=True)

    corpus = []
    for pmcid, why in PAPERS:
        meta = metadata(pmcid)
        licence = (meta.get("license") or "").lower()
        if licence not in ALLOWED_LICENCES:
            print(f"skip {pmcid}: licence {licence or 'unknown'}")
            continue
        xml_path = raw_dir / f"{pmcid}.xml"
        if not xml_path.exists():
            xml_path.write_bytes(fetch(f"{API}/{pmcid}/fullTextXML"))
            time.sleep(0.5)
        root = ET.fromstring(xml_path.read_bytes())
        full, short = reference(meta)
        doi = meta.get("doi", "")
        found = passages(root)
        corpus.append({
            "id": pmcid,
            "citation": full,
            "short": short,
            "doi": doi,
            "url": f"https://doi.org/{doi}" if doi else f"https://europepmc.org/article/PMC/{pmcid}",
            "licence": licence.upper(),
            "why": why,
            "passages": found,
        })
        print(f"{pmcid}  {short:<28} {licence:<8} {len(found):>3} passages  {why}")
    args.out.write_text(json.dumps(corpus, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    total = sum(len(p["passages"]) for p in corpus)
    print(f"wrote {len(corpus)} papers, {total} passages to {args.out}")


if __name__ == "__main__":
    main()

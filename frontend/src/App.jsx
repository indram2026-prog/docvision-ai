import React, { useState, useRef, useCallback, useEffect } from "react";
import PageViewer from "./PageViewer.jsx";
import Inspector from "./Inspector.jsx";
import { PageSkeleton, InspectorSkeleton } from "./Skeleton.jsx";
import { fileKey, getCached, putCached } from "./cache.js";
import { Prev, Next, Copy, Down, Plus, Check } from "./icons.jsx";
import InstallPrompt from "./InstallPrompt.jsx";

const API = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "") + "/api";
const VIEWS = [["structured", "Structured"], ["ocr_only", "OCR only"], ["split", "Compare"]];

export default function App() {
  const [result, setResult] = useState(null);     // finished analysis
  const [pending, setPending] = useState(null);   // optimistic: what the user just handed over, shown before the server answers
  const [fromCache, setFromCache] = useState(false);
  const [error, setError] = useState("");
  const [view, setView] = useState("structured");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState(null);
  const [drag, setDrag] = useState(false);
  const [copied, setCopied] = useState(false);
  const input = useRef();
  const abort = useRef(null);

  useEffect(() => () => pending?.previewUrl && URL.revokeObjectURL(pending.previewUrl), [pending]);

  const finish = (data, cached) => { setResult(data); setFromCache(cached); setPending(null); setPage(0); setSelected(null); };

  const run = useCallback(async ({ key, label, previewUrl, request }) => {
    setError(""); setSelected(null); setPage(0);
    const hit = key && await getCached(key);
    if (hit) { finish(hit, true); return; }          // cache hit: no waiting, no network
    setPending({ label, previewUrl }); setResult(null);
    abort.current?.abort(); abort.current = new AbortController();
    try {
      const r = await request(abort.current.signal);
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `Request failed (${r.status})`);
      const data = await r.json();
      if (key) putCached(key, data);
      finish(data, false);
    } catch (e) {
      if (e.name === "AbortError") return;
      setPending(null); setError(e.message);          // revert the optimistic view
    }
  }, []);

  const upload = async (file) => {
    if (!file) return;
    const key = await fileKey(file, "both");
    const fd = new FormData(); fd.append("file", file);
    run({ key, label: file.name, previewUrl: file.type.startsWith("image/") ? URL.createObjectURL(file) : null,
          request: (signal) => fetch(`${API}/analyze?mode=both`, { method: "POST", body: fd, signal }) });
  };
  const sample = () => {
    const seed = Math.floor(Math.random() * 900) + 1;
    run({ key: `sample:${seed}`, label: `sample ${seed}`, request: (signal) => fetch(`${API}/sample?seed=${seed}`, { signal }) });
  };
  const reset = () => { abort.current?.abort(); setResult(null); setPending(null); setError(""); };

  const doc = result?.structured, base = result?.ocr_only;
  const go = (d) => { setPage((p) => Math.min(result.pages - 1, Math.max(0, p + d))); setSelected(null); };

  useEffect(() => {
    if (!result) return;
    const h = (e) => { if (e.target.tagName === "INPUT") return; if (e.key === "ArrowLeft") go(-1); if (e.key === "ArrowRight") go(1); };
    window.addEventListener("keydown", h); return () => window.removeEventListener("keydown", h);
  });

  const download = (name, text, type) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click();
  };
  const copyJson = async () => {
    setCopied(true);                                   // optimistic: confirm instantly, undo only if it fails
    try { await navigator.clipboard.writeText(JSON.stringify(doc, null, 2)); setTimeout(() => setCopied(false), 1500); }
    catch { setCopied(false); setError("Clipboard is not available here. Use the JSON download instead."); }
  };

  const working = !!pending;
  return (
    <div className="app">
      <header>
        <div className="brand">DocVision <span>AI</span></div>
        <div className="tag">Finds headings, tables, figures and form fields, and keeps how they relate.</div>
        <InstallPrompt />
        {(result || working) && (
          <button className="icon" data-tip="Start over with another document" aria-label="New document" onClick={reset}><Plus /></button>
        )}
      </header>

      {!result && !working && (
        <main className="start">
          <h1>Upload a document</h1>
          <p className="lede">Drop in a PDF or an image. You get the page back with every region outlined, plus the tables and form values as data.</p>
          <div className={"drop" + (drag ? " over" : "")} role="button" tabIndex={0}
               onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current.click()}
               onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
               onDrop={(e) => { e.preventDefault(); setDrag(false); upload(e.dataTransfer.files[0]); }}
               onClick={() => input.current.click()}>
            <input ref={input} type="file" hidden accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp"
                   onChange={(e) => upload(e.target.files[0])} />
            <div className="big">Drop a file here, or click to choose</div>
            <div className="sub">PDF, PNG, JPG, TIFF, BMP or WebP. Up to 25 MB and 12 pages.</div>
          </div>
          <button className="link" onClick={sample}>Try a generated sample page instead</button>
          {error && <p className="error" role="alert">Error: {error}</p>}
          <p className="note">This is a research prototype. OCR and layout mistakes happen. Use public or permitted documents only, not confidential files or personal records.</p>
        </main>
      )}

      {(result || working) && (
        <main className="work">
          <div className="toolbar">
            <div className="seg" role="tablist" aria-label="View">
              {VIEWS.map(([k, l]) => (
                <button key={k} role="tab" aria-selected={view === k} className={view === k ? "on" : ""} disabled={working}
                        onClick={() => setView(k)}>{l}</button>
              ))}
            </div>
            {result?.pages > 1 && (
              <div className="pager">
                <button className="icon" data-tip="Previous page (left arrow)" aria-label="Previous page" disabled={page === 0} onClick={() => go(-1)}><Prev /></button>
                <span>Page {page + 1} of {result.pages}</span>
                <button className="icon" data-tip="Next page (right arrow)" aria-label="Next page" disabled={page === result.pages - 1} onClick={() => go(1)}><Next /></button>
              </div>
            )}
            <div className="meta">
              {working ? <>Analyzing {pending.label}</> : <>{result.filename}, {fromCache ? "loaded from cache" : `${result.seconds} s`}</>}
            </div>
            {result && (
              <div className="actions">
                <button className="icon" data-tip={copied ? "Copied" : "Copy JSON to clipboard"} aria-label="Copy JSON" onClick={copyJson}>{copied ? <Check /> : <Copy />}</button>
                <button onClick={() => download("document.json", JSON.stringify(doc, null, 2), "application/json")}><Down /> JSON</button>
                <button onClick={() => download("document.md", result.markdown, "text/markdown")}><Down /> Markdown</button>
              </div>
            )}
          </div>
          {error && <p className="error" role="alert">Error: {error}</p>}
          <div className="split">
            <section className={"viewers " + view}>
              {working ? (
                <div className="viewer"><div className="vtitle"><span>Page 1</span><span className="status">Analyzing</span></div>
                  <PageSkeleton previewUrl={pending.previewUrl} /></div>
              ) : (
                <>
                  {(view === "structured" || view === "split") && doc && (
                    <PageViewer title="Structured" img={result.images[page]} page={doc.pages[page]} selected={selected} onSelect={setSelected} />
                  )}
                  {(view === "ocr_only" || view === "split") && base && (
                    <PageViewer title="OCR only (plain text blocks)" img={result.images[page]} page={base.pages[page]} selected={null} onSelect={() => {}} muted />
                  )}
                </>
              )}
            </section>
            {working ? <InspectorSkeleton /> : <Inspector doc={doc} pageIndex={page} selected={selected} onSelect={setSelected} />}
          </div>
        </main>
      )}
    </div>
  );
}

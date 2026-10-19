import React, { useState } from "react";
import { COLORS, CLASS_HELP } from "./PageViewer.jsx";

function Outline({ nodes, depth = 0 }) {
  return (
    <ul className="tree">
      {nodes.map((n, i) => (
        <li key={i}>
          <div style={{ paddingLeft: depth * 12 }}><b>{n.title}</b> <small>p{n.page + 1} · {n.content.length} blocks</small></div>
          {n.children.length > 0 && <Outline nodes={n.children} depth={depth + 1} />}
        </li>
      ))}
    </ul>
  );
}

export default function Inspector({ doc, pageIndex, selected, onSelect }) {
  const [tab, setTab] = useState("elements");
  if (!doc) return <aside className="inspector"><p className="muted">Run in Structured mode to inspect.</p></aside>;
  const page = doc.pages[pageIndex];
  const els = page.elements;
  const forms = els.filter((e) => e.type === "form_field");
  const tables = els.filter((e) => e.type === "table");
  const sel = els.find((e) => e.order === selected);

  return (
    <aside className="inspector">
      <div className="tabs">
        {["elements", "tables", "forms", "outline", "json"].map((t) => (
          <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>{t === "json" ? "JSON" : t}</button>
        ))}
      </div>
      <div className="body">
        {tab === "elements" && (
          <>
            <div className="legend">{Object.entries(COLORS).map(([k, c]) => (
              <span key={k} className="tip" data-tip={CLASS_HELP[k]} tabIndex={0}><i style={{ background: c }} />{k} {els.filter((e) => e.type === k).length}</span>))}</div>
            {els.map((e) => (
              <div key={e.order} className={"row" + (selected === e.order ? " on" : "")} onClick={() => onSelect(e.order)}>
                <span className="chip" style={{ background: COLORS[e.type] }}>{e.type}</span>
                <span className="txt">{e.type === "figure" ? "(figure)" : e.type === "table" ? `${e.cells.length}×${e.cells[0]?.length || 0} table${e.continued ? " (continued)" : ""}` : e.text}</span>
                <span className="conf">{Math.round((e.conf || 0) * 100)}%</span>
              </div>
            ))}
            {sel && <div className="detail"><b>Selected:</b> bbox [{sel.bbox.join(", ")}] · source {sel.source}</div>}
          </>
        )}
        {tab === "tables" && (tables.length ? tables.map((t) => (
          <div key={t.order} className="tbl">
            <div className="muted">Table #{t.order}{t.continued ? " · continued from previous page" : ""}{t.source && t.source.startsWith("word-alignment") ? " · borderless (column alignment)" : ""}</div>
            <table><tbody>{t.cells.map((r, i) => (
              <tr key={i}>{r.map((c, j) => (i === 0 ? <th key={j}>{c}</th> : <td key={j}>{c}</td>))}</tr>))}</tbody></table>
          </div>)) : <p className="muted">No tables on this page.</p>)}
        {tab === "forms" && (forms.length ? (
          <table className="kv"><tbody>{forms.map((f) => (
            <tr key={f.order} onClick={() => onSelect(f.order)}><th>{f.key}</th><td>{f.value || <em>blank</em>} {f.value_type && f.value_type !== "text" && <small className="muted">[{f.value_type}]</small>}</td></tr>))}</tbody></table>
        ) : <p className="muted">No form fields on this page.</p>)}
        {tab === "outline" && (doc.sections.length ? <Outline nodes={doc.sections} /> : <p className="muted">No headings found.</p>)}
        {tab === "json" && <pre className="json">{JSON.stringify(page, (k, v) => (k === "cell_boxes" ? undefined : v), 2)}</pre>}
      </div>
    </aside>
  );
}

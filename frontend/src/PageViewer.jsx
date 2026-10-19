import React from "react";

// Okabe-Ito based: distinguishable under common colour-vision deficiencies. `ink` is the label text colour.
export const COLORS = {
  title: "#c0392b", heading: "#b36b00", paragraph: "#1f6fb2", table: "#0b8457",
  figure: "#a8326e", form_field: "#00798c", list_item: "#5b5f66", code: "#6b4fbb",
};
export const CLASS_HELP = {
  title: "Document title: largest type on the page",
  heading: "Section heading",
  paragraph: "Running body text",
  table: "Table with rows and columns, ruled or not",
  figure: "Picture, chart or diagram",
  form_field: "A label with its filled-in value",
  list_item: "One bullet or numbered item",
  code: "Source code or a monospaced listing",
};

export default function PageViewer({ title, img, page, selected, onSelect, muted, status }) {
  const { width: W, height: H } = page;
  return (
    <div className="viewer">
      <div className="vtitle"><span>{title}</span>{status && <span className="status">{status}</span>}</div>
      <div className="canvas">
        <img src={img} alt={title} />
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
          {page.elements.map((e) => {
            const [x0, y0, x1, y1] = e.bbox, c = COLORS[e.type] || "#000";
            const on = selected === e.order;
            return (
              <g key={e.order} onClick={() => onSelect(on ? null : e.order)} style={{ cursor: "pointer" }}>
                <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} fill={c} fillOpacity={on ? 0.26 : 0.08}
                      stroke={c} strokeWidth={on ? 3 : 2} vectorEffect="non-scaling-stroke" />
                {!muted && e.cell_boxes && on && e.cell_boxes.flat().map((b, i) => (
                  <rect key={i} x={b[0]} y={b[1]} width={b[2] - b[0]} height={b[3] - b[1]} fill="none" stroke={c}
                        strokeWidth="1" vectorEffect="non-scaling-stroke" />
                ))}
                <rect x={x0} y={Math.max(0, y0 - 24)} width={14 + 11 * (String(e.order).length + e.type.length + 1)} height={24} fill={c} />
                <text x={x0 + 6} y={Math.max(0, y0 - 24) + 17} fill="#fff" fontSize="16" fontWeight="600"
                      fontFamily="IBM Plex Mono, monospace">{e.order} {e.type}</text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

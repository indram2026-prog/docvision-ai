import React from "react";

// Placeholder bars in the same arrangement as the real page and inspector, so nothing jumps when data arrives.
const Bar = ({ w = "100%", h = 12, mt = 0 }) => <div className="sk" style={{ width: w, height: h, marginTop: mt }} />;

export function PageSkeleton({ previewUrl }) {
  return (
    <div className="canvas sk-page" aria-busy="true" aria-label="Analyzing page">
      {previewUrl ? <img src={previewUrl} alt="Uploaded page, analysis in progress" /> : (
        <div className="sk-sheet">
          <Bar w="46%" h={26} />
          <Bar w="100%" mt={28} /><Bar w="96%" mt={8} /><Bar w="88%" mt={8} />
          <Bar w="30%" h={18} mt={28} />
          <Bar w="100%" mt={16} /><Bar w="92%" mt={8} />
          <div className="sk" style={{ height: 120, marginTop: 28 }} />
          <Bar w="100%" mt={28} /><Bar w="70%" mt={8} />
        </div>
      )}
    </div>
  );
}

export function InspectorSkeleton() {
  return (
    <aside className="inspector" aria-busy="true" aria-label="Loading results">
      <div className="tabs"><Bar w="62%" h={26} /></div>
      <div className="body">
        {[70, 92, 55, 84, 64, 78, 48].map((w, i) => (
          <div className="row" key={i}><Bar w={64} h={18} /><Bar w={`${w}%`} h={14} /></div>
        ))}
      </div>
    </aside>
  );
}

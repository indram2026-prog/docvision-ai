import React from "react";
const p = { width: 16, height: 16, viewBox: "0 0 16 16", fill: "none", stroke: "currentColor", strokeWidth: 1.5,
            strokeLinecap: "round", strokeLinejoin: "round", "aria-hidden": true };
export const Prev = () => <svg {...p}><path d="M10 3 5 8l5 5" /></svg>;
export const Next = () => <svg {...p}><path d="m6 3 5 5-5 5" /></svg>;
export const Copy = () => <svg {...p}><rect x="5.5" y="5.5" width="8" height="8" rx="1" /><path d="M10.5 5.5V3.5a1 1 0 0 0-1-1h-6a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2" /></svg>;
export const Down = () => <svg {...p}><path d="M8 2.5v8M4.5 7.5 8 11l3.5-3.5M3 13.5h10" /></svg>;
export const Plus = () => <svg {...p}><path d="M8 3v10M3 8h10" /></svg>;
export const Check = () => <svg {...p}><path d="m3 8.5 3.2 3L13 4.5" /></svg>;

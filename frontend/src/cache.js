// Result cache: same file + same mode = instant, no network. IndexedDB with an in-memory fallback.
// Keys are a SHA-256 of the file bytes, so a renamed copy of the same file still hits.
const DB = "docvision", STORE = "results", MAX = 12;
const mem = new Map();

const open = () => new Promise((res, rej) => {
  try {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE);
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  } catch (e) { rej(e); }
});
const tx = (db, mode, fn) => new Promise((res, rej) => {
  const t = db.transaction(STORE, mode), s = t.objectStore(STORE);
  const out = fn(s);
  t.oncomplete = () => res(out && "result" in out ? out.result : undefined);
  t.onerror = () => rej(t.error);
});

export async function fileKey(file, mode = "both") {
  try {
    const buf = await file.arrayBuffer();
    const h = await crypto.subtle.digest("SHA-256", buf);
    return mode + ":" + [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, "0")).join("");
  } catch {
    return `${mode}:${file.name}:${file.size}:${file.lastModified}`;   // insecure context fallback
  }
}

export async function getCached(key) {
  if (mem.has(key)) return mem.get(key);
  try {
    const db = await open();
    const v = await tx(db, "readonly", (s) => s.get(key));
    if (v) mem.set(key, v.data);
    return v?.data ?? null;
  } catch { return null; }
}

export async function putCached(key, data) {
  mem.set(key, data);
  try {
    const db = await open();
    await tx(db, "readwrite", (s) => s.put({ data, ts: Date.now() }, key));
    const all = await tx(db, "readonly", (s) => s.getAll());
    const keys = await tx(db, "readonly", (s) => s.getAllKeys());
    if (keys.length > MAX) {                       // evict the oldest entries
      const order = keys.map((k, i) => [k, all[i].ts]).sort((a, b) => a[1] - b[1]).slice(0, keys.length - MAX);
      await tx(db, "readwrite", (s) => order.forEach(([k]) => s.delete(k)));
    }
  } catch { /* cache is an optimisation, never a requirement */ }
}

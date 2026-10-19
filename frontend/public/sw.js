// DocVision service worker: makes the app installable and the UI shell load offline.
// It never touches /api (analysis always needs the server) or cross-origin requests.
const CACHE = "docvision-shell-v1";
const SHELL = ["/", "/manifest.webmanifest", "/icons/icon-192.png", "/icons/icon-512.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});
self.addEventListener("fetch", (e) => {
  const req = e.request, url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== self.location.origin || url.pathname.startsWith("/api")) return;
  if (req.mode === "navigate") {            // page loads: network first so deploys show up, cache as offline fallback
    e.respondWith(fetch(req).then((r) => { caches.open(CACHE).then((c) => c.put("/", r.clone())); return r; })
      .catch(() => caches.match("/")));
    return;
  }
  e.respondWith(                             // static assets (Vite hashes filenames): cache first
    caches.match(req).then((hit) => hit || fetch(req).then((r) => {
      if (r.ok) caches.open(CACHE).then((c) => c.put(req, r.clone()));
      return r;
    }))
  );
});

// THURSDAY's service worker: the installed app opens from cache instantly, then refreshes from the server
// (network first for the page, so a new version shows on the next open). Offline: the last copy, or a short
// "can't reach THURSDAY" screen. API calls, the voice socket and Twilio are never cached.
const CACHE = "thursday-v37";  // bump on changes the installed app must pick up at once
const SHELL = ["/", "/assets/manifest.json?v=2", "/assets/icons/icon-192.png?v=2", "/assets/icons/apple-touch-icon.png?v=2"];
const OFFLINE = `<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1">
<title>THURSDAY</title><body style="margin:0;height:100vh;display:flex;align-items:center;justify-content:center;
background:#12110E;color:#F0ECE3;font:16px -apple-system,system-ui,sans-serif;text-align:center;padding:24px">
<div><p style="font-size:20px;margin:0 0 8px">Can't reach THURSDAY</p><p style="color:#8C877C;margin:0">
Check Wi-Fi or Tailscale, then pull down to retry.</p></div>`;

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;
  if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/ws/") || url.pathname.startsWith("/twilio/")) return;
  if (e.request.mode === "navigate") {
    e.respondWith(fetch(e.request).then((res) => {
      caches.open(CACHE).then((c) => c.put("/", res.clone()));
      return res;
    }).catch(() => caches.match("/").then((hit) => hit || new Response(OFFLINE, { headers: { "Content-Type": "text/html" } }))));
    return;
  }
  if (url.pathname.startsWith("/assets/")) {  // icons, fonts, sounds, splash: versioned, cache first
    e.respondWith(caches.match(e.request).then((hit) => hit || fetch(e.request).then((res) => {
      if (res.ok) caches.open(CACHE).then((c) => c.put(e.request, res.clone()));
      return res;
    })));
  }
});

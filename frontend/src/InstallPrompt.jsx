import React, { useEffect, useState } from "react";

// Shows an "Install app" button when the browser says the PWA can be installed
// (Chrome/Edge/Android). iOS Safari has no such event: use Share -> Add to Home Screen.
export default function InstallPrompt() {
  const [evt, setEvt] = useState(null);
  useEffect(() => {
    const on = (e) => { e.preventDefault(); setEvt(e); };
    const done = () => setEvt(null);
    window.addEventListener("beforeinstallprompt", on);
    window.addEventListener("appinstalled", done);
    return () => { window.removeEventListener("beforeinstallprompt", on); window.removeEventListener("appinstalled", done); };
  }, []);
  if (!evt) return null;
  return (
    <button className="install" onClick={async () => { evt.prompt(); await evt.userChoice; setEvt(null); }}>
      Install app
    </button>
  );
}

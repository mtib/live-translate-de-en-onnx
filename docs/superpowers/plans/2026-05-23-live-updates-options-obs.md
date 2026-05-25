# LiveTranslate ONNX: Live Web Updates + Options Window + OBS Overlay

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Worktree first:** Before touching any code, create a git worktree:
> ```bash
> cd ~/Code/live-translate-de-en-onnx
> REPO="live-translate-de-en-onnx"
> BRANCH="feat/live-updates-options-obs"
> SLUG="${BRANCH//\//-}"
> WT=~/Code/worktrees/"${REPO}-${SLUG}"
> git fetch origin
> git worktree add -b "$BRANCH" "$WT" origin/main
> cd "$WT"
> ```
> All edits, builds, and commits happen inside the worktree.

**Goal:** Three features for `live-translate-de-en-onnx`: (1) mobile web listen page shows rolling partial hypothesis (including partial translation) updated in real time, with stronger typography and auto-scroll; (2) options window accessible from the menu bar lets users tune font sizes, text colors, layout (mixed vs. side-by-side), and window opacity; (3) an OBS browser-source overlay endpoint that pushes German-translated subtitles whenever English speech is detected in any finalized sentence.

**Architecture:** All three share the `LiveAudioServer` HTTP layer and `Pipeline` data-flow. Feature 1 adds named SSE events (`hypothesis`, `hypothesis-done`) alongside the existing finalized-sentence SSE. Feature 2 introduces `AppSettings` (ObservableObject + AppStorage) propagated via SwiftUI environment objects, with a `Settings` SwiftUI scene. Feature 3 adds `/obs` + `/obs-events` routes to `LiveAudioServer` (always started, not gated on TTS), uses `NLLanguageRecognizer` for on-device English detection, and a second `AppleTranslator` instance fed by a second `.translationTask` modifier for the en→de pair.

**Key ONNX-repo differences from the Whisper variant:**
- `SherpaTranscriber` streams `.partial(text:, translation:)` — hypothesis SSE can carry live partial translation text, not just a state label
- Sentences **reuse** the chunk UUID (not a fresh UUID at graduation) — hypothesis and finalized rows have the same DOM key; graduation is an in-place content swap
- `OnnxTTSSpeaker` is already gated on `audioListenerCount > 0`; TTS never fires for OBS subscribers
- Source/target language is a compile-time constant from `ModelConfig` (`de`→`en`); runtime `NLLanguageRecognizer` is still needed for OBS because system audio may carry English regardless of the ASR model's primary language

**Tech Stack:** SwiftUI, `NaturalLanguage.NLLanguageRecognizer`, `Translation.TranslationSession`, `Network.NWListener`, `AppStorage`, `CoreImage`.

---

## File map

| Action | File | What changes |
|---|---|---|
| Modify | `Sources/LiveTranslate/LiveAudioServer.swift` | Hypothesis SSE methods; OBS `/obs` + `/obs-events` routes; `obsPageHTML`; always start server |
| Modify | `Sources/LiveTranslate/Pipeline.swift` | Emit hypothesis SSE on each lifecycle event; OBS English-detect + translate on graduate; `liveOBSURL` published property; always start server |
| Create | `Sources/LiveTranslate/AppSettings.swift` | `ObservableObject` with `@AppStorage` for font sizes, colors, layout mode, opacity |
| Create | `Sources/LiveTranslate/OptionsView.swift` | SwiftUI `Settings` form with sliders, `ColorPicker`s, segmented layout picker |
| Modify | `Sources/LiveTranslate/App.swift` | Add `Settings` scene; inject `AppSettings` as environment object; observe opacity |
| Modify | `Sources/LiveTranslate/TranscriptView.swift` | Read settings from environment; side-by-side layout; second `.translationTask` for en→de |
| Modify | `Sources/LiveTranslate/MenuBarView.swift` | Add gear/settings button |

---

## Feature 1: Mobile Web Live Hypothesis Updates

### Task 1: Add hypothesis SSE methods to `LiveAudioServer`

**Files:**
- Modify: `Sources/LiveTranslate/LiveAudioServer.swift`

SSE protocol supports named events:
```
event: hypothesis
data: {"id":"<chunk-uuid>","state":"partial","text":"Ich gehe","translation":"I go"}

event: hypothesis-done
data: {"id":"<chunk-uuid>"}
```

`EventSource` in JavaScript listens via `es.addEventListener('hypothesis', ...)`. Hypothesis events are **not** added to `eventReplay` — they're transient state that is meaningless after reconnect.

Because the ONNX repo reuses the chunk UUID for the graduated sentence, the JS can update the hypothesis row in-place when the finalized sentence SSE event arrives (same `start|end|transcription` key), then `hypothesis-done` cleans up the hypothesis CSS class.

- [ ] **Step 1: Add `obsSubscribers` dict and `publishHypothesis`/`publishHypothesisDone`/`publishOBSSubtitle` to `LiveAudioServer`**

In `LiveAudioServer.swift`, alongside `eventSubscribers` add:

```swift
/// Clients consuming the OBS subtitle SSE stream (`/obs-events`).
private var obsSubscribers: [UUID: NWConnection] = [:]
```

After the `publishTranscript(jsonLine:)` method, add:

```swift
/// Broadcast an inflight-chunk state update to all `/events` SSE subscribers.
/// Not replayed — hypothesis events are transient UI state.
/// `text` is the current partial transcription; `translation` is the throttled
/// partial translation (nil if not yet available).
func publishHypothesis(id: UUID, source: String, state: String,
                       text: String? = nil, translation: String? = nil) {
    var json = #"{"id":"\#(id.uuidString)","source":"\#(source)","state":"\#(state)""#
    if let text        { json += #","text":"\#(jsonEscape(text))""# }
    if let translation { json += #","translation":"\#(jsonEscape(translation))""# }
    json += "}"
    broadcastToEventSubscribers("event: hypothesis\ndata: \(json)\n\n")
}

/// Tell SSE subscribers that a hypothesis row graduated (chunk finalized).
/// Call this just before publishing the finalized sentence so the JS can
/// remove the hypothesis CSS class before the content-swap event arrives.
func publishHypothesisDone(id: UUID) {
    broadcastToEventSubscribers("event: hypothesis-done\ndata: {\"id\":\"\(id.uuidString)\"}\n\n")
}

/// Push a German subtitle line to every OBS subscriber.
/// Not replayed — subtitles are transient overlays.
func publishOBSSubtitle(germanText: String) {
    let data = "data: {\"text\":\"\(jsonEscape(germanText))\"}\n\n".data(using: .utf8) ?? Data()
    lock.lock()
    let conns = Array(obsSubscribers.values)
    lock.unlock()
    for c in conns {
        c.send(content: data, completion: .contentProcessed { err in if err != nil { c.cancel() } })
    }
}

/// Shared helper: broadcast raw SSE text to all `/events` subscribers.
private func broadcastToEventSubscribers(_ raw: String) {
    guard let data = raw.data(using: .utf8) else { return }
    lock.lock()
    let conns = Array(eventSubscribers.values)
    lock.unlock()
    for c in conns {
        c.send(content: data, completion: .contentProcessed { err in if err != nil { c.cancel() } })
    }
}

/// Minimal JSON string escaping for inline use in SSE data fields.
private func jsonEscape(_ s: String) -> String {
    s.replacingOccurrences(of: "\\", with: "\\\\")
     .replacingOccurrences(of: "\"", with: "\\\"")
     .replacingOccurrences(of: "\n", with: "\\n")
     .replacingOccurrences(of: "\r", with: "\\r")
}
```

- [ ] **Step 2: Update `stop()` to clean up `obsSubscribers`**

Replace the existing `stop()` method:

```swift
func stop() {
    heartbeatTask?.cancel()
    heartbeatTask = nil
    lock.lock()
    let audio  = Array(audioSubscribers.values)
    let events = Array(eventSubscribers.values)
    let obs    = Array(obsSubscribers.values)
    audioSubscribers.removeAll()
    eventSubscribers.removeAll()
    obsSubscribers.removeAll()
    eventReplay.removeAll()
    lock.unlock()
    for c in audio  { c.cancel() }
    for c in events { c.cancel() }
    for c in obs    { c.cancel() }
    listener?.cancel()
    listener = nil
}
```

- [ ] **Step 3: Add `/obs` and `/obs-events` routes to the request dispatcher**

In `handle(_:)`, find the `switch path {` block and add two cases:

```swift
case "/obs":         self.serveOBSPage(conn)
case "/obs-events":  self.serveOBSEventStream(conn)
```

- [ ] **Step 4: Implement `serveOBSPage` and `serveOBSEventStream`**

```swift
private func serveOBSPage(_ conn: NWConnection) {
    let body = Self.obsPageHTML.data(using: .utf8) ?? Data()
    var resp = "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: \(body.count)\r\nCache-Control: no-cache, no-store\r\nConnection: close\r\n\r\n".data(using: .utf8)!
    resp.append(body)
    conn.send(content: resp, completion: .contentProcessed { _ in conn.cancel() })
}

private func serveOBSEventStream(_ conn: NWConnection) {
    let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache, no-store\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n".data(using: .utf8)!
    var preamble = headers
    preamble.append(": ready\n\n".data(using: .utf8)!)
    conn.send(content: preamble, completion: .contentProcessed { [weak self] err in
        guard let self else { return }
        if err != nil { conn.cancel(); return }
        let id = UUID()
        self.lock.lock()
        self.obsSubscribers[id] = conn
        self.lock.unlock()
        Log.line("LiveAudioServer: OBS subscriber +1 (id=\(id.uuidString.prefix(8)))")
        conn.stateUpdateHandler = { [weak self] state in
            switch state {
            case .cancelled, .failed:
                self?.lock.lock()
                self?.obsSubscribers.removeValue(forKey: id)
                self?.lock.unlock()
            default: break
            }
        }
    })
}
```

- [ ] **Step 5: Extend the SSE heartbeat to also ping OBS subscribers**

In `startHeartbeat()`, after the block that sends `sseHeartbeat` to `evConns`, add:

```swift
self.lock.lock()
let obsConns = Array(self.obsSubscribers.values)
self.lock.unlock()
for c in obsConns {
    c.send(content: sseHeartbeat, completion: .contentProcessed { err in
        if err != nil { c.cancel() }
    })
}
```

- [ ] **Step 6: Add `obsPageHTML` string constant**

After the closing `"""` of `listenPageHTML`, add:

```swift
/// Single-file HTML page for OBS Browser Source.
/// Set the OBS source dimensions to match your overlay area (e.g. 1920×1080).
/// Background is `transparent` so it composites over the webcam feed.
private static let obsPageHTML: String = """
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>LiveTranslate OBS Overlay</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: transparent;
  width: 100vw; height: 100vh;
  overflow: hidden;
  font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
}
#subtitle {
  position: fixed;
  bottom: 8%;
  left: 0; right: 0;
  text-align: center;
  padding: 0 80px;
  font-size: 56px;
  font-weight: 700;
  line-height: 1.25;
  color: #ffffff;
  text-shadow:
    -3px -3px 0 #000,  3px -3px 0 #000,
    -3px  3px 0 #000,  3px  3px 0 #000,
    -3px  0   0 #000,  3px  0   0 #000,
     0   -3px 0 #000,  0    3px 0 #000;
  opacity: 0;
  transition: opacity 0.25s ease-in-out;
  word-wrap: break-word;
}
#subtitle.visible { opacity: 1; }
</style>
</head>
<body>
<div id="subtitle"></div>
<script>
(function () {
  const sub = document.getElementById('subtitle');
  let hideTimer = null;
  function show(text) {
    clearTimeout(hideTimer);
    sub.textContent = text;
    sub.classList.add('visible');
    hideTimer = setTimeout(() => sub.classList.remove('visible'), 6000);
  }
  function connect() {
    const es = new EventSource('/obs-events');
    es.onmessage = (e) => {
      try { const d = JSON.parse(e.data); if (d.text) show(d.text); } catch {}
    };
    es.onerror = () => setTimeout(connect, 2000);
  }
  connect();
})();
</script>
</body>
</html>
"""
```

- [ ] **Step 7: Build to confirm no compile errors**

```bash
cd ~/Code/worktrees/live-translate-de-en-onnx-feat-live-updates-options-obs
LIVETRANSLATE_SIGN_IDENTITY=LiveTranslateDev ./build.sh 2>&1 | tail -20
```

Expected: `Build complete!`

- [ ] **Step 8: Commit**

```bash
git add Sources/LiveTranslate/LiveAudioServer.swift
git commit -m "feat: hypothesis + OBS SSE methods, /obs and /obs-events routes"
```

---

### Task 2: Emit hypothesis SSE from `Pipeline` lifecycle events

**Files:**
- Modify: `Sources/LiveTranslate/Pipeline.swift`

The ONNX repo's `.partial(text:, translation:)` state is richer than Whisper's — the hypothesis SSE event carries both the partial transcription text AND the partial translation when available. This lets the mobile web page show rolling translated subtitles in real time, not just a state label.

`graduate()` reuses the chunk UUID for the sentence, so on the JS side the finalized sentence event (`onmessage`) can upgrade the hypothesis row in-place. `publishHypothesisDone` removes the hypothesis CSS class so the row renders as finalized styling.

- [ ] **Step 1: Add hypothesis calls to `applyLifecycle`**

In `Pipeline.swift`, modify `applyLifecycle` case by case:

`case .listening:` — after the `inflightChunks.append(...)`:
```swift
liveAudioServer?.publishHypothesis(id: id, source: source.rawValue, state: "listening")
```

`case .partial(let text):` — after updating `inflightChunks[idx].state`, add (place it right before the throttle guard so it fires on every partial, not just translated ones):
```swift
// Publish hypothesis SSE — send whatever translation we have so far (may be nil).
let currentTranslation: String?
if case .partial(_, let t) = inflightChunks[idx].state { currentTranslation = t }
else { currentTranslation = nil }
liveAudioServer?.publishHypothesis(id: id, source: source.rawValue, state: "partial",
                                   text: text, translation: currentTranslation)
```

`case .completed(...)` — at the point where we set `.translating(text:)` state, add:
```swift
liveAudioServer?.publishHypothesis(id: id, source: source.rawValue,
                                   state: "translating", text: text)
```

Also at the `inflightChunks[idx].state = .partial(text: text, translation: t)` hold-partial path, add:
```swift
liveAudioServer?.publishHypothesis(id: id, source: source.rawValue, state: "partial",
                                   text: text, translation: t)
```

- [ ] **Step 2: Call `publishHypothesisDone` in `graduate()`**

In `graduate(id:source:text:translation:createdAt:endsAt:)`, before `recordSentence(sentence)`:

```swift
liveAudioServer?.publishHypothesisDone(id: id)
recordSentence(sentence)
```

- [ ] **Step 3: Add `liveOBSURL` published property and always start the server**

In `Pipeline.swift`, alongside `liveStreamURL` add:

```swift
/// URL for the OBS browser-source overlay (`/obs`). Always set while a run
/// is active — not gated on TTS voice. Displayed in the share popover.
@Published private(set) var liveOBSURL: String?
```

In `run()`, find the `if srcLangCode != tgtLangCode, let voice = ...` block that creates `LiveAudioServer`. Replace it so the server always starts and TTS is layered on top:

```swift
// Always start the HTTP server so the listen page and OBS overlay are
// available every run. TTS is layered on top only when a suitable voice
// is installed and audioListenerCount > 0 — it never fires for OBS subscribers.
let server = LiveAudioServer(port: liveStreamPort)
do {
    try server.start()
    self.liveAudioServer = server
    self.liveOBSURL = LiveAudioServer.streamURL(port: liveStreamPort) + "obs"

    if srcLangCode != tgtLangCode,
       let voice = OnnxTTSSpeaker.bestVoice(forTargetCode: tgtLangCode) {
        let speaker = OnnxTTSSpeaker(voice: voice, onPCM: { [weak server] pcm in
            server?.append(pcm)
        }, onActivityChanged: { [weak server] active in
            server?.setSpeaking(active)
        }, onModelLoaded: { [weak self] in
            Task { @MainActor [weak self] in
                self?.ttsModelLoaded = true
                self?.recomputeTTSActive()
            }
        })
        server.onAudioListenerCountChanged = { [weak self] count in
            Task { @MainActor [weak self] in
                self?.ttsListenerCount = count
                self?.recomputeTTSActive()
            }
        }
        self.ttsSpeaker = speaker
        self.liveStreamURL = LiveAudioServer.streamURL(port: liveStreamPort)
        Log.line("Live audio stream: \(self.liveStreamURL ?? "?") voice=\(voice)")
    } else {
        Log.line("Live audio stream: TTS skipped (no voice for '\(tgtLangCode)' or src==tgt)")
    }
    Log.line("OBS overlay: \(self.liveOBSURL ?? "?")")
} catch {
    Log.line("LiveAudioServer.start failed: \(error.localizedDescription)")
}
```

Also add `liveOBSURL = nil` to the `defer` block alongside `liveStreamURL = nil`.

> **Note:** The exact `OnnxTTSSpeaker` initializer signature may differ from above — read `OnnxTTSSpeaker.swift` and adjust the constructor call to match the existing parameters. The key change is moving server creation outside the voice-availability guard.

- [ ] **Step 4: Build**

```bash
LIVETRANSLATE_SIGN_IDENTITY=LiveTranslateDev ./build.sh 2>&1 | tail -20
```

Expected: `Build complete!`

- [ ] **Step 5: Commit**

```bash
git add Sources/LiveTranslate/Pipeline.swift
git commit -m "feat: emit hypothesis SSE on each lifecycle event; always start server; liveOBSURL"
```

---

### Task 3: Rewrite the listen page HTML

**Files:**
- Modify: `Sources/LiveTranslate/LiveAudioServer.swift` (the `listenPageHTML` string)

Key changes vs. the current page:
- Drop the `.meta` div (time + source label)
- Increase translation font to 19 px, weight 600, color `#f8f8f8`
- Increase transcription caption color to `#aaa`
- Handle `hypothesis` named SSE events: show rolling partial translation as the primary line, partial transcription as the caption
- Handle `hypothesis-done`: remove hypothesis CSS class (row upgrades to finalized styling in-place on the next `onmessage`)
- Auto-scroll fires on every hypothesis update, not just finalized sentences
- Since the ONNX repo reuses chunk UUID for sentences, the finalized sentence key (`start|end|transcription`) uniquely identifies it — no UUID tracking needed in the sentence path

- [ ] **Step 1: Replace `listenPageHTML`**

In `LiveAudioServer.swift`, replace the entire `private static let listenPageHTML: String = """..."""` block with:

```swift
private static let listenPageHTML: String = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#0a0a0a">
<title>LiveTranslate</title>
<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
html, body {
  margin: 0; padding: 0;
  background: #0a0a0a; color: #f8f8f8;
  font: 16px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  min-height: 100vh; min-height: 100dvh;
}
body {
  display: flex; flex-direction: column;
  padding-top: env(safe-area-inset-top);
  padding-bottom: env(safe-area-inset-bottom);
}
header {
  position: sticky; top: 0;
  background: rgba(10,10,10,0.92);
  backdrop-filter: saturate(180%) blur(12px);
  -webkit-backdrop-filter: saturate(180%) blur(12px);
  padding: 14px 16px;
  border-bottom: 1px solid #1f1f1f;
  display: flex; align-items: center; gap: 14px;
  z-index: 10;
}
#play {
  flex: 0 0 auto; min-width: 110px; height: 44px;
  border-radius: 22px; border: 1.5px solid #f8f8f8;
  background: transparent; color: #f8f8f8;
  font-size: 15px; font-weight: 600; cursor: pointer;
  transition: background 0.12s, color 0.12s, transform 0.04s, border-color 0.12s;
}
#play:active { transform: scale(0.96); }
#play.live   { background: #2ecc40; color: #000; border-color: #2ecc40; }
#play.behind { background: #ff851b; color: #000; border-color: #ff851b; }
#play.error  { background: #ff4136; color: #000; border-color: #ff4136; }
.status { display: flex; align-items: center; gap: 8px; font-size: 13px; color: #999; flex: 1; min-width: 0; }
.status .text { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: #666; flex: 0 0 auto; }
.dot.live   { background: #2ecc40; box-shadow: 0 0 6px rgba(46,204,64,0.7); }
.dot.behind { background: #ff851b; }
.dot.error  { background: #ff4136; }
main { flex: 1; padding: 14px 16px 80px; overflow-y: auto; }
.row { padding: 10px 0; border-bottom: 1px solid #1a1a1a; }
.row:last-child { border-bottom: none; }
.row .translation {
  font-size: 19px; font-weight: 600; line-height: 1.35;
  color: #f8f8f8; word-wrap: break-word;
}
.row .transcription {
  font-size: 13px; font-weight: 400; line-height: 1.35;
  color: #aaa; margin-top: 3px; word-wrap: break-word;
}
/* Hypothesis rows: muted italic while in-flight */
.row.hypothesis .translation { color: #999; font-style: italic; font-weight: 400; }
.row.hypothesis .transcription { color: #666; }
/* Fade-in for new finalized rows */
@keyframes fadein {
  from { opacity: 0; transform: translateY(4px); }
  to   { opacity: 1; transform: translateY(0); }
}
.row.finalized { animation: fadein 0.18s ease-out; }
.empty { text-align: center; color: #555; padding: 60px 16px; font-size: 14px; }
audio { display: none; }
</style>
</head>
<body>
<header>
  <button id="play" type="button">Listen</button>
  <div class="status"><span id="dot" class="dot"></span><span id="text" class="text">Idle</span></div>
</header>
<main id="list"><div class="empty">Waiting for the first sentence…</div></main>
<audio id="audio" preload="none"></audio>
<script>
(function () {
  const STREAM = '/live.wav';
  const EVENTS = '/events';
  const MAX_LATENCY = 3.0;
  const audio = document.getElementById('audio');
  const btn   = document.getElementById('play');
  const dot   = document.getElementById('dot');
  const txt   = document.getElementById('text');
  const list  = document.getElementById('list');
  let mode = 'idle';

  function setMode(m, label) {
    mode = m;
    btn.classList.remove('live','behind','error');
    dot.classList.remove('live','behind','error');
    if (m === 'live')        { btn.textContent = 'Live';   btn.classList.add('live');   dot.classList.add('live');   }
    else if (m === 'behind') { btn.textContent = 'Resync'; btn.classList.add('behind'); dot.classList.add('behind'); }
    else if (m === 'error')  { btn.textContent = 'Retry';  btn.classList.add('error');  dot.classList.add('error');  }
    else if (m === 'connecting') { btn.textContent = '…'; }
    else                     { btn.textContent = 'Listen'; }
    if (label) txt.textContent = label;
  }
  function start() {
    setMode('connecting', 'Connecting…');
    audio.src = STREAM + '?t=' + Date.now();
    audio.load();
    audio.play().then(() => setMode('live', 'Streaming')).catch(() => setMode('error', 'Tap to retry'));
  }
  function resync() {
    try { audio.pause(); } catch (e) {}
    audio.src = ''; start();
  }
  btn.addEventListener('click', () => {
    if (mode === 'live')        { audio.pause(); setMode('idle', 'Paused'); }
    else if (mode === 'behind') { resync(); }
    else                        { start(); }
  });
  setInterval(() => {
    if (mode !== 'live') return;
    const b = audio.buffered; if (!b.length) return;
    const end = b.end(b.length - 1); const lag = end - audio.currentTime;
    if (lag > MAX_LATENCY) {
      try { audio.currentTime = end - 0.1; } catch (e) {}
      if (audio.buffered.length && audio.buffered.end(audio.buffered.length - 1) - audio.currentTime > MAX_LATENCY) { resync(); return; }
    }
    txt.textContent = 'Streaming · ' + lag.toFixed(1) + 's';
  }, 500);
  audio.addEventListener('error',   () => { if (mode === 'live') setMode('error', 'Audio error'); });
  audio.addEventListener('ended',   () => { if (mode === 'live') { setMode('error', 'Disconnected'); setTimeout(resync, 600); } });
  audio.addEventListener('stalled', () => { if (mode === 'live') setMode('behind', 'Stalled'); });
  document.addEventListener('visibilitychange', () => { if (!document.hidden && mode === 'live') resync(); });
  let wake = null;
  audio.addEventListener('playing', async () => { if ('wakeLock' in navigator) try { wake = await navigator.wakeLock.request('screen'); } catch (e) {} });
  audio.addEventListener('pause',   () => { if (wake) { wake.release(); wake = null; } });

  // ── Transcript ──────────────────────────────────────────────────────────────
  // Finalized sentences:  default onmessage (no event: prefix from server)
  // Hypothesis updates:   es.addEventListener('hypothesis', ...)
  // Hypothesis finalized: es.addEventListener('hypothesis-done', ...)
  //
  // Because the ONNX backend reuses the chunk UUID for the graduated sentence,
  // graduation looks like: hypothesis-done → onmessage with the same chunk id.
  // We store hypothesis rows by chunk UUID and remove the hypothesis class when
  // hypothesis-done fires — the next onmessage fills in the finalized content.

  const MAX_ROWS = 200;
  const seen = new Set();                  // keyed by "start|end|transcription"
  const hypothesisRows = new Map();        // chunkUUID → DOM div

  function scrollToBottom() {
    const nearBottom = window.scrollY + window.innerHeight >= document.body.scrollHeight - 160;
    if (nearBottom) window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
  }

  function trimRows() {
    const rows = [...list.querySelectorAll('.row.finalized')];
    while (rows.length > MAX_ROWS) {
      const old = rows.shift();
      if (old.dataset.key) seen.delete(old.dataset.key);
      old.remove();
    }
  }

  // Hypothesis update: create or update the hypothesis row for this chunk.
  function onHypothesis(d) {
    const empty = list.querySelector('.empty');
    if (empty) empty.remove();
    let row = hypothesisRows.get(d.id);
    if (!row) {
      row = document.createElement('div');
      row.className = 'row hypothesis';
      list.appendChild(row);
      hypothesisRows.set(d.id, row);
    }
    // Primary line: translation if we have it, otherwise the partial transcription.
    let tDiv = row.querySelector('.translation');
    if (!tDiv) { tDiv = document.createElement('div'); tDiv.className = 'translation'; row.appendChild(tDiv); }
    tDiv.textContent = (d.translation && d.translation.length > 0) ? d.translation
                     : (d.text || stateLabel(d.state));
    // Caption: transcription text (only when translation is showing in primary).
    let sDiv = row.querySelector('.transcription');
    if (d.translation && d.text && d.translation !== d.text) {
      if (!sDiv) { sDiv = document.createElement('div'); sDiv.className = 'transcription'; row.appendChild(sDiv); }
      sDiv.textContent = d.text;
    } else if (sDiv) {
      sDiv.remove();
    }
    scrollToBottom();
  }
  function stateLabel(s) {
    if (s === 'listening')   return 'listening…';
    if (s === 'partial')     return 'transcribing…';
    if (s === 'translating') return 'translating…';
    return s;
  }

  // Hypothesis done: remove the hypothesis class so the row is ready for
  // the finalized content that arrives in the next onmessage event.
  function onHypothesisDone(d) {
    const row = hypothesisRows.get(d.id);
    if (row) {
      row.classList.remove('hypothesis');
      row.classList.add('finalized');
      hypothesisRows.delete(d.id);
    }
  }

  // Finalized sentence: fill in authoritative content. If a hypothesis row
  // already exists (upgraded by hypothesis-done), update it in-place.
  // Otherwise add a new row.
  function onFinalized(rec) {
    const key = (rec.start || '') + '|' + (rec.end || '') + '|' + (rec.transcription || '');
    if (seen.has(key)) return;
    seen.add(key);
    const empty = list.querySelector('.empty');
    if (empty) empty.remove();

    // Try to reuse an upgraded hypothesis row (hypothesis-done already fired).
    let row = [...list.querySelectorAll('.row.finalized')].find(r => !r.dataset.key);
    if (!row) {
      row = document.createElement('div');
      row.className = 'row finalized';
      list.appendChild(row);
    }
    row.dataset.key = key;
    row.innerHTML = '';

    const t = document.createElement('div');
    t.className = 'translation';
    t.textContent = rec.translation || rec.transcription || '';
    row.appendChild(t);

    if (rec.translation && rec.transcription && rec.translation !== rec.transcription) {
      const s = document.createElement('div');
      s.className = 'transcription';
      s.textContent = rec.transcription;
      row.appendChild(s);
    }

    // If row was appended (not reused), make sure it's at the bottom.
    if (row.parentElement !== list) list.appendChild(row);
    trimRows();
    scrollToBottom();
  }

  function connectEvents() {
    const es = new EventSource(EVENTS);
    es.onmessage = (e) => {
      try { onFinalized(JSON.parse(e.data)); } catch {}
    };
    es.addEventListener('hypothesis', (e) => {
      try { onHypothesis(JSON.parse(e.data)); } catch {}
    });
    es.addEventListener('hypothesis-done', (e) => {
      try { onHypothesisDone(JSON.parse(e.data)); } catch {}
    });
  }
  connectEvents();
})();
</script>
</body>
</html>
"""
```

- [ ] **Step 2: Build and manually test**

```bash
LIVETRANSLATE_SIGN_IDENTITY=LiveTranslateDev ./build.sh 2>&1 | tail -20
open build/LiveTranslate.app
```

Start a session and open `http://<host>:8765/` in Safari on a phone. Verify:
- As you speak, a hypothesis row appears immediately with "listening…"
- Partial transcription appears within ~100 ms of words being spoken
- Partial translation fills in (italic/dimmed) within ~1 s
- When the sentence finalizes, the row transitions to bright finalized styling in-place
- Auto-scroll keeps the active row in view

- [ ] **Step 3: Commit**

```bash
git add Sources/LiveTranslate/LiveAudioServer.swift
git commit -m "feat: rewrite listen page — live partial hypothesis with translation, stronger typography"
```

---

## Feature 2: Options Window

### Task 4: Create `AppSettings`

**Files:**
- Create: `Sources/LiveTranslate/AppSettings.swift`

- [ ] **Step 1: Create `AppSettings.swift`**

```swift
import SwiftUI

/// Persisted display preferences. Inject as `.environmentObject(settings)` at the
/// app root so `TranscriptView`, `MenuBarView`, and row views all read from the
/// same source without prop-drilling.
final class AppSettings: ObservableObject {

    @AppStorage("settings.transcriptFontSize")   var transcriptFontSize: Double  = 13
    @AppStorage("settings.translationFontSize")  var translationFontSize: Double  = 16
    @AppStorage("settings.windowOpacity")         var windowOpacity: Double        = 0.7
    @AppStorage("settings.layoutModeRaw")         var layoutModeRaw: String        = LayoutMode.mixed.rawValue
    @AppStorage("settings.transcriptColorHex")    var transcriptColorHex: String   = "#8a8a8a"
    @AppStorage("settings.translationColorHex")   var translationColorHex: String  = "#eeeeee"

    var layoutMode: LayoutMode {
        get { LayoutMode(rawValue: layoutModeRaw) ?? .mixed }
        set { layoutModeRaw = newValue.rawValue }
    }

    var transcriptColor: Color {
        get { Color(hex: transcriptColorHex) ?? Color(nsColor: .secondaryLabelColor) }
        set { transcriptColorHex = newValue.hexString ?? transcriptColorHex }
    }

    var translationColor: Color {
        get { Color(hex: translationColorHex) ?? Color(nsColor: .labelColor) }
        set { translationColorHex = newValue.hexString ?? translationColorHex }
    }

    enum LayoutMode: String, CaseIterable, Identifiable {
        case mixed      = "mixed"
        case sideBySide = "sideBySide"
        var id: String { rawValue }
        var label: String { self == .mixed ? "Mixed" : "Side by Side" }
    }
}

// MARK: - Color ↔ hex string

extension Color {
    /// Parse `#rrggbb` (6-digit hex, leading `#` required).
    init?(hex: String) {
        guard hex.hasPrefix("#") else { return nil }
        let h = String(hex.dropFirst())
        guard h.count == 6, let rgb = UInt64(h, radix: 16) else { return nil }
        self.init(
            red:   Double((rgb >> 16) & 0xFF) / 255,
            green: Double((rgb >>  8) & 0xFF) / 255,
            blue:  Double( rgb        & 0xFF) / 255
        )
    }

    /// Render as `#rrggbb`. Returns nil if color space conversion fails.
    var hexString: String? {
        guard let c = NSColor(self).usingColorSpace(.sRGB) else { return nil }
        return String(format: "#%02x%02x%02x",
                      Int(c.redComponent * 255),
                      Int(c.greenComponent * 255),
                      Int(c.blueComponent * 255))
    }
}
```

- [ ] **Step 2: Build**

```bash
LIVETRANSLATE_SIGN_IDENTITY=LiveTranslateDev ./build.sh 2>&1 | tail -20
```

- [ ] **Step 3: Commit**

```bash
git add Sources/LiveTranslate/AppSettings.swift
git commit -m "feat: AppSettings — persisted font, color, layout, opacity preferences"
```

---

### Task 5: Create `OptionsView`

**Files:**
- Create: `Sources/LiveTranslate/OptionsView.swift`

- [ ] **Step 1: Create `OptionsView.swift`**

```swift
import SwiftUI

struct OptionsView: View {
    @ObservedObject var settings: AppSettings
    @State private var transcriptColor: Color  = .secondary
    @State private var translationColor: Color = .primary

    var body: some View {
        Form {
            Section("Typography") {
                LabeledContent("Transcript size") {
                    HStack {
                        Slider(value: $settings.transcriptFontSize, in: 10...28, step: 1)
                            .frame(width: 160)
                        Text("\(Int(settings.transcriptFontSize))pt")
                            .frame(width: 32, alignment: .trailing).monospacedDigit()
                    }
                }
                LabeledContent("Translation size") {
                    HStack {
                        Slider(value: $settings.translationFontSize, in: 10...32, step: 1)
                            .frame(width: 160)
                        Text("\(Int(settings.translationFontSize))pt")
                            .frame(width: 32, alignment: .trailing).monospacedDigit()
                    }
                }
            }
            Section("Colors") {
                ColorPicker("Transcript text",  selection: $transcriptColor,  supportsOpacity: false)
                    .onChange(of: transcriptColor)  { _, c in settings.transcriptColor  = c }
                ColorPicker("Translation text", selection: $translationColor, supportsOpacity: false)
                    .onChange(of: translationColor) { _, c in settings.translationColor = c }
            }
            Section("Layout") {
                Picker("Sentence layout", selection: $settings.layoutModeRaw) {
                    ForEach(AppSettings.LayoutMode.allCases) { m in
                        Text(m.label).tag(m.rawValue)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                Group {
                    if settings.layoutMode == .mixed { mixedPreview }
                    else { sideBySidePreview }
                }
            }
            Section("Window") {
                LabeledContent("Background opacity") {
                    HStack {
                        Slider(value: $settings.windowOpacity, in: 0.2...1.0, step: 0.05)
                            .frame(width: 160)
                        Text("\(Int(settings.windowOpacity * 100))%")
                            .frame(width: 36, alignment: .trailing).monospacedDigit()
                    }
                }
            }
        }
        .formStyle(.grouped)
        .frame(width: 380)
        .padding()
        .onAppear {
            transcriptColor  = settings.transcriptColor
            translationColor = settings.translationColor
        }
    }

    private var mixedPreview: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Translation appears here")
                .font(.system(size: settings.translationFontSize))
                .foregroundStyle(settings.translationColor)
            Text("Transcription sits below as a caption")
                .font(.system(size: settings.transcriptFontSize))
                .foregroundStyle(settings.transcriptColor)
        }
        .padding(.top, 4)
    }

    private var sideBySidePreview: some View {
        HStack(alignment: .top, spacing: 12) {
            Text("Transcription on the left")
                .font(.system(size: settings.transcriptFontSize))
                .foregroundStyle(settings.transcriptColor)
                .frame(maxWidth: .infinity, alignment: .leading)
            Divider()
            Text("Translation on the right")
                .font(.system(size: settings.translationFontSize))
                .foregroundStyle(settings.translationColor)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.top, 4)
    }
}
```

- [ ] **Step 2: Build**

```bash
LIVETRANSLATE_SIGN_IDENTITY=LiveTranslateDev ./build.sh 2>&1 | tail -20
```

- [ ] **Step 3: Commit**

```bash
git add Sources/LiveTranslate/OptionsView.swift
git commit -m "feat: OptionsView — font, color, layout, opacity settings form"
```

---

### Task 6: Wire `AppSettings` into the app and row views

**Files:**
- Modify: `Sources/LiveTranslate/App.swift`
- Modify: `Sources/LiveTranslate/TranscriptView.swift`
- Modify: `Sources/LiveTranslate/MenuBarView.swift`

The ONNX repo uses a merged `displayRows` ForEach (sentences + inflightChunks rendered by a single `TranscriptRow` view). Read `TranscriptView.swift` in full before editing to understand exactly how `TranscriptRow` is structured, then apply the settings reads there.

- [ ] **Step 1: Add `AppSettings` + `Settings` scene to `App.swift`**

Add `@StateObject private var settings = AppSettings()` to `LiveTranslateApp`.

Pass `.environmentObject(settings)` on the `TranscriptView` and on `MenuBarView`. Add a `Settings` scene. Add `onChange(of: settings.windowOpacity)` to apply opacity to the `NSWindow`.

```swift
@main
struct LiveTranslateApp: App {
    @StateObject private var pipeline = Pipeline()
    @StateObject private var settings = AppSettings()
    @State private var mainWindow: NSWindow?
    @State private var isWindowVisible = true

    init() {
        Log.startup()
        Task.detached(priority: .background) {
            await CrashRecovery.recoverPendingSessions()
        }
    }

    var body: some Scene {
        Window("LiveTranslate", id: "main") {
            TranscriptView(pipeline: pipeline)
                .frame(minWidth: 260, minHeight: 80)
                .environmentObject(settings)
                .background(WindowAccessor { window in
                    mainWindow = window
                    configure(window)
                })
                .onAppear { installTerminateHook(pipeline: pipeline) }
                .onChange(of: settings.windowOpacity) { _, opacity in
                    mainWindow?.alphaValue = CGFloat(opacity)
                }
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentMinSize)
        .defaultSize(width: 520, height: 480)
        .commands {
            CommandMenu("Debug") {
                Button("Load fixture sentences") { pipeline.loadDebugFixtures() }
                    .keyboardShortcut("d", modifiers: [.command, .shift])
                Button("Clear sentences") { pipeline.clear() }
                    .keyboardShortcut("k", modifiers: [.command, .shift])
            }
        }

        Settings {
            OptionsView(settings: settings)
        }

        MenuBarExtra {
            MenuBarView(
                pipeline: pipeline,
                settings: settings,
                isWindowVisible: $isWindowVisible,
                mainWindow: mainWindow
            )
            .environmentObject(settings)
        } label: {
            Image(systemName: pipeline.isRunning ? "waveform.circle.fill" : "waveform.circle")
        }
        .menuBarExtraStyle(.window)
    }

    // Keep the existing `installTerminateHook` and `configure` methods unchanged.
}
```

> **Note:** The `MenuBarExtra` label image name may differ — keep whatever the current repo uses.

- [ ] **Step 2: Add `settings` parameter + gear button to `MenuBarView`**

Read `MenuBarView.swift` first. Then:

Add `@ObservedObject var settings: AppSettings` and `@Environment(\.openSettings) private var openSettings` to `MenuBarView`.

Update the initializer callers in `App.swift` (done above) and `MenuBarView.swift` struct definition.

Add a settings gear button to the compact bar (alongside the existing action buttons):

```swift
private var settingsButton: some View {
    Button { openSettings() } label: {
        Image(systemName: "gearshape")
            .font(.system(size: 12, weight: .medium))
            .foregroundStyle(.secondary)
    }
    .buttonStyle(.plain)
    .help("Open settings (⌘,)")
}
```

Add `.environmentObject(settings)` to the root `VStack` in `body` so child row views receive it.

- [ ] **Step 3: Apply settings in `TranscriptView` and `TranscriptRow`**

Read `TranscriptView.swift` in full first. Then:

Add `@EnvironmentObject var settings: AppSettings` to `TranscriptView`.

Change the hardcoded `.opacity(0.7)` background to `settings.windowOpacity`:
```swift
Color(nsColor: .textBackgroundColor)
    .opacity(settings.windowOpacity)
    .ignoresSafeArea()
```

In `TranscriptRow` (or whatever the row view is named), add `@EnvironmentObject var settings: AppSettings` and replace hardcoded `.font(.body)` / `.font(.caption)` / `.foregroundStyle(.primary)` / `.foregroundStyle(.secondary)` with:
- Translation line: `.font(.system(size: settings.translationFontSize))` + `.foregroundStyle(settings.translationColor)`
- Transcription/caption line: `.font(.system(size: settings.transcriptFontSize))` + `.foregroundStyle(settings.transcriptColor)`

For in-flight rows (`.partial` / `.translating` states), keep the italic style but apply settings sizes at reduced opacity:
```swift
.font(.system(size: settings.translationFontSize))
.foregroundStyle(settings.translationColor.opacity(0.5))
.italic()
```

- [ ] **Step 4: Add side-by-side layout**

In `TranscriptView.swift`, add a `SideBySideSentenceRow` view (after `StreamShareView`):

```swift
struct SideBySideSentenceRow: View {
    let sentence: Sentence
    @EnvironmentObject var settings: AppSettings

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Text(sentence.text)
                .font(.system(size: settings.transcriptFontSize))
                .foregroundStyle(settings.transcriptColor)
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
            Divider()
            Text(sentence.translation.isEmpty ? sentence.text : sentence.translation)
                .font(.system(size: settings.translationFontSize))
                .foregroundStyle(settings.translationColor)
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
        }
        .padding(.vertical, 4)
    }
}
```

In the sentence list (wherever `displayRows` / `ForEach` is rendered), branch on `settings.layoutMode`:

```swift
// For completed sentences:
if settings.layoutMode == .sideBySide, let s = row.asSentence {
    SideBySideSentenceRow(sentence: s)
        // ... existing id / transition
} else {
    TranscriptRow(row: row, compact: compact)
        // ... existing id / transition
}
```

> `row.asSentence` or however the ONNX repo distinguishes completed sentences from inflight chunks in `displayRows` — read the actual `TranscriptView.swift` to get the exact type/enum.

- [ ] **Step 5: Build and verify**

```bash
LIVETRANSLATE_SIGN_IDENTITY=LiveTranslateDev ./build.sh 2>&1 | tail -20
open build/LiveTranslate.app
```

Verify:
- Cmd+, opens the OptionsView settings window
- Gear icon in menu bar popover opens the same window
- Font size sliders update row text live
- Color pickers update text colors live
- Opacity slider changes main window background translucency
- "Side by Side" layout shows transcription and translation in two columns
- Settings persist after quitting and relaunching (check UserDefaults via `defaults read`)

- [ ] **Step 6: Commit**

```bash
git add Sources/LiveTranslate/App.swift Sources/LiveTranslate/TranscriptView.swift Sources/LiveTranslate/MenuBarView.swift
git commit -m "feat: wire AppSettings into app — font, color, layout, opacity live-configurable"
```

---

## Feature 3: OBS Overlay — English Detection + German Translation

### Task 7: Add OBS translator and English detection to `Pipeline`

**Files:**
- Modify: `Sources/LiveTranslate/Pipeline.swift`

`NLLanguageRecognizer.dominantLanguage(for:)` is synchronous and returns in < 1 ms. It correctly identifies English vs. German at sentence length with > 99% accuracy. It handles brand names and mixed-language sentences by returning the dominant language. Import is `NaturalLanguage` (ships with macOS, no package dependency needed).

The OBS translator needs its own `AppleTranslator` instance and its own `TranslationSession` (configured en→de). The session can only come from SwiftUI's `.translationTask` modifier, so we add a second one in `TranscriptView`.

- [ ] **Step 1: Add `import NaturalLanguage` and `obsTranslator` to `Pipeline.swift`**

At the top of `Pipeline.swift`, add:
```swift
import NaturalLanguage
```

Alongside `private let translator: Translator`, add:
```swift
/// Second translator, always en→de, for the OBS overlay. Session installed
/// by `TranscriptView`'s second `.translationTask`.
private let obsTranslator = AppleTranslator()
```

Add alongside `installTranslationSession`:
```swift
func installOBSTranslationSession(_ session: TranslationSession?) {
    obsTranslator.setSession(session)
}
```

- [ ] **Step 2: Detect English and publish OBS subtitle in `graduate()`**

In `graduate(id:source:text:translation:createdAt:endsAt:)`, after `enforceMaxCount()`:

```swift
// OBS overlay: detect if the transcription is English and, if so,
// translate it to German and push to any OBS browser sources connected.
// NLLanguageRecognizer is synchronous, < 1 ms, fully on-device.
if let server = liveAudioServer {
    let recognizer = NLLanguageRecognizer()
    recognizer.processString(text)
    if recognizer.dominantLanguage == NLLanguage.english {
        Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                let german = try await self.obsTranslator.translate(text)
                server.publishOBSSubtitle(germanText: german)
                Log.line("OBS: published German subtitle \"\(german.prefix(40))\"")
            } catch {
                Log.line("OBS translator error: \(error.localizedDescription)")
            }
        }
    }
}
```

- [ ] **Step 3: Add en→de `.translationTask` to `TranscriptView`**

In `TranscriptView.swift`, the existing `.translationTask` parks the main translator session. Add a second modifier on the same root view (they stack independently):

```swift
.translationTask(
    TranslationSession.Configuration(
        source: Locale.Language(identifier: "en"),
        target: Locale.Language(identifier: "de")
    )
) { session in
    pipeline.installOBSTranslationSession(session)
    defer { pipeline.installOBSTranslationSession(nil) }
    do {
        try await session.prepareTranslation()
        Log.line("OBS en→de translation prepared")
    } catch {
        Log.line("OBS prepareTranslation failed: \(error.localizedDescription)")
    }
    let (parked, holder) = AsyncStream<Never>.makeStream()
    defer { holder.finish() }
    for await _ in parked { }
}
```

- [ ] **Step 4: Build**

```bash
LIVETRANSLATE_SIGN_IDENTITY=LiveTranslateDev ./build.sh 2>&1 | tail -20
```

- [ ] **Step 5: Commit**

```bash
git add Sources/LiveTranslate/Pipeline.swift Sources/LiveTranslate/TranscriptView.swift
git commit -m "feat: OBS overlay — NLLanguageRecognizer English detection, en→de translation on graduate"
```

---

### Task 8: Surface OBS URL in share popover; end-to-end test

**Files:**
- Modify: `Sources/LiveTranslate/TranscriptView.swift` (`StreamShareView`)
- Modify: `Sources/LiveTranslate/MenuBarView.swift`

- [ ] **Step 1: Add OBS URL section to `StreamShareView`**

In `TranscriptView.swift`, update `StreamShareView` to accept an optional `obsURL`:

```swift
struct StreamShareView: View {
    let url: String
    var obsURL: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Live translated audio")
                .font(.headline)
            Text("Open on a phone with headphones to hear translations in near-real time.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            urlRow(url)
            if let img = qrImage(for: url) {
                Image(nsImage: img).interpolation(.none).resizable().scaledToFit()
                    .frame(width: 200, height: 200).frame(maxWidth: .infinity).padding(.top, 2)
            }
            if let obsURL {
                Divider()
                Text("OBS Browser Source")
                    .font(.headline)
                Text("Add as Browser Source in OBS. Set dimensions to your overlay size (e.g. 1920×1080).")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                urlRow(obsURL)
            }
        }
    }

    @ViewBuilder
    private func urlRow(_ u: String) -> some View {
        HStack(spacing: 6) {
            Text(u).font(.system(.caption, design: .monospaced))
                .textSelection(.enabled).lineLimit(1).truncationMode(.middle)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button {
                let pb = NSPasteboard.general; pb.clearContents()
                pb.setString(u, forType: .string)
            } label: { Image(systemName: "doc.on.doc").font(.system(size: 11)) }
            .buttonStyle(.borderless).help("Copy URL")
        }
    }

    private func qrImage(for string: String) -> NSImage? {
        let filter = CIFilter.qrCodeGenerator()
        filter.message = Data(string.utf8); filter.correctionLevel = "M"
        guard let out = filter.outputImage?.transformed(by: CGAffineTransform(scaleX: 8, y: 8)) else { return nil }
        guard let cg = CIContext().createCGImage(out, from: out.extent) else { return nil }
        return NSImage(cgImage: cg, size: NSSize(width: out.extent.width, height: out.extent.height))
    }
}
```

Update the `streamShareButton` popover call site in `TranscriptView` to pass `obsURL: pipeline.liveOBSURL`.

Update the corresponding call site in `MenuBarView` the same way.

Also make the share icon visible whenever either URL is set (OBS URL is always set when server runs):

```swift
// In TranscriptView.streamShareButton:
if pipeline.liveStreamURL != nil || pipeline.liveOBSURL != nil {
    let url = pipeline.liveStreamURL ?? pipeline.liveOBSURL ?? ""
    // ... existing button body, update popover to pass obsURL
}
```

Apply the same change in `MenuBarView`.

- [ ] **Step 2: Build and end-to-end test**

```bash
LIVETRANSLATE_SIGN_IDENTITY=LiveTranslateDev ./build.sh 2>&1 | tail -20
open build/LiveTranslate.app
```

Verify:
- Share icon appears even without a TTS voice installed
- Popover shows both "Live translated audio" and "OBS Browser Source" sections
- Navigate to `http://<host>:8765/obs` in OBS or a browser — transparent page with no content initially
- Speak English sentences → after ~1–2 s, German subtitle appears on the OBS page and fades after 6 s
- Speak German sentences → no OBS subtitle (NLLanguageRecognizer returns `.german`, not `.english`)
- Tail log: `grep "OBS" /tmp/livetranslate.log` confirms subtitle publishes

- [ ] **Step 3: Commit**

```bash
git add Sources/LiveTranslate/TranscriptView.swift Sources/LiveTranslate/MenuBarView.swift
git commit -m "feat: OBS URL in share popover; share icon visible whenever server is running"
```

---

### Task 9: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update files table and design-decisions section**

Add to the files table:

| `AppSettings.swift` | Persisted display preferences (`@AppStorage`). Font sizes, hex-string colors, layout mode, window opacity. `Color(hex:)` extension. |
| `OptionsView.swift` | SwiftUI `Settings` form: font size sliders, `ColorPicker`s, segmented layout picker, opacity slider. Opened via Cmd+, or the gear icon in `MenuBarView`. |

Update `LiveAudioServer.swift` row to mention: hypothesis SSE events (`event: hypothesis`, `event: hypothesis-done`), OBS routes (`/obs`, `/obs-events`), always started unconditionally.

Update `Pipeline.swift` row to mention: `liveOBSURL`, `obsTranslator`, `NLLanguageRecognizer` English detection, always-start server.

Add to "Key design decisions":

- **`NLLanguageRecognizer` for OBS English detection.** Synchronous, on-device, < 1 ms per call. In `graduate()`, `NLLanguageRecognizer.dominantLanguage(for: text)` returns `NLLanguage.english` for English sentences. If English, the `obsTranslator` (en→de `AppleTranslator`) translates and `liveAudioServer.publishOBSSubtitle` pushes to `/obs-events`. TTS is never involved — OBS and TTS are independent paths.
- **OBS server always on, TTS still listener-gated.** `LiveAudioServer` now starts every run. `liveOBSURL` is always published. `liveStreamURL` (audio WAV) is only published when a TTS voice is available. The `ttsSpeaker.enqueue()` call remains gated on `server.audioListenerCount > 0` — synthesis never fires for OBS-only sessions.
- **`AppSettings` via SwiftUI environment object.** All display preferences propagate from `App.swift` via `.environmentObject(settings)`. Row views read font size and color from it; `TranscriptView` reads layout mode and opacity. Persisted via `@AppStorage`.
- **Hypothesis SSE carries partial translation.** Because `SherpaTranscriber` produces `.partial(text:, translation:)` with a rolling translated preview, the `hypothesis` SSE event includes both `text` and `translation` fields. The mobile listen page shows the partial translation as the primary line (bold), falling back to the transcription if translation isn't ready yet.

Add to "Things that have bitten us":

> **N. Named SSE events need `es.addEventListener('name', ...)`, not `onmessage`.** The default `es.onmessage` only fires for events without an `event:` field. Hypothesis and hypothesis-done events use named types; hooking them to `onmessage` silently drops them.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md — AppSettings, OBS overlay, hypothesis SSE, NLLanguageRecognizer"
```

---

## Post-implementation: open a PR

Once all tasks are committed to the worktree branch:

```bash
cd ~/Code/worktrees/live-translate-de-en-onnx-feat-live-updates-options-obs
git push -u origin feat/live-updates-options-obs
gh pr create \
  --title "feat: live hypothesis web updates, options window, OBS overlay" \
  --body "$(cat <<'EOF'
## Summary
- Mobile listen page now shows rolling partial hypothesis (including partial translation) updated in real time via named SSE events; stronger typography; auto-scroll
- Options window (Cmd+, or gear icon in menu bar) controls font sizes, text colors, mixed/side-by-side layout, and window opacity
- OBS browser-source overlay at /obs: German subtitles appear when English speech is detected (NLLanguageRecognizer, on-device, < 1 ms); fades after 6 s

## Test plan
- [ ] Open http://<host>:8765/ on a phone; speak; verify partial translation appears in < 1 s and finalizes smoothly
- [ ] Cmd+, opens options window; sliders and pickers update the UI live; settings persist after relaunch
- [ ] Side-by-side layout shows transcription left, translation right
- [ ] Open http://<host>:8765/obs in OBS Browser Source; speak English; German subtitle appears and fades
- [ ] Speak German; no OBS subtitle
- [ ] Share icon visible even without TTS voice; popover shows OBS URL section

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

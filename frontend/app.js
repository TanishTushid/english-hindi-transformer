/* ─────────────────────────────────────────
   bhashantara – Frontend Logic
   ───────────────────────────────────────── */

const API_BASE = window.location.origin;
let selectedModel = 'nllb';
let isTranslating = false;
let speechRecognition = null;

// ─── Initialise ───────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  await fetchModelStatus();
  document.getElementById('input-text').addEventListener('keydown', e => {
    if (e.ctrlKey && e.key === 'Enter') translateText();
  });
});

// ─── Model Selection ──────────────────────
function selectModel(model) {
  selectedModel = model;
  document.getElementById('btn-nllb').classList.toggle('active', model === 'nllb');
  document.getElementById('btn-scratch').classList.toggle('active', model === 'scratch');
  document.getElementById('btn-nllb').setAttribute('aria-pressed', model === 'nllb');
  document.getElementById('btn-scratch').setAttribute('aria-pressed', model === 'scratch');
}

// ─── Model Status ─────────────────────────
async function fetchModelStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/models`);
    if (!res.ok) throw new Error('API not reachable');
    const data = await res.json();
    updateStatusDot('nllb-status', data.nllb?.available);
    updateStatusDot('scratch-status', data.scratch?.available);
    setStatus('Backend connected. Ready to translate.', 'success');
  } catch (e) {
    setStatus('⚠ Backend not reachable. Start the FastAPI server first: uvicorn main:app --reload', 'error');
    updateStatusDot('nllb-status', false);
    updateStatusDot('scratch-status', false);
  }
}

function updateStatusDot(id, available) {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = 'status-dot ' + (available ? 'available' : 'unavailable');
  el.title = available ? 'Available' : 'Not available';
}

// ─── Translation ──────────────────────────
async function translateText() {
  const text = document.getElementById('input-text').value.trim();
  if (!text) { showToast('Please enter some text first.'); return; }
  if (isTranslating) return;

  isTranslating = true;
  setTranslating(true);
  clearOutput();

  try {
    const res = await fetch(`${API_BASE}/api/translate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, model: selectedModel })
    });

    const data = await res.json();

    if (!res.ok) {
      const msg = data.detail || 'Translation failed.';
      showOutputError(msg);
      setStatus('Translation failed: ' + msg, 'error');
      return;
    }

    showTranslation(data.translation);
    document.getElementById('meta-info').textContent =
      `⚡ ${data.model_used.toUpperCase()} · ${data.time_ms.toFixed(0)} ms`;
    setStatus('', '');

  } catch (e) {
    showOutputError('Network error. Is the backend running?');
    setStatus('Network error: ' + e.message, 'error');
  } finally {
    isTranslating = false;
    setTranslating(false);
  }
}

function setTranslating(active) {
  const btn = document.getElementById('translate-btn');
  const spinner = document.getElementById('spinner');
  const output = document.getElementById('output-area');
  btn.disabled = active;
  spinner.classList.toggle('hidden', !active);
  if (active) output.classList.add('loading');
  else output.classList.remove('loading');
}

function showTranslation(text) {
  const area = document.getElementById('output-area');
  area.innerHTML = '';
  // Animate text in word-by-word
  const words = text.split(' ');
  let delay = 0;
  words.forEach((word, i) => {
    const span = document.createElement('span');
    span.textContent = (i > 0 ? ' ' : '') + word;
    span.style.opacity = '0';
    span.style.transition = `opacity 0.2s ease ${delay}ms`;
    area.appendChild(span);
    delay += 35;
  });
  requestAnimationFrame(() => {
    area.querySelectorAll('span').forEach(s => { s.style.opacity = '1'; });
  });
}

function showOutputError(msg) {
  const area = document.getElementById('output-area');
  area.innerHTML = `<span style="color: var(--error)">⚠ ${msg}</span>`;
}

function clearOutput() {
  const area = document.getElementById('output-area');
  area.innerHTML = '<div class="placeholder-msg">Translating…</div>';
  document.getElementById('meta-info').textContent = '';
}

// ─── Input Helpers ────────────────────────
function onInputChange() {
  const val = document.getElementById('input-text').value;
  document.getElementById('char-count').textContent = `${val.length} / 1000`;
}

function clearInput() {
  document.getElementById('input-text').value = '';
  document.getElementById('char-count').textContent = '0 / 1000';
  document.getElementById('output-area').innerHTML = '<div class="placeholder-msg">Translation will appear here…</div>';
  document.getElementById('meta-info').textContent = '';
}

function setExample(text) {
  document.getElementById('input-text').value = text;
  onInputChange();
  translateText();
}

async function pasteFromClipboard() {
  try {
    const text = await navigator.clipboard.readText();
    document.getElementById('input-text').value = text;
    onInputChange();
    showToast('Pasted from clipboard');
  } catch {
    showToast('Could not access clipboard');
  }
}

// ─── Copy Output ──────────────────────────
async function copyOutput() {
  const area = document.getElementById('output-area');
  const text = area.innerText.trim();
  if (!text || text.includes('Translation will appear') || text.includes('Translating')) {
    showToast('Nothing to copy yet');
    return;
  }
  try {
    await navigator.clipboard.writeText(text);
    showToast('✓ Copied to clipboard');
  } catch {
    showToast('Could not copy');
  }
}

// ─── TTS ──────────────────────────────────
function speakOutput() {
  const area = document.getElementById('output-area');
  const text = area.innerText.trim();
  if (!text || text.includes('Translation will appear')) { showToast('Nothing to speak yet'); return; }
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'hi-IN';
  utterance.rate = 0.9;
  speechSynthesis.speak(utterance);
}

// ─── Speech Input ─────────────────────────
function toggleSpeech() {
  const btn = document.getElementById('btn-mic');
  if (speechRecognition && speechRecognition.running) {
    speechRecognition.stop();
    speechRecognition.running = false;
    btn.classList.remove('active');
    btn.title = 'Speak to translate';
    return;
  }

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) { showToast('Speech recognition not supported in this browser'); return; }

  speechRecognition = new SpeechRecognition();
  speechRecognition.lang = 'en-US';
  speechRecognition.continuous = false;
  speechRecognition.interimResults = false;

  speechRecognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    document.getElementById('input-text').value = transcript;
    onInputChange();
    translateText();
  };

  speechRecognition.onend = () => {
    speechRecognition.running = false;
    btn.classList.remove('active');
    btn.title = 'Speak to translate';
  };

  speechRecognition.onerror = (e) => {
    showToast('Speech error: ' + e.error);
    btn.classList.remove('active');
  };

  speechRecognition.start();
  speechRecognition.running = true;
  btn.classList.add('active');
  btn.title = 'Stop recording';
  showToast('🎤 Listening…');
}

// ─── Status Bar ───────────────────────────
function setStatus(msg, type) {
  const bar = document.getElementById('status-bar');
  bar.textContent = msg;
  bar.className = 'status-bar' + (type ? ' ' + type : '');
}

// ─── Toast ────────────────────────────────
let toastTimer;
function showToast(msg) {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 2400);
}

const chat = document.getElementById('chat');
const form = document.getElementById('chatForm');
const question = document.getElementById('question');
const trace = document.getElementById('trace');
const sourceUsed = document.getElementById('sourceUsed');
const welcome = document.getElementById('welcome');
const tracePanel = document.getElementById('tracePanel');
const steps = [...document.querySelectorAll('.flow-step')];

const STEP_KEYWORDS = {
  route: ['route', 'routing', 'classif'],
  retrieve: ['retriev', 'private', 'kb', 'knowledge', 'pinecone', 'vector'],
  grade: ['grade', 'grading', 'evidence', 'relevan'],
  web: ['web', 'tavily', 'search', 'external'],
  rewrite: ['rewrite', 'rewrit', 'retry', 'rephras'],
  answer: ['answer', 'generat', 'grounded', 'final']
};

function escapeHtml(s = '') {
  return String(s).replace(/[&<>'"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[c]));
}

// Safe lightweight markdown: escape first, then format.
function formatText(s = '') {
  const lines = escapeHtml(s).split('\n');
  let html = '', list = null;
  const inline = t => t
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };

  for (const raw of lines) {
    const line = raw.trim();
    const ul = line.match(/^[-*•]\s+(.*)/);
    const ol = line.match(/^\d+[.)]\s+(.*)/);
    if (ul || ol) {
      const tag = ul ? 'ul' : 'ol';
      if (list !== tag) { closeList(); html += `<${tag}>`; list = tag; }
      html += `<li>${inline((ul || ol)[1])}</li>`;
    } else if (line === '') {
      closeList();
    } else {
      closeList();
      html += `<p>${inline(line)}</p>`;
    }
  }
  closeList();
  return html;
}

function scrollToBottom() { chat.scrollTop = chat.scrollHeight; }

function addMessage(role, text, source = '', citations = []) {
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;
  const isWeb = /web/i.test(source);
  const sourceHtml = source ? `<div class="answer-source ${isWeb ? 'web' : ''}">Source: ${escapeHtml(source)}</div>` : '';
  const citeHtml = citations.length
    ? `<div class="citations"><strong>Sources</strong><br>${citations.map(c => c.url
        ? `<a href="${escapeHtml(c.url)}" target="_blank" rel="noopener">${escapeHtml(c.title)}</a>`
        : escapeHtml(c.title)).join('<br>')}</div>`
    : '';
  const body = role === 'user' ? escapeHtml(text).replace(/\n/g, '<br>') : formatText(text);
  wrap.innerHTML = `<div class="avatar">AI</div><div class="bubble">${body}${sourceHtml}${citeHtml}</div>`;
  chat.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function addError(message) {
  const wrap = document.createElement('div');
  wrap.className = 'message assistant';
  wrap.innerHTML = `<div class="avatar">AI</div><div class="bubble error"><strong>Request failed</strong><p>${escapeHtml(message)}</p></div>`;
  chat.appendChild(wrap);
  scrollToBottom();
}

function addTyping() {
  const wrap = document.createElement('div');
  wrap.className = 'message assistant';
  wrap.innerHTML = '<div class="avatar">AI</div><div class="bubble"><div class="typing"><i></i><i></i><i></i></div></div>';
  chat.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function resetPipeline() { steps.forEach(s => s.classList.remove('done', 'active')); }

function updatePipeline(items = []) {
  resetPipeline();
  const text = items.join(' ').toLowerCase();
  steps.forEach(s => {
    const words = STEP_KEYWORDS[s.dataset.key] || [];
    if (words.some(w => text.includes(w))) s.classList.add('done');
  });
}

function renderTrace(items = []) {
  trace.innerHTML = items.length
    ? items.map((x, i) => `<div class="trace-item" style="--i:${i}">${escapeHtml(x)}</div>`).join('')
    : '<div class="empty">No trace.</div>';
  updatePipeline(items);
}

async function askAgent(q) {
  if (welcome) welcome.remove();
  addMessage('user', q);
  question.value = '';
  autoGrow();
  resetPipeline();
  steps[0] && steps[0].classList.add('active');
  renderTrace(['Running the agent…']);
  sourceUsed.textContent = 'Working…';
  const btn = form.querySelector('button[type="submit"]');
  btn.disabled = true;
  const typing = addTyping();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q })
    });
    const raw = await res.text();
    let data = {};
    try { data = raw ? JSON.parse(raw) : {}; } catch (_) { /* non-JSON response */ }
    if (!raw) throw new Error('The server sent an empty response (HTTP ' + res.status + '). Check the Flask terminal for the error.');
    if (!res.ok) throw new Error(data.detail || ('Server error (HTTP ' + res.status + ').'));
    typing.remove();
    addMessage('assistant', data.answer, data.source_used, data.citations || []);
    renderTrace(data.trace || []);
    sourceUsed.textContent = data.source_used || '—';
  } catch (e) {
    typing.remove();
    addError(e.message);
    resetPipeline();
    renderTrace(['Request failed']);
    sourceUsed.textContent = 'Error';
  } finally {
    btn.disabled = false;
    question.focus();
  }
}

function autoGrow() {
  question.style.height = 'auto';
  question.style.height = Math.min(question.scrollHeight, 160) + 'px';
}

question.addEventListener('input', autoGrow);
question.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});
form.addEventListener('submit', e => {
  e.preventDefault();
  const q = question.value.trim();
  if (q) askAgent(q);
});
document.querySelectorAll('.example').forEach(b =>
  b.addEventListener('click', () => askAgent(b.textContent.trim()))
);

// Trace drawer (small screens)
document.getElementById('toggleTrace').onclick = () => tracePanel.classList.add('open');
document.getElementById('closeTrace').onclick = () => tracePanel.classList.remove('open');

// Upload modal
const modal = document.getElementById('uploadModal');
document.getElementById('openUpload').onclick = () => modal.classList.remove('hidden');
document.getElementById('closeUpload').onclick = () => modal.classList.add('hidden');
modal.addEventListener('click', e => { if (e.target === modal) modal.classList.add('hidden'); });
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') { modal.classList.add('hidden'); tracePanel.classList.remove('open'); }
});

document.getElementById('uploadBtn').onclick = async () => {
  const file = document.getElementById('fileInput').files[0];
  const key = document.getElementById('adminKey').value;
  const status = document.getElementById('uploadStatus');
  if (!file) { status.textContent = 'Choose a file first.'; return; }
  status.textContent = 'Indexing document…';
  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch('/api/ingest', { method: 'POST', headers: { 'X-Admin-Key': key }, body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Upload failed');
    status.textContent = `Indexed ${d.file}: ${d.chunks} chunks.`;
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
  }
};
// ────────────────────────────────────────────────────────────
//  Needle AI — Frontend Application Logic
// ────────────────────────────────────────────────────────────

const API_BASE = window.location.origin;

// ── State ───────────────────────────────────────────────────
const state = {
  documents: [],
  activeDocumentId: null,
  isStreaming: false,
  messages: [],
};

// ── DOM References ──────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dom = {
  // Sidebar
  sidebar: $(".sidebar"),
  sidebarOverlay: $(".sidebar-overlay"),
  sidebarToggle: $(".sidebar-toggle"),
  newDocBtn: $("#new-doc-btn"),
  attachBtn: $("#attach-btn"),
  dropOverlay: $("#drop-overlay"),
  uploadInput: $("#file-upload"),
  uploadProgress: $(".upload-progress"),
  progressFill: $(".progress-fill"),
  progressText: $(".progress-text"),
  docList: $(".doc-list"),
  docEmpty: $(".doc-empty"),

  // Chat
  chatMessages: $(".chat-messages"),
  welcomeScreen: $(".welcome-screen"),
  chatInput: $(".chat-input"),
  sendBtn: $(".send-btn"),
  activeDocBadge: $(".active-doc-badge"),
  activeDocName: $(".active-doc-name"),
  inputHint: $(".input-hint"),

  // Toast
  toastContainer: $(".toast-container"),
};

// ── Initialization ──────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initUpload();
  initChat();
  initSidebar();
  loadDocuments();
});

// ── Upload Logic ────────────────────────────────────────────
function initUpload() {
  const input = dom.uploadInput;

  if(dom.newDocBtn) dom.newDocBtn.addEventListener("click", () => input.click());
  if(dom.attachBtn) dom.attachBtn.addEventListener("click", () => input.click());

  input.addEventListener("change", (e) => {
    if (e.target.files.length > 0) uploadFile(e.target.files[0]);
  });

  // Full-Screen Drag and Drop
  let dragCounter = 0;
  
  window.addEventListener("dragenter", (e) => {
    e.preventDefault();
    dragCounter++;
    if(dom.dropOverlay) dom.dropOverlay.classList.remove("hidden");
  });

  window.addEventListener("dragover", (e) => {
    e.preventDefault();
  });

  window.addEventListener("dragleave", () => {
    dragCounter--;
    if (dragCounter === 0) {
      if(dom.dropOverlay) dom.dropOverlay.classList.add("hidden");
    }
  });

  window.addEventListener("drop", (e) => {
    e.preventDefault();
    dragCounter = 0;
    if(dom.dropOverlay) dom.dropOverlay.classList.add("hidden");
    if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
  });
}

async function uploadFile(file) {
  // Validate
  const allowedExtensions = [".pdf", ".txt", ".md", ".text", ".docx", ".pptx"];
  const ext = "." + file.name.split(".").pop().toLowerCase();
  if (!allowedExtensions.includes(ext)) {
    showToast("Please upload a valid document format.", "error");
    return;
  }

  if (file.size > 50 * 1024 * 1024) {
    showToast("File size must be under 50MB.", "error");
    return;
  }

  // Show progress
  dom.uploadProgress.classList.add("active");
  dom.progressFill.style.width = "0%";
  dom.progressText.textContent = "Uploading document...";

  // Animate progress (indeterminate feel)
  let progress = 0;
  const progressInterval = setInterval(() => {
    progress += Math.random() * 5;
    if (progress > 85) progress = 85;
    dom.progressFill.style.width = progress + "%";
  }, 500);

  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE}/api/upload`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Upload failed");
    }

    const data = await response.json();
    const taskId = data.task_id;
    
    dom.progressText.textContent = "Processing large file (this may take a few minutes)...";
    
    // Poll for status
    let isCompleted = false;
    let docData = null;
    
    while (!isCompleted) {
        await new Promise(r => setTimeout(r, 2000)); // Poll every 2 seconds
        
        const statusRes = await fetch(`${API_BASE}/api/upload/status/${taskId}`);
        if (!statusRes.ok) throw new Error("Failed to check upload status.");
        
        const statusData = await statusRes.json();
        
        if (statusData.status === "completed") {
            isCompleted = true;
            docData = statusData;
        } else if (statusData.status === "failed") {
            throw new Error(statusData.error || "Processing failed on server.");
        }
    }

    clearInterval(progressInterval);

    // Complete progress
    dom.progressFill.style.width = "100%";
    dom.progressText.textContent = "Done!";

    setTimeout(() => {
      dom.uploadProgress.classList.remove("active");
      dom.progressFill.style.width = "0%";
    }, 1500);

    showToast(docData.message, "success");

    // Reload documents and keep global selected
    await loadDocuments();
    selectGlobal();

  } catch (err) {
    clearInterval(progressInterval);
    dom.uploadProgress.classList.remove("active");
    showToast(`Upload failed: ${err.message}`, "error");
  }

  // Reset input
  dom.uploadInput.value = "";
}

// ── Document Management ─────────────────────────────────────
async function loadDocuments() {
  try {
    const response = await fetch(`${API_BASE}/api/documents`);
    const data = await response.json();
    state.documents = data.documents || [];
    renderDocumentList();
  } catch (err) {
    console.error("Failed to load documents:", err);
  }
}

function renderDocumentList() {
  dom.docList.innerHTML = "";

  if (state.documents.length === 0) {
    dom.docEmpty.classList.remove("hidden");
    dom.activeDocBadge.classList.add("hidden");
    return;
  }

  dom.docEmpty.classList.add("hidden");

  state.documents.forEach((doc) => {
    const li = document.createElement("li");
    li.className = `doc-item${doc.id === state.activeDocumentId ? " active" : ""}`;
    li.dataset.id = doc.id;

    let icon = "📝";
    const lowerName = doc.name.toLowerCase();
    if (lowerName.endsWith(".pdf")) icon = "📄";
    else if (lowerName.endsWith(".docx") || lowerName.endsWith(".doc")) icon = "📘";
    else if (lowerName.endsWith(".xlsx") || lowerName.endsWith(".csv")) icon = "📊";
    else if (lowerName.endsWith(".pptx")) icon = "📽️";
    
    const pages = doc.max_page || "?";
    const chunks = doc.chunk_count || "?";

    li.innerHTML = `
      <div class="doc-icon">${icon}</div>
      <div class="doc-meta">
        <div class="doc-name" title="${escapeHtml(doc.name)}">${escapeHtml(doc.name)}</div>
        <div class="doc-info">${pages} pages · ${chunks} chunks</div>
      </div>
      <button class="doc-delete" title="Delete document" onclick="event.stopPropagation(); deleteDoc('${doc.id}')">✕</button>
    `;

    li.addEventListener("click", () => selectDocument(doc.id));
    dom.docList.appendChild(li);
  });
}

function selectGlobal() {
  state.activeDocumentId = null;
  $$(".doc-item").forEach(el => el.classList.remove("active"));
  const globalBtn = $("#global-chat-btn");
  if(globalBtn) {
    globalBtn.classList.add("active");
  }
  
  dom.activeDocBadge.classList.remove("hidden");
  dom.activeDocName.textContent = "Global Database";
  dom.chatInput.placeholder = "Ask a question across all documents...";
  dom.chatInput.focus();
}

function selectDocument(docId) {
  state.activeDocumentId = docId;
  const doc = state.documents.find((d) => d.id === docId);

  // Update UI
  $$(".doc-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.id === docId);
  });
  
  const globalBtn = $("#global-chat-btn");
  if(globalBtn) {
    globalBtn.classList.remove("active");
  }

  if (doc) {
    dom.activeDocBadge.classList.remove("hidden");
    dom.activeDocName.textContent = doc.name;
    dom.chatInput.placeholder = `Ask about "${doc.name}"...`;
  }

  dom.chatInput.focus();
}

async function deleteDoc(docId) {
  try {
    const response = await fetch(`${API_BASE}/api/documents/${docId}`, {
      method: "DELETE",
    });

    if (!response.ok) throw new Error("Delete failed");

    showToast("Document deleted.", "info");

    if (state.activeDocumentId === docId) {
      state.activeDocumentId = null;
      dom.activeDocBadge.classList.add("hidden");
      dom.chatInput.placeholder = "Upload a document first...";
    }

    await loadDocuments();
  } catch (err) {
    showToast(`Delete failed: ${err.message}`, "error");
  }
}

// ── Chat Logic ──────────────────────────────────────────────
function initChat() {
  // Send on click
  dom.sendBtn.addEventListener("click", sendMessage);

  // Send on Enter (Shift+Enter for new line)
  dom.chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Auto-resize textarea
  dom.chatInput.addEventListener("input", () => {
    dom.chatInput.style.height = "auto";
    dom.chatInput.style.height = Math.min(dom.chatInput.scrollHeight, 120) + "px";
  });
}

async function sendMessage() {
  const query = dom.chatInput.value.trim();
  if (!query || state.isStreaming) return;

  if (state.documents.length === 0) {
    showToast("Please upload a document first.", "warning");
    return;
  }

  // Hide welcome screen
  if (dom.welcomeScreen) {
    dom.welcomeScreen.remove();
  }

  // Add user message
  appendMessage("user", query);
  dom.chatInput.value = "";
  dom.chatInput.style.height = "auto";

  // Add AI message placeholder
  const aiMessageEl = appendMessage("ai", "", true);
  const bubbleEl = aiMessageEl.querySelector(".message-bubble");

  state.isStreaming = true;
  dom.sendBtn.disabled = true;

  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        document_id: state.activeDocumentId,
      }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Chat request failed");
    }

    // Read SSE stream
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullText = "";
    let sources = null;
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Process complete SSE events
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6));

            if (data.type === "sources") {
              sources = data.data;
            } else if (data.type === "chunk") {
              fullText += data.content;
              bubbleEl.innerHTML = renderMarkdown(fullText);
              bubbleEl.classList.add("streaming-cursor");
              scrollToBottom();
            } else if (data.type === "done") {
              bubbleEl.classList.remove("streaming-cursor");
              if (sources && sources.length > 0) {
                appendSources(aiMessageEl, sources);
              }
            } else if (data.type === "error") {
              throw new Error(data.content);
            }
          } catch (parseErr) {
            // Skip malformed JSON
            if (parseErr.message !== "Unexpected end of JSON input") {
              console.warn("SSE parse error:", parseErr);
            }
          }
        }
      }
    }

    // Final render
    bubbleEl.classList.remove("streaming-cursor");
    if (fullText) {
      bubbleEl.innerHTML = renderMarkdown(fullText);
    }

  } catch (err) {
    bubbleEl.innerHTML = `<span style="color: var(--error);">⚠ Error: ${escapeHtml(err.message)}</span>`;
    showToast(`Chat error: ${err.message}`, "error");
  }

  state.isStreaming = false;
  dom.sendBtn.disabled = false;
  dom.chatInput.focus();
  scrollToBottom();
}

// ── Message Rendering ───────────────────────────────────────
function appendMessage(role, content, isStreaming = false) {
  const messageEl = document.createElement("div");
  messageEl.className = `message ${role}`;

  const avatarIcon = role === "ai" ? "🧠" : "👤";
  const bubbleContent = isStreaming
    ? '<div class="typing-indicator"><span></span><span></span><span></span></div>'
    : renderMarkdown(content);

  messageEl.innerHTML = `
    <div class="message-avatar">${avatarIcon}</div>
    <div class="message-content">
      <div class="message-bubble">${bubbleContent}</div>
    </div>
  `;

  dom.chatMessages.appendChild(messageEl);
  scrollToBottom();

  state.messages.push({ role, content });
  return messageEl;
}

function appendSources(messageEl, sources) {
  const contentEl = messageEl.querySelector(".message-content");
  const sourcesEl = document.createElement("div");
  sourcesEl.className = "sources-container";

  const chips = sources
    .map(
      (s, i) => `
    <button class="source-chip" onclick="toggleSourceDetail(this, ${i})" data-source-index="${i}">
      📄 ${s.header_context ? `[${s.header_context}]` : `Page ${s.page_number}`}
      <span class="chip-score">${Math.round(s.similarity_score)} RRF</span>
    </button>`
    )
    .join("");

  const details = sources
    .map(
      (s, i) => `
    <div class="source-detail" id="source-detail-${i}">
      <div class="source-detail-header">
        <span>📄 ${escapeHtml(s.document_name)} ${s.header_context ? `— Section: ${s.header_context}` : `— Page ${s.page_number}`}</span>
        <button class="source-detail-close" onclick="this.closest('.source-detail').classList.remove('active')">✕</button>
      </div>
      <div>${escapeHtml(s.text)}</div>
    </div>`
    )
    .join("");

  sourcesEl.innerHTML = `
    <div class="sources-label">📌 Sources retrieved via Hybrid Search</div>
    <div class="sources-chips">${chips}</div>
    ${details}
  `;

  contentEl.appendChild(sourcesEl);
  scrollToBottom();
}

// Toggle source detail visibility
window.toggleSourceDetail = function (chipEl, index) {
  const container = chipEl.closest(".sources-container");
  const detail = container.querySelector(`#source-detail-${index}`);

  // Close others
  container.querySelectorAll(".source-detail").forEach((el) => {
    if (el !== detail) el.classList.remove("active");
  });

  detail.classList.toggle("active");
  scrollToBottom();
};

// ── Sidebar Logic ───────────────────────────────────────────
function initSidebar() {
  const toggle = dom.sidebarToggle;
  const overlay = dom.sidebarOverlay;

  if (toggle) {
    toggle.addEventListener("click", () => {
      if (window.innerWidth <= 768) {
        dom.sidebar.classList.toggle("open");
        overlay.classList.toggle("active");
      } else {
        dom.sidebar.classList.toggle("collapsed");
      }
    });
  }

  if (overlay) {
    overlay.addEventListener("click", () => {
      dom.sidebar.classList.remove("open");
      overlay.classList.remove("active");
    });
  }

  const globalBtn = $("#global-chat-btn");
  if (globalBtn) {
    globalBtn.addEventListener("click", selectGlobal);
  }
}

// ── Markdown Renderer (Lightweight) ─────────────────────────
function renderMarkdown(text) {
  if (!text) return "";

  let html = escapeHtml(text);

  // Code blocks (```)
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
  });

  // Inline code (`)
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

  // Bold (**text**)
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  // Italic (*text*)
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

  // Headers
  html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
  html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");

  // Unordered lists
  html = html.replace(/^[-*] (.+)$/gm, "<li>$1</li>");
  html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>");

  // Ordered lists
  html = html.replace(/^\d+\. (.+)$/gm, "<li>$1</li>");

  // Blockquotes
  html = html.replace(/^&gt; (.+)$/gm, "<blockquote>$1</blockquote>");

  // Links
  html = html.replace(
    /\[([^\]]+)\]\(([^)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener">$1</a>'
  );

  // Page references [Page X]
  html = html.replace(
    /\[Page (\d+)\]/g,
    '<span class="source-chip" style="display:inline-flex; padding:2px 8px; font-size:0.7rem; vertical-align:middle; cursor:default;">📄 Page $1</span>'
  );

  // Paragraphs (double newline)
  html = html.replace(/\n\n/g, "</p><p>");

  // Single newlines to <br>
  html = html.replace(/\n/g, "<br>");

  // Wrap in paragraph
  html = `<p>${html}</p>`;

  // Clean up empty paragraphs
  html = html.replace(/<p>\s*<\/p>/g, "");

  return html;
}

// ── Utilities ───────────────────────────────────────────────
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
  });
}

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const icons = { success: "✅", error: "❌", info: "ℹ️", warning: "⚠️" };

  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || "ℹ️"}</span>
    <span class="toast-text">${escapeHtml(message)}</span>
    <button class="toast-close" onclick="this.parentElement.remove()">✕</button>
  `;

  dom.toastContainer.appendChild(toast);

  // Auto-remove after 5 seconds
  setTimeout(() => {
    toast.style.animation = "toastOut 0.3s ease-in forwards";
    setTimeout(() => toast.remove(), 300);
  }, 5000);
}

// Expose deleteDoc globally for onclick handlers
window.deleteDoc = deleteDoc;

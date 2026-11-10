const $ = (selector) => document.querySelector(selector);
const elements = {
  activity: $("#activity-list"),
  activeFileLabel: $("#active-file-label"),
  assistantStatus: $("#task-status"),
  changeCounts: $("#change-counts"),
  changeFileIcon: $("#change-file-icon"),
  changeFileName: $("#change-file-name"),
  changeList: $("#change-list"),
  changePath: $("#change-path"),
  changeTabs: $("#change-tabs"),
  changeView: $("#change-view"),
  connection: $("#connection-label"),
  connectionDot: $("#connection-dot"),
  conversation: $("#conversation"),
  empty: $("#empty-editor"),
  files: $("#file-tree"),
  model: $("#model-select"),
  prompt: $("#prompt-input"),
  project: $("#project-name"),
  projectHeading: $("#tree-project-name"),
  send: $("#send-button"),
  cancel: $("#cancel-task"),
  statusProject: $("#status-project"),
  statusConnection: $("#status-connection"),
  editorContainer: $("#editor-container"),
  fileContentDisplay: $("#file-content-display"),
};
let sessionId = sessionStorage.getItem("harness-session") || crypto.randomUUID();
sessionStorage.setItem("harness-session", sessionId);
let busy = false;
let changedFiles = new Set();
const changes = new Map();
let activityStarted = Date.now();

function activity(message, className = "", detail = "") {
  const empty = elements.activity.querySelector(".activity-empty");
  if (empty) empty.remove();
  const row = document.createElement("div");
  row.className = `activity-entry ${className}`;
  const icon = document.createElement("span");
  icon.className = "activity-symbol";
  icon.textContent = className === "failure" ? "\u00d7" : className === "success" ? "\u2713" : "\u25e6";
  const label = document.createElement("span");
  label.className = "activity-label";
  label.textContent = message;
  const time = document.createElement("span");
  time.className = "activity-time";
  time.textContent = `${Math.max(1, Math.round((Date.now() - activityStarted) / 1000))}s`;
  row.append(icon, label, time);
  if (detail) {
    const output = document.createElement("span");
    output.className = "activity-detail";
    output.textContent = detail;
    row.append(output);
  }
  elements.activity.append(row);
  elements.activity.scrollTop = elements.activity.scrollHeight;
}

function chatMessage(role, text, isError = false) {
  $("#welcome")?.remove();
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}${isError ? " error" : ""}`;
  const heading = document.createElement("div");
  heading.className = "message-heading";
  const avatar = document.createElement("span");
  avatar.className = `avatar${role === "user" ? " user-avatar" : ""}`;
  avatar.textContent = role === "user" ? "N" : "\u2733";
  heading.append(avatar, document.createTextNode(role === "user" ? "You" : "Assistant"));
  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = text;
  wrapper.append(heading, body);
  elements.conversation.append(wrapper);
  elements.conversation.scrollTop = elements.conversation.scrollHeight;
}

function fileIcon(path) {
  const ext = path.split(".").pop().toLowerCase();
  return ({ py: "Py", ts: "TS", tsx: "TS", js: "JS", jsx: "JS", css: "CS", html: "HT", md: "M", json: "{}", yml: "\u2699", yaml: "\u2699" })[ext] || "\u00b7";
}

function renderFiles(files) {
  elements.files.replaceChildren();
  const root = { folders: new Map(), files: [] };
  for (const file of files) {
    const parts = file.path.split("/");
    let node = root;
    for (const part of parts.slice(0, -1)) {
      if (!node.folders.has(part)) node.folders.set(part, { folders: new Map(), files: [] });
      node = node.folders.get(part);
    }
    node.files.push({ name: parts.at(-1), path: file.path });
  }
  function draw(node, depth) {
    for (const [name, folder] of [...node.folders].sort(([a], [b]) => a.localeCompare(b))) {
      const row = document.createElement("div");
      row.className = "tree-row directory";
      row.style.paddingLeft = `${depth * 11}px`;
      const arrow = document.createElement("span");
      arrow.className = "tree-glyph";
      arrow.textContent = "\u2304";
      const label = document.createElement("span");
      label.className = "tree-file-name";
      label.textContent = name;
      row.append(arrow, label);
      elements.files.append(row);
      draw(folder, depth + 1);
    }
    for (const file of node.files.sort((a, b) => a.name.localeCompare(b.name))) {
      const row = document.createElement("div");
      row.className = `tree-row${changedFiles.has(file.path) ? " changed" : ""}`;
      row.title = file.path;
      row.style.paddingLeft = `${depth * 11}px`;
      const icon = document.createElement("span");
      icon.className = "tree-file-icon";
      icon.textContent = changedFiles.has(file.path) ? "M" : fileIcon(file.path);
      const label = document.createElement("span");
      label.className = "tree-file-name";
      label.textContent = file.name;
      row.append(icon, label);
      row.tabIndex = 0;
      row.setAttribute("role", "button");
      row.title = changes.has(file.path) ? `Show changes in ${file.path}` : `Open ${file.path}`;
      row.addEventListener("click", () => switchToDiffOrContent(file.path));
      row.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          switchToDiffOrContent(file.path);
        }
      });
      elements.files.append(row);
    }
  }
  draw(root, 0);
  if (!files.length) {
    const note = document.createElement("div");
    note.className = "loading-note";
    note.textContent = "This workspace has no files yet.";
    elements.files.append(note);
  }
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error(`Could not load workspace (HTTP ${response.status}).`);
    const data = await response.json();
    const name = data.root.replace(/[\\/]+$/, "").split(/[\\/]/).at(-1) || "Workspace";
    elements.project.textContent = name;
    elements.projectHeading.textContent = name.toUpperCase();
    elements.statusProject.textContent = name;
    renderFiles(data.files);
    if (elements.model) {
      elements.model.replaceChildren();
      const models = data.models.includes("qwen3.5:9b") ? data.models : [data.model, ...data.models];
      for (const model of [...new Set(models)]) {
        const option = document.createElement("option");
        option.value = model;
        option.textContent = model + (model === data.model ? " \u00b7 default" : "");
        elements.model.append(option);
      }
      elements.model.value = data.model;
    }
    const connected = data.connected && data.model_available;
    elements.connectionDot.className = `connection-dot ${connected ? "connected" : "error"}`;
    elements.connection.textContent = !data.connected ? "Ollama unavailable" :
      data.model_available ? "Ollama connected" : `${data.model} not installed`;
    elements.statusConnection.textContent = connected ? "\u25c9 Ollama connected" : "\u25c9 Ollama unavailable";
    if (!connected) activity(data.error || `${data.model} is not installed. Start Ollama and run ollama pull ${data.model}.`, "failure");
  } catch (error) {
    elements.connectionDot.className = "connection-dot error";
    elements.connection.textContent = "Ollama unavailable";
    elements.statusConnection.textContent = "\u25c9 Ollama unavailable";
    elements.files.textContent = error.message;
    activity(error.message, "failure");
  }
}

async function loadFileContent(path) {
  const response = await fetch(`/api/file?path=${encodeURIComponent(path)}`);
  if (!response.ok) {
    const data = await response.json();
    throw new Error(data.error || `Failed to load file (${response.status}).`);
  }
  return response.text();
}

async function renderDiff(change) {
  const isNewChange = !changes.has(change.path);
  changes.set(change.path, change);
  changedFiles.add(change.path);
  elements.empty.hidden = true;
  elements.changeView.hidden = false;
  elements.editorContainer.style.display = "none";
  if (elements.fileContentDisplay) {
    elements.fileContentDisplay.textContent = "";
  }
  elements.changeTabs.replaceChildren();
  for (const [path, fileChange] of changes) {
    const tab = document.createElement("button");
    tab.className = `change-tab${path === change.path ? " active" : ""}`;
    tab.type = "button";
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-selected", String(path === change.path));
    const name = document.createElement("span");
    name.textContent = path.split("/").at(-1);
    const counts = document.createElement("span");
    counts.className = "change-tab-counts";
    counts.textContent = `+${fileChange.added} \u2212${fileChange.removed}`;
    tab.append(name, counts);
    tab.addEventListener("click", () => renderDiff(changes.get(path)));
    elements.changeTabs.append(tab);
  }
  elements.changeFileName.textContent = change.path.split("/").at(-1);
  elements.changePath.textContent = change.path;
  elements.activeFileLabel.textContent = change.path;
  elements.changeFileIcon.textContent = fileIcon(change.path);
  elements.changeCounts.replaceChildren();
  const addCount = document.createElement("span");
  addCount.style.color = "#81b88b";
  addCount.textContent = `+${change.added} `;
  const removeCount = document.createElement("span");
  removeCount.style.color = "#e88787";
  removeCount.textContent = `\u2212${change.removed}`;
  elements.changeCounts.append(addCount, removeCount);
  elements.changeList.replaceChildren();
  let oldNumber = 0;
  let newNumber = 0;
  for (const line of change.diff) {
    let type = "context";
    let marker = " ";
    let text = line;
    if (line.startsWith("@@")) {
      type = "hunk";
      marker = "";
      const match = line.match(/@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
      if (match) { oldNumber = Number(match[1]) - 1; newNumber = Number(match[2]) - 1; }
    } else if (line.startsWith("+++") || line.startsWith("---")) {
      type = "meta";
      marker = "";
    } else if (line.startsWith("+")) {
      type = "addition";
      marker = "+";
      text = line.slice(1);
      newNumber++;
    } else if (line.startsWith("-")) {
      type = "deletion";
      marker = "\u2212";
      text = line.slice(1);
      oldNumber++;
    } else {
      if (type !== "hunk" && type !== "meta") { oldNumber++; newNumber++; }
      text = line.startsWith(" ") ? line.slice(1) : line;
    }
    const row = document.createElement("div");
    row.className = `diff-line ${type}`;
    const numbers = document.createElement("span");
    numbers.className = "line-number";
    numbers.textContent = type === "addition" ? newNumber : type === "deletion" ? oldNumber : `${newNumber || ""}`;
    const prefix = document.createElement("span");
    prefix.className = "diff-marker";
    prefix.textContent = marker;
    const content = document.createElement("code");
    content.className = "diff-content";
    content.textContent = text;
    row.append(numbers, prefix, content);
    elements.changeList.append(row);
  }
  if (isNewChange) loadStatus();
}

function setBusy(next) {
  busy = next;
  if (elements.send) elements.send.disabled = next;
  if (elements.prompt) elements.prompt.disabled = next;
  if (elements.cancel) {
    elements.cancel.hidden = !next;
    elements.cancel.disabled = false;
  }
  elements.assistantStatus.className = `task-status${next ? " running" : ""}`;
  elements.assistantStatus.innerHTML = next ? "<i></i> Working" : "<i></i> Ready";
}

async function cancelTask() {
  if (!busy || !elements.cancel) return;
  elements.cancel.disabled = true;
  try {
    const response = await fetch("/api/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Stop request failed (HTTP ${response.status}).`);
    if (!data.cancelled) throw new Error("No running task was found to stop.");
    activity("Stop requested");
  } catch (error) {
    activity(error.message, "failure");
    elements.cancel.disabled = false;
  }
}

async function sendMessage(text = elements.prompt?.value || "") {
  if (busy || !text.trim()) return;
  const message = text.trim();
  elements.prompt.value = "";
  setBusy(true);
  activityStarted = Date.now();
  chatMessage("user", message);
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId, model: elements.model.value }),
    });
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error || `Request failed (HTTP ${response.status}).`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let ended = false;
    while (!ended) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        switch (event.type) {
          case "status":
            elements.assistantStatus.innerHTML = `<i></i> ${event.text}`;
            activity(event.text);
            break;
          case "user_message": break;
          case "assistant_message":
            chatMessage("assistant", event.text);
            break;
          case "tool_started":
            {
              const origin = event.origin === "user_request" ? " \u00b7 requested by you" :
                event.origin === "verification" ? " \u00b7 verification" : "";
              const label = `${event.tool.replaceAll("_", " ")} \u00b7 ${event.label}${origin}`;
              elements.assistantStatus.innerHTML = `<i></i> ${label}`;
              activity(label);
            }
            break;
          case "tool_completed":
            activity(`${event.tool.replaceAll("_", " ")} \u00b7 ${event.label}`, event.success ? "success" : "failure", event.result);
            if (!event.success) elements.assistantStatus.className = "task-status error";
            break;
          case "change":
            renderDiff(event);
            break;
          case "error":
            chatMessage("assistant", event.message, true);
            activity(event.message, "failure");
            break;
          case "complete":
            ended = true;
            break;
        }
      }
      if (done) break;
    }
    if (buffer.trim()) {
      const event = JSON.parse(buffer);
      if (event.type === "error") chatMessage("assistant", event.message, true);
    }
  } catch (error) {
    chatMessage("assistant", error.message, true);
    activity(error.message, "failure");
    elements.assistantStatus.className = "task-status error";
  } finally {
    setBusy(false);
    loadStatus();
  }
}

function switchToDiff(path) {
  if (changes.has(path)) {
    renderDiff(changes.get(path));
    // Remove active class from other tabs
    elements.changeTabs.querySelectorAll(".change-tab").forEach(tab => {
      tab.classList.remove("active");
      tab.setAttribute("aria-selected", "false");
    });
  }
}

async function switchToFileContent(path) {
  elements.empty.hidden = true;
  elements.changeView.hidden = true;
  elements.editorContainer.style.display = "flex";
  elements.activeFileLabel.textContent = path;
  try {
    elements.fileContentDisplay.textContent = await loadFileContent(path);
  } catch (error) {
    chatMessage("assistant", `Error loading file: ${error.message}`, true);
  }
}

async function openFileInEditor(path) {
  await switchToFileContent(path);
}

async function switchToDiffOrContent(path) {
  if (changes.has(path)) {
    renderDiff(changes.get(path));
  } else {
    await openFileInEditor(path);
  }
}

elements.send?.addEventListener("click", () => sendMessage());
elements.prompt?.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  } else if (event.key === "Enter" && event.ctrlKey) {
    event.preventDefault();
    sendMessage();
  }
});
document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.dataset.prompt));
});
$("#new-chat").addEventListener("click", newChat);
elements.cancel?.addEventListener("click", cancelTask);
$("#clear-chat")?.addEventListener("click", newChat);
$("#clear-activity").addEventListener("click", () => {
  elements.activity.replaceChildren();
  const note = document.createElement("div");
  note.className = "activity-empty";
  note.textContent = "Activity will appear when you start a task.";
  elements.activity.append(note);
});
$("#refresh-files").addEventListener("click", loadStatus);
$("#toggle-activity").addEventListener("click", () => {
  const panel = $("#activity-panel");
  panel.hidden = !panel.hidden;
});

function newChat() {
  if (busy) return;
  sessionId = crypto.randomUUID();
  sessionStorage.setItem("harness-session", sessionId);
  elements.conversation.replaceChildren();
  const welcome = document.createElement("div");
  welcome.className = "welcome";
  welcome.id = "welcome";
  welcome.innerHTML = '<div class="welcome-icon">\u2733</div><div class="welcome-title">Build with your local AI</div><p>Ask a question or describe a change. The assistant will inspect your project, make real edits, and show each diff.</p><button class="suggestion" data-prompt="Explore this project and give me a concise overview."><span>\u2197</span> Explore this project</button>';
  elements.conversation.append(welcome);
  welcome.querySelector(".suggestion").addEventListener("click", (event) => sendMessage(event.currentTarget.dataset.prompt));
  elements.activity.replaceChildren();
  changedFiles.clear();
  changes.clear();
  elements.changeTabs.replaceChildren();
  elements.changeView.hidden = true;
  elements.editorContainer.style.display = "none";
  elements.empty.hidden = false;
  elements.activeFileLabel.textContent = "Session";
  loadStatus();
}

loadStatus();

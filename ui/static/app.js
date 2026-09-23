/**
 * SEVERANCE Client Application Logic
 */

const THEME_STORAGE_KEY = "severance-theme";

/** Applies a theme (persisted to localStorage) and syncs every toggle
 * button's icon on the page. The actual FIRST paint's theme is set by a
 * small blocking inline script at the top of each page's <head> (before
 * this file loads) to avoid a flash of the wrong theme -- this function is
 * what the toggle button itself calls afterward, and what syncs icons once
 * the DOM is ready. */
function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch (e) { /* private browsing / storage disabled -- theme just won't persist */ }
  document.querySelectorAll(".theme-toggle-icon").forEach(icon => {
    icon.textContent = theme === "light" ? "dark_mode" : "light_mode";
  });
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme") || "dark";
  setTheme(current === "dark" ? "light" : "dark");
}

/** Call once on page load to sync toggle icon(s) to whatever theme the
 * blocking init script already applied (so the icon doesn't briefly show
 * the wrong state before this file finishes loading). */
function syncThemeToggleIcon() {
  const current = document.documentElement.getAttribute("data-theme") || "dark";
  document.querySelectorAll(".theme-toggle-icon").forEach(icon => {
    icon.textContent = current === "light" ? "dark_mode" : "light_mode";
  });
}

async function fetchAPI(url, options = {}) {
  const defaultOptions = {
    headers: {
      "Content-Type": "application/json",
    },
    credentials: "same-origin",
  };

  const merged = {
    ...defaultOptions,
    ...options,
    headers: {
      ...defaultOptions.headers,
      ...(options.headers || {}),
    },
  };

  const res = await fetch(url, merged);
  if (res.status === 401 && !url.includes("/auth/login")) {
    window.location.href = "/login";
    throw new Error("Session expired. Please log in.");
  }
  return res;
}

async function getCurrentPrincipal() {
  try {
    const res = await fetchAPI("/me");
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.error("Auth check failed:", err);
  }
  return null;
}

async function doLogout() {
  try {
    await fetchAPI("/auth/logout", { method: "POST" });
  } finally {
    window.location.href = "/login";
  }
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/**
 * Minimal, dependency-free Markdown -> HTML renderer.
 * Input is always HTML-escaped first, so no raw HTML from the source
 * text (e.g. an LLM answer) can inject markup or scripts.
 */
function renderMarkdown(source) {
  const escaped = escapeHtml(source || "");

  // Inline formatting: bold, italic, inline code
  const inline = (text) => text
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, "<em>$1</em>");

  const lines = escaped.split(/\r?\n/);
  const htmlParts = [];
  let listBuffer = [];
  let listType = null;
  let inCodeBlock = false;
  let codeBuffer = [];
  let codeLang = "";

  const flushList = () => {
    if (listBuffer.length > 0) {
      htmlParts.push(`<${listType}>${listBuffer.join("")}</${listType}>`);
      listBuffer = [];
      listType = null;
    }
  };

  for (const rawLine of lines) {
    // Fenced code blocks (```lang ... ```): matched against the RAW line
    // (not blockquote-stripped, not inline-formatted) since code content
    // must render verbatim -- no bold/italic/inline-code processing inside
    // a block that's already a code block. `escaped` already HTML-escaped
    // this whole source once at the top, so code content here is already
    // safe to insert as-is (no injection risk from an LLM answer).
    const fenceMatch = rawLine.trim().match(/^```([\w+-]*)\s*$/);
    if (fenceMatch) {
      if (!inCodeBlock) {
        flushList();
        inCodeBlock = true;
        codeLang = fenceMatch[1] || "";
        codeBuffer = [];
      } else {
        inCodeBlock = false;
        const langClass = codeLang ? ` class="language-${escapeHtml(codeLang)}"` : "";
        htmlParts.push(`<pre><code${langClass}>${codeBuffer.join("\n")}</code></pre>`);
      }
      continue;
    }
    if (inCodeBlock) {
      codeBuffer.push(rawLine);
      continue;
    }

    // Strip leading Markdown blockquote markers ("> ", possibly repeated/nested)
    // so a line like "> ## Heading" is still recognized as a heading rather than
    // falling through to a literal, unrendered paragraph. Matches against the
    // already-HTML-escaped text, so ">" here is "&gt;".
    const line = rawLine.trim().replace(/^(?:&gt;\s*)+/, "");

    if (line === "") {
      flushList();
      continue;
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      flushList();
      const level = headingMatch[1].length;
      htmlParts.push(`<h${level}>${inline(headingMatch[2])}</h${level}>`);
      continue;
    }

    const ulMatch = line.match(/^[-*]\s+(.*)$/);
    const olMatch = line.match(/^\d+\.\s+(.*)$/);
    if (ulMatch) {
      if (listType !== "ul") { flushList(); listType = "ul"; }
      listBuffer.push(`<li>${inline(ulMatch[1])}</li>`);
      continue;
    }
    if (olMatch) {
      if (listType !== "ol") { flushList(); listType = "ol"; }
      listBuffer.push(`<li>${inline(olMatch[1])}</li>`);
      continue;
    }

    flushList();
    htmlParts.push(`<p>${inline(line)}</p>`);
  }
  if (inCodeBlock && codeBuffer.length > 0) {
    // Generation ended mid-block (e.g. hit a length cap) -- flush what was
    // captured rather than silently dropping it.
    htmlParts.push(`<pre><code>${codeBuffer.join("\n")}</code></pre>`);
  }
  flushList();

  return htmlParts.join("");
}

function renderUserChip(principal, containerId = "userChipContainer") {
  const container = document.getElementById(containerId);
  if (!container || !principal) return;

  const comps = principal.compartments && principal.compartments.length > 0
    ? principal.compartments.join(", ")
    : "none";

  container.innerHTML = `
    <div class="user-chip">
      <div class="user-info">
        <div class="user-name">${principal.name} (${principal.job_title})</div>
        <div class="user-grade">Grade ${principal.grade} // Compartments: [${comps}]</div>
      </div>
      <button onclick="doLogout()">Sign Out</button>
    </div>
  `;
}

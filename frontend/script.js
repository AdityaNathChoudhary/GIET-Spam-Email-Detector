const MAX_CHARS = 2000;

// Use relative URL when served from FastAPI; fall back to localhost:8000 for file:// or dev servers
const API_BASE_URL =
  window.location.protocol === "file:" ||
  (window.location.port && window.location.port !== "8000")
    ? "http://127.0.0.1:8000"
    : "";

const spamForm = document.getElementById("spamForm");
const messageInput = document.getElementById("messageInput");
const charCounter = document.getElementById("charCounter");
const errorBanner = document.getElementById("errorBanner");
const checkBtn = document.getElementById("checkBtn");
const btnLabel = document.getElementById("btnLabel");
const btnIcon = checkBtn.querySelector(".btn-icon");
const btnSpinner = checkBtn.querySelector(".btn-spinner");

const resultCard = document.getElementById("resultCard");
const hamIcon = document.getElementById("hamIcon");
const spamIcon = document.getElementById("spamIcon");
const resultHeading = document.getElementById("resultHeading");
const resultSubtext = document.getElementById("resultSubtext");
const confidenceValue = document.getElementById("confidenceValue");
const progressTrack = document.getElementById("progressTrack");
const progressFill = document.getElementById("progressFill");

function updateCharCounter() {
  const length = messageInput.value.length;
  charCounter.textContent = `${length} / ${MAX_CHARS}`;
  charCounter.classList.toggle("near-limit", length >= MAX_CHARS * 0.85 && length < MAX_CHARS);
  charCounter.classList.toggle("at-limit", length >= MAX_CHARS);

  if (messageInput.value.trim()) {
    clearError();
  }
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.hidden = false;
  messageInput.classList.add("invalid");
}

function clearError() {
  errorBanner.textContent = "";
  errorBanner.hidden = true;
  messageInput.classList.remove("invalid");
}

function setLoading(isLoading) {
  checkBtn.disabled = isLoading;
  btnIcon.hidden = isLoading;
  btnSpinner.hidden = !isLoading;
  btnLabel.textContent = isLoading ? "Checking..." : "Check for Spam";
}

function renderResult(label, confidence) {
  const isSpam = label === "spam";
  const pct = Math.max(0, Math.min(100, confidence * 100));
  const formattedPct = `${pct.toFixed(1)}%`;

  resultCard.classList.remove("is-ham", "is-spam");
  resultCard.classList.add(isSpam ? "is-spam" : "is-ham");

  hamIcon.hidden = isSpam;
  spamIcon.hidden = !isSpam;

  resultHeading.textContent = isSpam ? "Spam Detected" : "Ham (Not Spam)";
  resultSubtext.textContent = isSpam
    ? "This message looks like spam. Be cautious!"
    : "This message looks safe.";

  confidenceValue.textContent = formattedPct;
  progressTrack.setAttribute("aria-valuenow", pct.toFixed(1));

  resultCard.hidden = false;
  progressFill.style.width = "0%";
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      progressFill.style.width = `${pct.toFixed(1)}%`;
    });
  });
}

async function handleSubmit(event) {
  event.preventDefault();
  clearError();

  const text = messageInput.value.trim();
  if (!text) {
    showError("Please enter an email or SMS message to check.");
    messageInput.focus();
    return;
  }

  setLoading(true);

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(data.detail || `Server returned status ${response.status}`);
    }

    renderResult(data.label, data.confidence);
  } catch (err) {
    const message =
      err instanceof TypeError
        ? "Cannot reach the FastAPI server at http://127.0.0.1:8000."
        : err.message || "An unexpected error occurred.";
    showError(message);
  } finally {
    setLoading(false);
  }
}

messageInput.addEventListener("input", updateCharCounter);
spamForm.addEventListener("submit", handleSubmit);

// Allow Ctrl+Enter / Cmd+Enter to trigger prediction
messageInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    spamForm.requestSubmit();
  }
});

updateCharCounter();

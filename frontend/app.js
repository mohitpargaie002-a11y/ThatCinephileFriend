const form = document.querySelector("#search-form");
const results = document.querySelector("#results");
const status = document.querySelector("#status");
const template = document.querySelector("#card");
const queryInput = document.querySelector("#query");

setReadyState(false, "Waking up the recommendation engine…");
waitForApi();

// Handle quick suggestion chips
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const mode = chip.dataset.mode || "discover";
    const query = chip.dataset.query || "";
    form.mode.value = mode;
    queryInput.value = query;
    form.dispatchEvent(new Event("submit", { cancelable: true }));
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  const mode = form.mode.value;

  if (mode === "similar" && !query) {
    return setStatus("Enter a title to find similar picks.");
  }

  const params = new URLSearchParams({ limit: "12" });
  if (query) params.set(mode === "similar" ? "title" : "query", query);
  const type = form.type.value;
  if (type) params.set("media_type", type);
  if (form.niche.checked) params.set("niche", "true");

  results.innerHTML = "";
  setStatus("Asking the cinephile…");

  try {
    const response = await fetch(`/${mode}?${params}`);
    const rawText = await response.text();
    let body = {};
    try {
      body = rawText ? JSON.parse(rawText) : {};
    } catch {
      body = { detail: rawText || `Server returned status ${response.status}` };
    }

    if (!response.ok) {
      throw new Error(body.detail || `Request failed (${response.status})`);
    }

    const items = body.results || [];
    render(items);
    setStatus(
      items.length
        ? `${items.length} picks for your watchlist.`
        : "No matches — try a wider search or different mood.",
    );
  } catch (error) {
    console.error("Search error:", error);
    setStatus(error.message || "Unable to fetch recommendations. Please try again.");
  }
});

function render(items) {
  items.forEach((item) => {
    const card = template.content.cloneNode(true);
    const image = card.querySelector("img");
    image.src =
      item.poster_url ||
      "https://placehold.co/500x750/2a202c/f6eee3?text=No+poster";
    image.alt = `${item.title} poster`;
    card.querySelector("h2").textContent = item.title;
    card.querySelector(".type").textContent = item.media_type;
    card.querySelector(".rating").textContent =
      `★ ${Number(item.vote_average || 0).toFixed(1)} · ${Number(item.vote_count || 0).toLocaleString()} ratings`;
    card.querySelector(".overview").textContent = item.overview || "No overview available.";
    card.querySelector(".genres").textContent = (item.genres || []).join(" · ");
    results.append(card);
  });
}

function setStatus(message) {
  status.textContent = message;
}

function setReadyState(ready, message) {
  form.querySelectorAll("input, button").forEach((element) => {
    element.disabled = !ready;
  });
  status.classList.toggle("loading", !ready);
  status.textContent = message;
  if (!ready) {
    status.prepend(
      Object.assign(document.createElement("span"), {
        className: "spinner",
        ariaHidden: "true",
      }),
    );
  }
}

async function waitForApi() {
  try {
    const response = await fetch("/health", { cache: "no-store" });
    if (!response.ok) throw new Error();
    const health = await response.json();
    setReadyState(
      true,
      health.qdrant_configured
        ? "Pick a mood, a genre, or a favourite title."
        : "API is awake, but Qdrant has not been configured.",
    );
  } catch {
    setTimeout(waitForApi, 2000);
  }
}

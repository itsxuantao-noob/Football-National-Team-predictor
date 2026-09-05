const el = (id) => document.getElementById(id);

const form = el("predict-form");
const homeInput = el("home-team");
const awayInput = el("away-team");
const errorBox = el("error");
const resultCard = el("result");
const submitButton = el("submit");

let eloByTeam = new Map();

const percent = (value) => `${(value * 100).toFixed(1)}%`;

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = !message;
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed (${response.status})`);
  }
  return payload;
}

async function loadModelStatus() {
  try {
    const health = await requestJson("/api/health");
    if (health.status !== "ok") {
      el("model-status").textContent = "No trained model found. Run `wcp train` and refresh.";
      submitButton.disabled = true;
      return false;
    }
    const trained = new Date(health.trained_at).toLocaleDateString();
    el("model-status").textContent =
      `${health.model} (${health.calibration} calibration) - ${health.teams} teams - trained ${trained}`;
    return true;
  } catch (error) {
    el("model-status").textContent = `Could not reach the API: ${error.message}`;
    submitButton.disabled = true;
    return false;
  }
}

async function loadTeams() {
  const teams = await requestJson("/api/teams");
  eloByTeam = new Map(teams.map((row) => [row.team, row.elo]));

  const datalist = el("team-list");
  datalist.replaceChildren(
    ...teams.map((row) => {
      const option = document.createElement("option");
      option.value = row.team;
      option.label = `ELO ${Math.round(row.elo)}`;
      return option;
    })
  );

  if (teams.length >= 2) {
    homeInput.value = teams[0].team;
    awayInput.value = teams[1].team;
    updateElo();
  }
}

function updateElo() {
  for (const [input, target] of [
    [homeInput, el("home-elo")],
    [awayInput, el("away-elo")],
  ]) {
    const elo = eloByTeam.get(input.value.trim());
    target.textContent = elo === undefined ? "" : `ELO ${Math.round(elo)}`;
  }
}

async function loadFooterMetrics() {
  try {
    const metrics = await requestJson("/api/metrics");
    const test = metrics.test.model;
    el("footer-metrics").textContent =
      `Held-out test (${metrics.split.test_start} onwards, ${metrics.split.sizes.test} matches): ` +
      `log loss ${test.log_loss.toFixed(4)}, accuracy ${(test.accuracy * 100).toFixed(1)}%`;
  } catch {
    /* metrics are a nice-to-have; the predictor still works without them */
  }
}

function renderSegment(id, value, label) {
  const node = el(id);
  node.style.width = `${value * 100}%`;
  node.textContent = value >= 0.12 ? percent(value) : "";
  node.title = `${label}: ${percent(value)}`;
}

function renderAdvance(prediction) {
  const container = el("advance");
  if (prediction.home_advance === null || prediction.home_advance === undefined) {
    container.hidden = true;
    return;
  }

  const rows = [
    { team: prediction.home_team, value: prediction.home_advance, color: "var(--home)" },
    { team: prediction.away_team, value: prediction.away_advance, color: "var(--away)" },
  ];

  el("advance-rows").replaceChildren(
    ...rows.map(({ team, value, color }) => {
      const row = document.createElement("div");
      row.className = "advance-row";

      const name = document.createElement("span");
      name.textContent = team;

      const track = document.createElement("div");
      track.className = "advance-track";
      const fill = document.createElement("div");
      fill.className = "advance-fill";
      fill.style.width = `${value * 100}%`;
      fill.style.background = color;
      track.append(fill);

      const number = document.createElement("span");
      number.className = "advance-value";
      number.textContent = percent(value);

      row.append(name, track, number);
      return row;
    })
  );
  container.hidden = false;
}

function renderPrediction(prediction) {
  el("result-title").textContent = `${prediction.home_team} vs ${prediction.away_team}`;

  const venue = prediction.neutral ? "Neutral venue" : `${prediction.home_team} at home`;
  const diff = prediction.home_elo - prediction.away_elo;
  el("result-context").textContent =
    `${venue} - ${prediction.competition} - ELO ${Math.round(prediction.home_elo)} vs ` +
    `${Math.round(prediction.away_elo)} (${diff >= 0 ? "+" : ""}${Math.round(diff)})`;

  renderSegment("seg-home", prediction.home_win, `${prediction.home_team} win`);
  renderSegment("seg-draw", prediction.draw, "Draw");
  renderSegment("seg-away", prediction.away_win, `${prediction.away_team} win`);

  el("label-home").textContent = `${prediction.home_team} win ${percent(prediction.home_win)}`;
  el("label-draw").textContent = `Draw ${percent(prediction.draw)}`;
  el("label-away").textContent = `${prediction.away_team} win ${percent(prediction.away_win)}`;

  renderAdvance(prediction);
  resultCard.hidden = false;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  showError("");
  submitButton.disabled = true;

  try {
    const prediction = await requestJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        home_team: homeInput.value.trim(),
        away_team: awayInput.value.trim(),
        neutral: el("neutral").checked,
        competition: el("competition").value,
        knockout: el("knockout").checked,
      }),
    });
    renderPrediction(prediction);
  } catch (error) {
    resultCard.hidden = true;
    showError(error.message);
  } finally {
    submitButton.disabled = false;
  }
});

el("swap").addEventListener("click", () => {
  [homeInput.value, awayInput.value] = [awayInput.value, homeInput.value];
  updateElo();
});

homeInput.addEventListener("input", updateElo);
awayInput.addEventListener("input", updateElo);

(async function init() {
  if (await loadModelStatus()) {
    await Promise.all([loadTeams(), loadFooterMetrics()]);
  }
})();

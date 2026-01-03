// static/sketch.js

let scenarioData = null;
let simulationState = null;
let nodePositions = {};
let hoveredNodeId = -1;

function setup() {
  // Target the graph panel in Left Column
  let container = document.getElementById('graph-panel');
  let w = container.clientWidth;
  let h = container.clientHeight;

  let canvas = createCanvas(w, h);
  canvas.parent('graph-panel');
  textAlign(CENTER, CENTER);

  // 1. Initialize Configuration UI (Right Column)
  initScenarioSelector();
  initModelSelector();

  // 2. Button Listeners (Control Panel - Left Column)
  document.getElementById('btn-reveal').addEventListener('click', () => {
    fetch('/api/simulation/step/reveal', { method: 'POST' }).then(res => res.json()).then(updateSimulationState);
  });
  document.getElementById('btn-explore').addEventListener('click', () => {
    fetch('/api/simulation/step/explore', { method: 'POST' }).then(res => res.json()).then(updateSimulationState);
  });
  document.getElementById('btn-decision').addEventListener('click', () => {
    fetch('/api/simulation/step/decision', { method: 'POST' }).then(res => res.json()).then(updateSimulationState);
  });

  // 3. Apply/Reset Button (Right Column)
  document.getElementById('btn-apply').addEventListener('click', applyConfiguration);
}

// --- CONFIGURATION LOGIC ---

function initScenarioSelector() {
  fetch('/api/scenarios_list')
    .then(res => res.json())
    .then(list => {
      let select = document.getElementById('scenario-select');
      list.forEach(item => {
        let opt = document.createElement('option');
        opt.value = item.id;
        opt.innerText = item.name;
        select.appendChild(opt);
      });
      // Initial load is handled when Model selector finishes or Apply is clicked
    });
}

function initModelSelector() {
  fetch('/api/models_list')
    .then(res => res.json())
    .then(list => {
      let select = document.getElementById('model-select');
      list.forEach(name => {
        let opt = document.createElement('option');
        opt.value = name;
        opt.innerText = name;
        select.appendChild(opt);
      });

      // Load Schema for first model and trigger initial setup
      if (list.length > 0) {
        loadModelSchema(list[0]);
      }

      select.addEventListener('change', (e) => loadModelSchema(e.target.value));
    });
}

function loadModelSchema(modelName) {
  fetch(`/api/model_schema/${modelName}`)
    .then(res => res.json())
    .then(schema => {
      renderConfigInputs(schema);
      // Auto-apply initial configuration on first load
      if (!scenarioData) applyConfiguration();
    });
}

function renderConfigInputs(schema) {
  const container = document.getElementById('params-container');
  container.innerHTML = '';

  for (const [key, meta] of Object.entries(schema)) {
    let section = document.createElement('div');
    section.className = 'config-section';

    let label = document.createElement('label');
    label.className = 'config-label';
    label.innerText = meta.label || key;

    let input = document.createElement('input');
    input.className = 'config-input param-input';
    input.type = 'number';
    input.name = key;
    input.value = meta.default;

    if (meta.min !== undefined) input.min = meta.min;
    if (meta.max !== undefined) input.max = meta.max;
    if (meta.step !== undefined) input.step = meta.step;

    section.appendChild(label);
    section.appendChild(input);
    container.appendChild(section);
  }
}

function applyConfiguration() {
  const scenarioId = document.getElementById('scenario-select').value || 1;
  const modelName = document.getElementById('model-select').value || "SimpleLearner";

  // Gather Params
  let params = {};
  document.querySelectorAll('.param-input').forEach(input => {
    params[input.name] = parseFloat(input.value);
  });

  const payload = {
    scenarioId: scenarioId,
    modelName: modelName,
    params: params
  };

  fetch('/api/init', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
    .then(res => res.json())
    .then(data => {
      scenarioData = data;
      calculateNodePositions();

      // Fetch fresh state
      fetch('/api/simulation/state').then(r => r.json()).then(updateSimulationState);
    })
    .catch(err => console.error("Init failed:", err));
}

// --- VISUALIZATION LOGIC ---

function windowResized() {
  let container = document.getElementById('graph-panel');
  if (container) {
    resizeCanvas(container.clientWidth, container.clientHeight);
    calculateNodePositions();
  }
}

function updateSimulationState(state) {
  simulationState = state;

  // 1. Matrix
  if (state && state.matrix) {
    renderMatrixHTML(state.matrix);
  }

  // 2. Probs
  const isDecisionPhase = (state.phase === "DECISION");
  const currentProbs = state.decision_probs || Array(9).fill(0);
  renderProbabilities(currentProbs, isDecisionPhase);

  // 3. UI Controls
  const btnReveal = document.getElementById('btn-reveal');
  const btnExplore = document.getElementById('btn-explore');
  const btnDecision = document.getElementById('btn-decision');
  const statusText = document.getElementById('status-text');

  // Reset opacities
  btnReveal.style.opacity = "0.5"; btnReveal.disabled = true;
  btnExplore.style.opacity = "0.5"; btnExplore.disabled = true;
  btnDecision.style.opacity = "0.5"; btnDecision.disabled = true;
  statusText.style.color = "#888";

  if (state.phase === "REVEAL") {
    btnReveal.disabled = false;
    btnReveal.style.opacity = "1";
    statusText.innerText = `Phase 1: Pre-Exposure (${state.reveal_progress} / ${state.reveal_total})`;
  }
  else if (state.phase === "EXPLORE") {
    btnExplore.disabled = false;
    btnExplore.style.opacity = "1";
    let timeStr = state.elapsed_time ? state.elapsed_time.toFixed(1) : "0.0";
    statusText.innerText = `Phase 2: Exploration (${timeStr}s / ${state.time_limit}s)`;
  }
  else if (state.phase === "DECISION") {
    btnDecision.disabled = false;
    btnDecision.style.opacity = "1";
    statusText.innerText = "Phase 3: Decision (Select Path)";
    statusText.style.color = "#FFD700";
  }
  else if (state.phase === "FINISHED") {
    if (state.path_valid === true) {
      statusText.innerText = "Goal Reached! (Valid Path)";
      statusText.style.color = "#2ecc71";
    } else if (state.path_valid === false) {
      statusText.innerText = "Failed: Invalid Path Detected.";
      statusText.style.color = "#e74c3c";
    } else {
      statusText.innerText = "Simulation Finished.";
    }
  }
}

function calculateNodePositions() {
  if (!scenarioData) return;
  // Use margins relative to the canvas size
  let margin = 80;
  let availableWidth = width - 2 * margin;
  let availableHeight = height - 2 * margin;

  scenarioData.nodes.forEach(node => {
    let x = margin + node.grid_x * (availableWidth / 2);
    let y = margin + node.grid_y * (availableHeight / 2);
    nodePositions[node.id] = createVector(x, y);
  });
}

function renderMatrixHTML(matrix) {
  const container = document.getElementById('matrix-container');
  container.innerHTML = '';
  container.classList.remove('matrix-placeholder');

  const size = matrix.length;

  // Header Row
  container.appendChild(document.createElement('div'));
  for (let i = 0; i < size; i++) {
    let label = document.createElement('div');
    label.className = 'matrix-label';
    label.innerText = i;
    container.appendChild(label);
  }

  for (let row = 0; row < size; row++) {
    let rowLabel = document.createElement('div');
    rowLabel.className = 'matrix-label';
    rowLabel.innerText = row;
    container.appendChild(rowLabel);

    for (let col = 0; col < size; col++) {
      let val = matrix[row][col];
      let cell = document.createElement('div');
      cell.className = 'matrix-cell';
      let intensity = Math.min(val, 1.0);
      cell.style.backgroundColor = `rgba(52, 152, 219, ${0.1 + (intensity * 0.9)})`;
      cell.innerText = val.toFixed(2).replace('0.', '.');
      if (row === col) cell.style.border = "1px solid #555";
      container.appendChild(cell);
    }
  }
}

function renderProbabilities(probs, isActive) {
  const container = document.getElementById('probs-container');
  container.innerHTML = '';

  for (let i = 0; i < 9; i++) {
    let p = probs[i];

    let col = document.createElement('div');
    col.className = 'vector-col';

    let label = document.createElement('div');
    label.className = 'vector-label';
    label.innerText = i;

    let cell = document.createElement('div');
    cell.className = 'vector-cell';

    if (!isActive) {
      cell.style.backgroundColor = "#2a2a2a";
      cell.style.color = "#444";
      cell.innerText = "0.0";
    } else {
      cell.innerText = p.toFixed(2).replace(/^0+/, '');
      if (cell.innerText === '.') cell.innerText = '.00';

      let alpha = 0.1 + (p * 0.9);
      if (p < 0.01) {
        cell.style.backgroundColor = "#2a2a2a";
        cell.style.color = "#666";
      } else {
        cell.style.backgroundColor = `rgba(46, 204, 113, ${alpha})`;
        cell.style.color = "#fff";
        if (p > 0.2) cell.style.fontWeight = "bold";
      }
    }

    col.appendChild(label);
    col.appendChild(cell);
    container.appendChild(col);
  }
}

// --- MAIN DRAW LOOP (Same logic, just ensuring it runs) ---
function draw() {
  background(30);

  if (!scenarioData) {
    fill(150);
    noStroke();
    text("Loading...", width / 2, height / 2);
    return;
  }

  // 1. Detect Hover
  hoveredNodeId = -1;
  for (let node of scenarioData.nodes) {
    let pos = nodePositions[node.id];
    if (dist(mouseX, mouseY, pos.x, pos.y) < 35) {
      hoveredNodeId = node.id;
      break;
    }
  }

  // 2. State Variables
  let backendActiveId = (simulationState && simulationState.active_node !== null)
    ? simulationState.active_node
    : -1;
  let decisionPath = (simulationState && simulationState.decision_path)
    ? simulationState.decision_path
    : [];

  // 3. Draw Edges (Standard)
  scenarioData.edges.forEach(edge => {
    drawConnection(edge[0], edge[1], nodePositions, scenarioData, color(80), 2);
  });

  // 4. Draw Outgoing Arrows (White)
  scenarioData.edges.forEach(edge => {
    let startId = edge[0];
    if (startId === backendActiveId || startId === hoveredNodeId) {
      drawConnection(edge[0], edge[1], nodePositions, scenarioData, color(255), 4);
    }
  });

  // 5. Draw Decision Path (Gold)
  if (decisionPath.length > 1) {
    for (let i = 0; i < decisionPath.length - 1; i++) {
      drawConnection(decisionPath[i], decisionPath[i + 1], nodePositions, scenarioData, color(255, 204, 0), 4);
    }
  }

  // 6. Draw Nodes
  scenarioData.nodes.forEach(node => {
    let pos = nodePositions[node.id];
    let isHover = (node.id === hoveredNodeId);
    let isActive = (node.id === backendActiveId);
    let isInPath = decisionPath.includes(node.id);
    let size = (isHover || isActive) ? 75 : 60;

    if (isInPath || (isActive && simulationState.phase === "DECISION")) {
      stroke(255, 204, 0); strokeWeight(4);
    } else if (isActive || isHover) {
      stroke(255); strokeWeight(3);
    } else {
      stroke(20); strokeWeight(2);
    }

    fill(node.color);
    ellipse(pos.x, pos.y, size, size);

    noStroke();
    fill(255);
    textStyle(BOLD);
    textSize((isHover || isActive) ? 14 : 12);
    text(node.label, pos.x, pos.y - 6);

    fill(200);
    textSize((isHover || isActive) ? 12 : 10);
    text(`(ID: ${node.id})`, pos.x, pos.y + 12);
  });
}
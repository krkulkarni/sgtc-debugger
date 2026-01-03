# SGTC Debugger: Structure & Graph Transition Cognition

**SGTC Debugger** is an interactive visualization tool designed to simulate and compare computational models of decision-making and structure learning. It specifically targets hypotheses regarding addiction, modeling how agents learn graph topologies ("cognitive maps") and how specific biases in learning, valuation, or action selection can lead to maladaptive behaviors (e.g., favoring drug-associated nodes).

The application separates the "Brain" (Python/FastAPI) from the "Visuals" (p5.js), allowing for rigorous mathematical modeling alongside real-time visual feedback.

---

## Features

*   **Interactive Graph Simulation:** Visualizes agent movement through various 3x3 graph topologies.
*   **Step-by-Step Execution:** Manually step through three distinct phases:
    1.  **Pre-Exposure:** Passive observation of specific nodes.
    2.  **Exploration:** Active random-walk exploration with time constraints.
    3.  **Decision:** Exploitation phase using the learned matrix.
*   **Real-Time Internals:**
    *   **Transition Matrix ($T$):** Heatmap visualization of the agent's learned connections.
    *   **Probability Vector:** Horizontal heatmap strip showing the softmax probability of the next step.
*   **Model Configuration:** Switch between models and tune free parameters (Learning Rate, Noise, Bias, etc.) on the fly.

---

## Computational Models

This tool implements five distinct agent architectures to test different hypotheses of addiction:

| Model | Hypothesis | Mechanism |
| :--- | :--- | :--- |
| **SimpleLearner** | **Baseline** | Standard Model-Based learning. Uses a Delta Rule to update the transition matrix towards ground truth. Decisions are made via a standard Softmax policy on the learned map. No specific addiction biases. |
| **HabitBiasAgent** | **Action Selection Bias** | Model-free frequency counting. The agent increments connection strengths ($+\text{rate}$) when observed. Transitions are only valid if they exceed a noise `threshold`. Decisions are hijacked by a compulsion parameter ($\beta$), forcing the selection of drug nodes if available. |
| **ValueBiasAgent** | **Reward Magnitude Bias** | Goal-directed but with distorted utility. The agent learns the map objectively ($T$ is correct). However, the intrinsic value ($V$) of drug nodes is artificially inflated by $\beta$. Decisions are made by weighting the map by these inflated values ($T \times V$). |
| **FixedPlanningBiasAgent** | **Attention/Salience Bias** | Model-based structure learning deficit. The agent learns drug-related transitions faster than neutral ones. The learning rate $\alpha$ is boosted by $\beta$ (in logit space) when updating drug nodes, representing "Attentional Capture." |
| **TimeGatedAgent** | **Temporal Encoding Cost** | Time-dependent structure learning. Neutral connections require sustained attention (high duration) to be encoded, while drug connections are encoded instantly. Brief exploration leads to a sparse map for neutral areas but a detailed map for drug areas. |

**Core Mechanisms:**
*   **Global Drift (Wiener Process):** Gaussian noise is injected into the transition matrix at every time step, simulating memory decay and entropy. The agent must explore faster than it forgets.
*   **Inverse Temperature ($\tau$):** Controls the rationality of the Softmax decision policy (Exploration vs. Exploitation).

---

## Tech Stack

*   **Backend:** Python 3.9+, FastAPI, NumPy.
*   **Frontend:** HTML5, CSS3, p5.js (Canvas visualization).
*   **Deployment:** Vercel (Serverless Functions).

---

## Local Installation & Running

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/sgtc-debugger.git
    cd sgtc-debugger
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Development Server:**
    ```bash
    uvicorn app.main:app --reload
    ```

4.  **Open in Browser:**
    Navigate to `http://localhost:8000`.

---

## Deployment (Vercel)

This project is configured for deployment on **Vercel** using Python Serverless Functions.

### Directory Structure
To support the Vercel Python Runtime, the project uses the following structure:

```text
/
├── api/
│   └── index.py          # Entry point for Vercel Serverless
├── app/
│   ├── __init__.py       # App initialization
│   ├── main.py           # FastAPI App definition
│   ├── models.py         # Computational Models
│   ├── simulation.py     # State Management
│   └── scenarios.py      # Graph Data
├── static/
│   ├── index.html        # Frontend Entry
│   ├── sketch.js         # p5.js Logic
│   ├── draw_utils.js     # p5.js Drawing Helpers
│   └── style.css         # Styling
├── requirements.txt      # Python Dependencies
└── vercel.json           # Routing Configuration
# SGTC Debugger: Structure & Graph Transition Cognition

**SGTC Debugger** is an interactive visualization tool designed to simulate and compare computational models of decision-making and structure learning. It specifically targets hypotheses regarding addiction, modeling how agents learn graph topologies ("cognitive maps") and how specific biases in learning or valuation can lead to maladaptive behaviors (e.g., favoring drug-associated nodes).

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
    *   **Probability Vector:** Dynamic bar chart showing the softmax probability of the next step.
*   **Model Configuration:** Switch between models and tune free parameters (Learning Rate, Noise, Bias, etc.) on the fly.

---

## Computational Models

This tool implements four distinct agent architectures to test different hypotheses of addiction:

| Model | Hypothesis | Mechanism |
| :--- | :--- | :--- |
| **HabitLearner** | **Action Selection Bias** | Model-free frequency counting. The agent learns structure quickly ($+1$ per visit). Decisions are hijacked by a compulsion parameter ($\beta$), forcing the selection of drug nodes if available, regardless of long-term planning. |
| **ValuationLearner** | **Reward Magnitude Bias** | Goal-directed but with distorted utility. The agent learns the map objectively ($T$ is correct). However, the intrinsic value ($V$) of drug nodes is artificially inflated by $\beta$. Decisions are made by weighting the map by these inflated values ($T \times V$). |
| **PlanningBiasLearner** | **Attention Bias** | Model-based structure learning deficit. The agent suffers from an "attention deficit" for neutral stimuli. Drug transitions are learned with $\alpha_{max}$, while neutral transitions are penalized ($\alpha - \beta$), resulting in a warped map where drug paths appear more certain. |
| **TimeGatedLearner** | **Temporal Encoding Cost** | Time-dependent structure learning. Neutral connections require sustained attention (high duration) to be encoded, while drug connections are encoded instantly. Brief exploration leads to a sparse map for neutral areas but a detailed map for drug areas. |

**Core Mechanisms:**
*   **Global Drift (Wiener Process):** Gaussian noise is injected into the matrix at every time step, simulating memory decay. The agent must explore faster than it forgets.
*   **Inverse Temperature ($\tau$):** Controls the rationality of the Softmax decision policy.

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
│   └── draw_utils.js     # p5.js Drawing Utilities
│   └── style.css         # p5.js Styling
├── requirements.txt      # Python Dependencies
└── vercel.json           # Routing Configuration
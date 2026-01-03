# app/main.py
from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.scenarios import get_scenario_data, SCENARIOS
from app.simulation import current_simulation, MODEL_REGISTRY
from typing import Dict, Any

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root(): return FileResponse('static/index.html')

@app.get("/api/scenarios_list")
def get_scenarios_list():
    return [{"id": s["id"], "name": s["name"]} for s in SCENARIOS]

# --- MODEL CONFIGURATION ENDPOINTS ---

@app.get("/api/models_list")
def get_models_list():
    """Returns list of available model class names."""
    return list(MODEL_REGISTRY.keys())

@app.get("/api/model_schema/{model_name}")
def get_model_schema(model_name: str):
    """Returns the parameter configuration schema for the UI."""
    cls = MODEL_REGISTRY.get(model_name)
    if cls:
        return cls.get_config_schema()
    return {}

@app.post("/api/init")
def init_simulation(payload: Dict[str, Any] = Body(...)):
    """
    Full Initialization. 
    Payload: { "scenarioId": 1, "modelName": "SimpleLearner", "params": {...} }
    """
    scenario_id = int(payload.get("scenarioId", 1))
    model_name = payload.get("modelName", "SimpleLearner")
    params = payload.get("params", {})
    
    current_simulation.reset(scenario_id, model_name, params)
    
    # Return initial data + scenario data to redraw grid
    scenario_data = get_scenario_data(scenario_id)
    return scenario_data

# ... (Keep existing read_scenario for compatibility if needed, and step endpoints) ...
@app.get("/api/scenario/{scenario_id}")
def read_scenario(scenario_id: int):
    # Fallback legacy endpoint
    current_simulation.reset(scenario_id)
    return get_scenario_data(scenario_id)

@app.get("/api/simulation/state")
def get_simulation_state(): return current_simulation.get_state()

@app.post("/api/simulation/step/reveal")
def step_reveal():
    current_simulation.step_reveal()
    return current_simulation.get_state()

@app.post("/api/simulation/step/explore")
def step_explore():
    current_simulation.step_explore()
    return current_simulation.get_state()

@app.post("/api/simulation/step/decision")
def step_decision():
    current_simulation.step_decision()
    return current_simulation.get_state()
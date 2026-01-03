# app/simulation.py
import random
from app.models import (
    SimpleLearner, 
    HabitBiasAgent,
    ValueBiasAgent,
    FixedPlanningBiasAgent,
    TimeGatedAgent
)
from app.scenarios import get_scenario_data

# Registry of available models
MODEL_REGISTRY = {
    "SimpleLearner": SimpleLearner,
    "HabitBiasAgent": HabitBiasAgent,
    "ValueBiasAgent": ValueBiasAgent,
    "FixedPlanningBiasAgent": FixedPlanningBiasAgent,
    "TimeGatedAgent": TimeGatedAgent
}

class Simulation:
    def __init__(self):
        self.learner = None
        self.scenario_data = None
        
        # Persist settings for resets
        self.current_model_name = "SimpleLearner"
        self.current_params = {}
        
        # State
        self.phase = "IDLE" 
        self.active_node = None 
        self.decision_path = [] 
        self.reveal_index = 0
        self.elapsed_time = 0.0
        self.time_limit = 30.0

    def reset(self, scenario_id, model_name=None, custom_params=None):
        data = get_scenario_data(scenario_id)
        if not data: return False
        
        self.scenario_data = data
        
        # Determine Model Class
        if model_name: self.current_model_name = model_name
        ModelClass = MODEL_REGISTRY.get(self.current_model_name, SimpleLearner)
        
        # Determine Params
        if custom_params is not None: self.current_params = custom_params
        
        # Instantiate
        self.learner = ModelClass(data, self.current_params)
        
        # Reset State
        self.reveal_index = 0
        self.phase = "REVEAL"
        self.active_node = None
        self.elapsed_time = 0.0
        self.decision_path = []
        return True

    def validate_path(self):
        if not self.decision_path or len(self.decision_path) < 2: return True
        valid_edges = set((u, v) for u, v in self.scenario_data['edges'])
        for i in range(len(self.decision_path) - 1):
            if (self.decision_path[i], self.decision_path[i+1]) not in valid_edges: return False
        return True
    def step_reveal(self):
        if self.phase != "REVEAL" or not self.scenario_data: return
        seq = self.scenario_data['revealSequence']
        if self.reveal_index < len(seq):
            node_id = seq[self.reveal_index]
            self.active_node = node_id
            if self.learner: self.learner.update_matrix(node_id, duration=2.0)
            self.reveal_index += 1
        else:
            self.active_node = None
            self.phase = "EXPLORE"
            if self.scenario_data: self.active_node = self.scenario_data['startNode']
    def step_explore(self):
        if self.phase != "EXPLORE" or not self.learner: return
        if self.elapsed_time >= self.time_limit:
            self.phase = "DECISION"
            start_node = self.scenario_data['startNode']
            self.active_node = start_node
            self.decision_path = [start_node] 
            return
        step_duration = random.uniform(1.5, 2.5)
        if self.active_node is not None:
            self.learner.update_matrix(self.active_node, duration=step_duration)
            next_node = self.learner.exploration_policy(self.active_node)
            self.active_node = int(next_node)
            self.elapsed_time += step_duration
            if self.active_node == self.scenario_data['goalNode']:
                self.active_node = self.scenario_data['startNode']
    def step_decision(self):
        if self.phase != "DECISION" or not self.learner: return
        if self.active_node == self.scenario_data['goalNode']:
            self.phase = "FINISHED"
            return
        visited_set = set(self.decision_path)
        next_node = self.learner.decision_policy(self.active_node, visited_set)
        if next_node is not None:
            self.active_node = next_node
            self.decision_path.append(next_node)
        else:
            self.phase = "FINISHED"
    def get_state(self):
        if not self.learner: return None
        total_reveal = len(self.scenario_data['revealSequence']) if self.scenario_data else 0
        path_valid = None
        if self.phase == "FINISHED": path_valid = self.validate_path()
        decision_probs = None
        if self.phase == "DECISION" and self.active_node is not None:
             visited_set = set(self.decision_path)
             decision_probs = self.learner.get_decision_probs(self.active_node, visited_set).tolist()
        return {
            "matrix": self.learner.T.tolist(),
            "active_node": self.active_node,
            "phase": self.phase,
            "reveal_progress": self.reveal_index,
            "reveal_total": total_reveal,
            "elapsed_time": self.elapsed_time,
            "time_limit": self.time_limit,
            "decision_path": self.decision_path,
            "path_valid": path_valid,
            "decision_probs": decision_probs
        }

current_simulation = Simulation()
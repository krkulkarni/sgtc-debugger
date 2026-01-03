# app/models.py
import numpy as np

class SimpleLearner:
    
    # 1. Configuration Schema (Used by Frontend)
    @staticmethod
    def get_config_schema():
        return {
            "alpha_max":   {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "label": "Learning Rate (Alpha)"},
            "epsilon":     {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, "label": "Exploration (Epsilon)"},
            "noise_scale": {"type": "float", "default": 0.01, "min": 0.0, "max": 0.2, "step": 0.001, "label": "Noise Scale (Sigma)"},
            "temperature": {"type": "float", "default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1, "label": "Softmax Temp"},
        }

    def __init__(self, scenario_data, custom_params=None):
        self.scenario = scenario_data
        self.num_nodes = len(scenario_data['nodes'])
        
        # Merge Defaults with Custom Params
        defaults = {k: v['default'] for k, v in self.get_config_schema().items()}
        if custom_params:
            defaults.update(custom_params)
        self.params = defaults

        # Matrices
        self.T = np.zeros((self.num_nodes, self.num_nodes))
        self.R = np.ones(self.num_nodes)
        
        # Adjacency Map
        self.adj = {i: [] for i in range(self.num_nodes)}
        for u, v in scenario_data['edges']:
            if u < self.num_nodes and v < self.num_nodes:
                self.adj[u].append(v)

        # Apply Params to Attributes for easier access
        self.alpha_max = float(self.params['alpha_max'])
        self.epsilon = float(self.params['epsilon'])
        self.noise_scale = float(self.params['noise_scale'])
        self.temperature = float(self.params['temperature'])

    def get_alpha(self, node_id, duration): 
        return self.alpha_max

    def inject_noise(self, duration):
        if duration <= 0: return
        sigma = self.noise_scale * duration 
        noise = np.random.normal(loc=0.0, scale=sigma, size=self.T.shape)
        self.T += noise

    def update_matrix(self, current_node_id, duration=2.0):
        self.inject_noise(duration)
        neighbors = self.adj.get(current_node_id, [])
        target_vector = np.zeros(self.num_nodes)
        for neighbor in neighbors: target_vector[neighbor] = 1.0
        
        current_estimation = self.T[current_node_id, :]
        pe = target_vector - current_estimation
        alpha = self.get_alpha(current_node_id, duration)
        self.T[current_node_id, :] += alpha * pe

    def exploration_policy(self, current_node):
        if np.random.random() < self.epsilon: return np.random.choice(self.num_nodes)
        neighbors = self.adj.get(current_node, [])
        if not neighbors: return np.random.choice(self.num_nodes)
        return np.random.choice(neighbors)

    def softmax(self, x):
        if np.all(np.isneginf(x)): return np.zeros_like(x)
        max_val = np.max(x, where=~np.isneginf(x), initial=-np.inf)
        if np.isneginf(max_val): max_val = 0 
        e_x = np.exp((x - max_val) / self.temperature)
        sum_e_x = np.sum(e_x)
        if sum_e_x == 0: return np.ones_like(x) / len(x)
        return e_x / sum_e_x

    def get_decision_probs(self, current_node, visited):
        logits = self.T[current_node, :].copy()
        for v in visited: logits[v] = -np.inf
        return self.softmax(logits)

    def decision_policy(self, current_node, visited):
        probs = self.get_decision_probs(current_node, visited)
        if np.sum(probs) == 0:
            candidates = [n for n in range(self.num_nodes) if n not in visited]
            if not candidates: return None 
            return np.random.choice(candidates)
        next_node = np.random.choice(range(self.num_nodes), p=probs)
        return int(next_node)

class HabitLearner:
    
    @staticmethod
    def get_config_schema():
        return {
            "beta":    {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "label": "Compulsion (Beta)"},
            "epsilon": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05, "label": "Randomness (Epsilon)"}
        }

    def __init__(self, scenario_data, custom_params=None):
        self.scenario = scenario_data
        self.num_nodes = len(scenario_data['nodes'])
        
        # Merge Params
        defaults = {k: v['default'] for k, v in self.get_config_schema().items()}
        if custom_params:
            defaults.update(custom_params)
        self.params = defaults
        
        self.beta = float(self.params['beta'])
        self.epsilon = float(self.params['epsilon'])
        
        # 1. Transition Matrix (T) - Frequency Counts
        # We start with 0.
        self.T = np.zeros((self.num_nodes, self.num_nodes))
        
        # 2. Drug Encoding Vector
        # One-hot-ish vector: 1 if Drug, 0 if Neutral/Start/Goal
        self.drug_vector = np.zeros(self.num_nodes)
        
        # Build Adjacency (Ground Truth for Exploration)
        self.adj = {i: [] for i in range(self.num_nodes)}
        for u, v in scenario_data['edges']:
            if u < self.num_nodes and v < self.num_nodes:
                self.adj[u].append(v)
                
        # Parse Roles from Scenario Data to populate Drug Vector
        # We look at the 'nodes' list provided by get_scenario_data
        for node in scenario_data['nodes']:
            if node['role'] == 'drug':
                self.drug_vector[node['id']] = 1.0

    def update_matrix(self, current_node_id, duration=2.0):
        """
        Habit Update: Simple frequency counting.
        Increment connection to observed neighbors by 1.
        Ignores duration/learning rate.
        """
        neighbors = self.adj.get(current_node_id, [])
        for neighbor in neighbors:
            self.T[current_node_id, neighbor] += 1.0

    def exploration_policy(self, current_node):
        """
        Standard Epsilon-Greedy for Exploration Phase.
        Same as SimpleLearner.
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_nodes)
        
        neighbors = self.adj.get(current_node, [])
        if not neighbors:
            return np.random.choice(self.num_nodes)
            
        return np.random.choice(neighbors)

    def get_decision_probs(self, current_node, visited):
        """
        Habit Decision Logic:
        1. Epsilon chance to pick ANY unvisited node.
        2. (1-Epsilon) chance to pick from neighbors in T > 0.
           - Split prob between Drug and Neutral neighbors based on Beta.
        """
        probs = np.zeros(self.num_nodes)
        
        # 1. Identify Candidates (Unvisited)
        candidates = [n for n in range(self.num_nodes) if n not in visited]
        if not candidates:
            return probs # All zeros (trapped)

        # 2. Identify Valid Neighbors (Learned structure T > 0)
        # We only care if T > 0 (Binary check)
        learned_row = self.T[current_node, :]
        valid_neighbors = [c for c in candidates if learned_row[c] > 0]
        
        # --- CASE A: No Learned Neighbors ---
        # If we haven't learned any paths from here, pure random walk among unvisited
        if not valid_neighbors:
            count = len(candidates)
            for c in candidates:
                probs[c] = 1.0 / count
            return probs

        # --- CASE B: Has Learned Neighbors ---
        
        # A. Teleportation Mass (Epsilon) distributed across ALL candidates
        # (This represents the "Random unvisited node" logic)
        teleport_prob_total = self.epsilon
        teleport_prob_per_node = teleport_prob_total / len(candidates)
        for c in candidates:
            probs[c] += teleport_prob_per_node
            
        # B. Bias Mass (1 - Epsilon) distributed across Valid Neighbors
        bias_mass_total = 1.0 - self.epsilon
        
        # Categorize Neighbors
        drug_neighbors = []
        neutral_neighbors = []
        
        for n in valid_neighbors:
            if self.drug_vector[n] == 1.0:
                drug_neighbors.append(n)
            else:
                neutral_neighbors.append(n)
        
        n_drug = len(drug_neighbors)
        n_neutral = len(neutral_neighbors)
        
        # Distribute Bias Mass
        if n_drug > 0 and n_neutral > 0:
            # Both exist: Split by Beta
            # Drug Group gets: bias_mass * Beta
            # Neutral Group gets: bias_mass * (1-Beta)
            
            p_per_drug = (bias_mass_total * self.beta) / n_drug
            p_per_neutral = (bias_mass_total * (1.0 - self.beta)) / n_neutral
            
            for d in drug_neighbors: probs[d] += p_per_drug
            for n in neutral_neighbors: probs[n] += p_per_neutral
            
        elif n_drug > 0:
            # Only Drugs exist: They get all the bias mass
            p_per_drug = bias_mass_total / n_drug
            for d in drug_neighbors: probs[d] += p_per_drug
            
        elif n_neutral > 0:
            # Only Neutrals exist: They get all the bias mass
            p_per_neutral = bias_mass_total / n_neutral
            for n in neutral_neighbors: probs[n] += p_per_neutral
            
        return probs

    def decision_policy(self, current_node, visited):
        probs = self.get_decision_probs(current_node, visited)
        
        # Safety check for sum=0
        if np.sum(probs) == 0:
            candidates = [n for n in range(self.num_nodes) if n not in visited]
            if not candidates: return None
            return np.random.choice(candidates)
            
        # Normalize just in case of float errors (though logic ensures sum=1)
        probs = probs / np.sum(probs)
        
        next_node = np.random.choice(range(self.num_nodes), p=probs)
        return int(next_node)

class ValuationLearner:
    
    @staticmethod
    def get_config_schema():
        return {
            "alpha_max":   {"type": "float", "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05, "label": "Learning Rate (Alpha)"},
            "beta":        {"type": "float", "default": 2.0, "min": 1.0, "max": 10.0, "step": 0.5, "label": "Drug Value (Beta)"},
            "temperature": {"type": "float", "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1, "label": "Softmax Temp (Tau)"},
            "noise_scale": {"type": "float", "default": 0.01, "min": 0.0, "max": 0.2, "step": 0.001, "label": "Noise Scale"},
            "epsilon":     {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, "label": "Exploration Eps"}
        }

    def __init__(self, scenario_data, custom_params=None):
        self.scenario = scenario_data
        self.num_nodes = len(scenario_data['nodes'])
        
        # Merge Params
        defaults = {k: v['default'] for k, v in self.get_config_schema().items()}
        if custom_params: defaults.update(custom_params)
        self.params = defaults
        
        # Unpack for easy access
        self.alpha_max = float(self.params['alpha_max'])
        self.beta = float(self.params['beta'])
        self.temperature = float(self.params['temperature'])
        self.noise_scale = float(self.params['noise_scale'])
        self.epsilon = float(self.params['epsilon'])

        # 1. Transition Matrix (T) - Learned Map
        self.T = np.zeros((self.num_nodes, self.num_nodes))
        
        # 2. Value Vector (V) - Static Incentive Salience
        # Default all to 1.0 (Neutral)
        self.V = np.ones(self.num_nodes)
        
        # Populate Drug Values based on Role
        for node in scenario_data['nodes']:
            if node['role'] == 'drug':
                self.V[node['id']] = self.beta
                
        # Adjacency for Learning
        self.adj = {i: [] for i in range(self.num_nodes)}
        for u, v in scenario_data['edges']:
            if u < self.num_nodes and v < self.num_nodes:
                self.adj[u].append(v)

    # --- Learning (Identical to SimpleLearner) ---
    def get_alpha(self, node_id, duration): 
        return self.alpha_max

    def inject_noise(self, duration):
        if duration <= 0: return
        sigma = self.noise_scale * duration 
        noise = np.random.normal(loc=0.0, scale=sigma, size=self.T.shape)
        self.T += noise

    def update_matrix(self, current_node_id, duration=2.0):
        # 1. Global Drift
        self.inject_noise(duration)
        
        # 2. Target = All Neighbors
        neighbors = self.adj.get(current_node_id, [])
        target_vector = np.zeros(self.num_nodes)
        for neighbor in neighbors: target_vector[neighbor] = 1.0
        
        # 3. Update
        current_estimation = self.T[current_node_id, :]
        pe = target_vector - current_estimation
        alpha = self.get_alpha(current_node_id, duration)
        self.T[current_node_id, :] += alpha * pe

    def exploration_policy(self, current_node):
        # Standard Epsilon-Greedy during exploration
        if np.random.random() < self.epsilon: return np.random.choice(self.num_nodes)
        neighbors = self.adj.get(current_node, [])
        if not neighbors: return np.random.choice(self.num_nodes)
        return np.random.choice(neighbors)

    # --- Decision Policy (The Key Difference) ---
    def softmax(self, x):
        if np.all(np.isneginf(x)): return np.zeros_like(x)
        max_val = np.max(x, where=~np.isneginf(x), initial=-np.inf)
        if np.isneginf(max_val): max_val = 0 
        e_x = np.exp((x - max_val) / self.temperature)
        sum_e_x = np.sum(e_x)
        if sum_e_x == 0: return np.ones_like(x) / len(x)
        return e_x / sum_e_x

    def get_decision_probs(self, current_node, visited):
        """
        Decision Score = TransitionStrength * NodeValue
        Logits = T[current] * V
        """
        # 1. Get Learned Transitions (The Map)
        transitions = self.T[current_node, :].copy()
        
        # 2. Weight by Value (The Incentive)
        # Element-wise multiplication: 
        # If T=0.8 and V=1 (Neutral) -> Score 0.8
        # If T=0.8 and V=10 (Drug)    -> Score 8.0
        weighted_scores = transitions * self.V
        
        # 3. Mask Visited (Set to -inf)
        for v in visited:
            weighted_scores[v] = -np.inf
            
        # 4. Softmax
        return self.softmax(weighted_scores)

    def decision_policy(self, current_node, visited):
        probs = self.get_decision_probs(current_node, visited)
        
        if np.sum(probs) == 0:
            candidates = [n for n in range(self.num_nodes) if n not in visited]
            if not candidates: return None 
            return np.random.choice(candidates)
            
        next_node = np.random.choice(range(self.num_nodes), p=probs)
        return int(next_node)

class PlanningBiasLearner:
    
    @staticmethod
    def get_config_schema():
        return {
            # alpha_max: Base learning rate (Probability space 0-1)
            "alpha_max":   {"type": "float", "default": 0.8, "min": 0.01, "max": 0.99, "step": 0.01, "label": "Max Alpha (Prob)"},
            # beta: Bias penalty (Logit space). Range [-5, 5] allows shifting sigmoid effectively.
            "beta":        {"type": "float", "default": 2.0, "min": -5.0, "max": 5.0, "step": 0.5, "label": "Bias Penalty (Logit)"},
            
            # Standard params
            "noise_scale": {"type": "float", "default": 0.01, "min": 0.0, "max": 0.2, "step": 0.001, "label": "Noise Scale"},
            "temperature": {"type": "float", "default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1, "label": "Softmax Temp"},
            "epsilon":     {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, "label": "Exploration Eps"}
        }

    def __init__(self, scenario_data, custom_params=None):
        self.scenario = scenario_data
        self.num_nodes = len(scenario_data['nodes'])
        
        defaults = {k: v['default'] for k, v in self.get_config_schema().items()}
        if custom_params: defaults.update(custom_params)
        self.params = defaults
        
        self.alpha_max = float(self.params['alpha_max'])
        self.beta = float(self.params['beta'])
        self.noise_scale = float(self.params['noise_scale'])
        self.temperature = float(self.params['temperature'])
        self.epsilon = float(self.params['epsilon'])

        self.T = np.zeros((self.num_nodes, self.num_nodes))
        self.R = np.ones(self.num_nodes)
        
        # Identify Drug Nodes
        self.drug_vector = np.zeros(self.num_nodes)
        for node in scenario_data['nodes']:
            if node['role'] == 'drug':
                self.drug_vector[node['id']] = 1.0
                
        self.adj = {i: [] for i in range(self.num_nodes)}
        for u, v in scenario_data['edges']:
            if u < self.num_nodes and v < self.num_nodes:
                self.adj[u].append(v)

    # --- Math Helpers ---
    def sigmoid(self, x):
        """ Logit Space -> Probability Space """
        return 1 / (1 + np.exp(-x))

    def logit(self, p):
        """ Probability Space -> Logit Space """
        # Clip to prevent inf/nan
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.log(p / (1 - p))

    # --- Update Logic ---
    def inject_noise(self, duration):
        if duration <= 0: return
        sigma = self.noise_scale * duration 
        noise = np.random.normal(loc=0.0, scale=sigma, size=self.T.shape)
        self.T += noise

    def update_matrix(self, current_node_id, duration=2.0):
        # 1. Global Drift
        self.inject_noise(duration)
        
        # 2. Get Neighbors & Target
        neighbors = self.adj.get(current_node_id, [])
        target_vector = np.zeros(self.num_nodes)
        for neighbor in neighbors: target_vector[neighbor] = 1.0
        
        # 3. Calculate Prediction Error
        current_estimation = self.T[current_node_id, :]
        pe = target_vector - current_estimation

        # 4. Calculate Vectorized Alpha
        # 4a. Convert base alpha to Logit Score
        base_score = self.logit(self.alpha_max)
        
        # 4b. Create Alpha Vector
        alpha_vector = np.zeros(self.num_nodes)
        
        for i in range(self.num_nodes):
            # Start with base score
            score = base_score
            
            # Apply Penalty if NOT a drug node (Neutral)
            # Logic: Drug nodes get full attention (base_score).
            # Neutral nodes get penalized (base_score - beta).
            if self.drug_vector[i] == 0.0:
                score -= self.beta
            
            # Convert back to Probability Space
            alpha_vector[i] = self.sigmoid(score)

        # 5. Apply Vectorized Update
        # Element-wise multiplication: Each column 'i' is updated with its specific alpha[i]
        self.T[current_node_id, :] += alpha_vector * pe

    # --- Policies (Standard) ---
    def exploration_policy(self, current_node):
        if np.random.random() < self.epsilon: return np.random.choice(self.num_nodes)
        neighbors = self.adj.get(current_node, [])
        if not neighbors: return np.random.choice(self.num_nodes)
        return np.random.choice(neighbors)

    def softmax(self, x):
        if np.all(np.isneginf(x)): return np.zeros_like(x)
        max_val = np.max(x, where=~np.isneginf(x), initial=-np.inf)
        if np.isneginf(max_val): max_val = 0 
        e_x = np.exp((x - max_val) / self.temperature)
        sum_e_x = np.sum(e_x)
        if sum_e_x == 0: return np.ones_like(x) / len(x)
        return e_x / sum_e_x

    def get_decision_probs(self, current_node, visited):
        # Standard Softmax on T (The bias is encoded IN the matrix T itself)
        logits = self.T[current_node, :].copy()
        for v in visited: logits[v] = -np.inf
        return self.softmax(logits)

    def decision_policy(self, current_node, visited):
        probs = self.get_decision_probs(current_node, visited)
        if np.sum(probs) == 0:
            candidates = [n for n in range(self.num_nodes) if n not in visited]
            if not candidates: return None 
            return np.random.choice(candidates)
        next_node = np.random.choice(range(self.num_nodes), p=probs)
        return int(next_node)

class TimeGatedLearner:
    
    @staticmethod
    def get_config_schema():
        return {
            # Learning Parameters
            "alpha_max":   {"type": "float", "default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05, "label": "Max Alpha"},
            
            # Time-Gating Parameters
            "beta0":       {"type": "float", "default": 3.0, "min": 0.5, "max": 10.0, "step": 0.5, "label": "Neutral Threshold (s)"},
            "beta1":       {"type": "float", "default": 2.5, "min": 0.0, "max": 10.0, "step": 0.5, "label": "Drug Advantage (s)"},
            "k":           {"type": "float", "default": 5.0, "min": 1.0, "max": 20.0, "step": 1.0, "label": "Gating Sharpness (k)"},
            
            # Standard Parameters
            "noise_scale": {"type": "float", "default": 0.01, "min": 0.0, "max": 0.2, "step": 0.001, "label": "Noise Scale"},
            "temperature": {"type": "float", "default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1, "label": "Softmax Temp"},
            "epsilon":     {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, "label": "Exploration Eps"}
        }

    def __init__(self, scenario_data, custom_params=None):
        self.scenario = scenario_data
        self.num_nodes = len(scenario_data['nodes'])
        
        defaults = {k: v['default'] for k, v in self.get_config_schema().items()}
        if custom_params: defaults.update(custom_params)
        self.params = defaults
        
        # Unpack params
        self.alpha_max = float(self.params['alpha_max'])
        self.beta0 = float(self.params['beta0'])
        self.beta1 = float(self.params['beta1'])
        self.k = float(self.params['k'])
        self.noise_scale = float(self.params['noise_scale'])
        self.temperature = float(self.params['temperature'])
        self.epsilon = float(self.params['epsilon'])

        self.T = np.zeros((self.num_nodes, self.num_nodes))
        self.R = np.ones(self.num_nodes)
        
        # Identify Drug Nodes (1.0 for Drug, 0.0 for Neutral)
        self.drug_vector = np.zeros(self.num_nodes)
        for node in scenario_data['nodes']:
            if node['role'] == 'drug':
                self.drug_vector[node['id']] = 1.0
                
        self.adj = {i: [] for i in range(self.num_nodes)}
        for u, v in scenario_data['edges']:
            if u < self.num_nodes and v < self.num_nodes:
                self.adj[u].append(v)

    # --- Update Logic ---
    def inject_noise(self, duration):
        if duration <= 0: return
        sigma = self.noise_scale * duration 
        noise = np.random.normal(loc=0.0, scale=sigma, size=self.T.shape)
        self.T += noise

    def update_matrix(self, current_node_id, duration=2.0):
        # 1. Global Drift
        self.inject_noise(duration)
        
        # 2. Get Neighbors & Target
        neighbors = self.adj.get(current_node_id, [])
        target_vector = np.zeros(self.num_nodes)
        for neighbor in neighbors: target_vector[neighbor] = 1.0
        
        # 3. Calculate Prediction Error
        current_estimation = self.T[current_node_id, :]
        pe = target_vector - current_estimation

        # 4. Calculate Vectorized Alpha (Time-Gated Logic)
        
        # 4a. Calculate Thresholds for all nodes
        # theta = beta0 - (beta1 * is_drug)
        # Result: Neutrals get beta0, Drugs get (beta0 - beta1)
        theta_vector = self.beta0 - (self.beta1 * self.drug_vector)
        
        # 4b. Calculate Logistic Scaling factor based on duration
        # x = k * (duration - theta)
        # Using clip to prevent overflow in exp
        x = self.k * (duration - theta_vector)
        x = np.clip(x, -500, 500) 
        
        logistic_factor = 1.0 / (1.0 + np.exp(-x))
        
        # 4c. Final Alpha Vector
        alpha_vector = self.alpha_max * logistic_factor

        # 5. Apply Vectorized Update
        self.T[current_node_id, :] += alpha_vector * pe

    # --- Policies (Standard) ---
    def exploration_policy(self, current_node):
        if np.random.random() < self.epsilon: return np.random.choice(self.num_nodes)
        neighbors = self.adj.get(current_node, [])
        if not neighbors: return np.random.choice(self.num_nodes)
        return np.random.choice(neighbors)

    def softmax(self, x):
        if np.all(np.isneginf(x)): return np.zeros_like(x)
        max_val = np.max(x, where=~np.isneginf(x), initial=-np.inf)
        if np.isneginf(max_val): max_val = 0 
        e_x = np.exp((x - max_val) / self.temperature)
        sum_e_x = np.sum(e_x)
        if sum_e_x == 0: return np.ones_like(x) / len(x)
        return e_x / sum_e_x

    def get_decision_probs(self, current_node, visited):
        logits = self.T[current_node, :].copy()
        for v in visited: logits[v] = -np.inf
        return self.softmax(logits)

    def decision_policy(self, current_node, visited):
        probs = self.get_decision_probs(current_node, visited)
        if np.sum(probs) == 0:
            candidates = [n for n in range(self.num_nodes) if n not in visited]
            if not candidates: return None 
            return np.random.choice(candidates)
        next_node = np.random.choice(range(self.num_nodes), p=probs)
        return int(next_node)
# app/scenarios.py

# --- Role Definitions & Metadata ---
NODE_METADATA = {
    # ROLE: start
    "HOME":      {"role": "start",   "label": "HOME", "color": "#3498db"},
    
    # ROLE: goal
    "WORK":      {"role": "goal",    "label": "WORK", "color": "#2ecc71"},
    
    # ROLE: drug
    "The Bar":   {"role": "drug",    "label": "The Bar", "color": "#e74c3c"},
    "Liquor":    {"role": "drug",    "label": "Liquor", "color": "#e74c3c"},
    "Club":      {"role": "drug",    "label": "Club", "color": "#e74c3c"},
    "Pub":       {"role": "drug",    "label": "Pub", "color": "#e74c3c"},
    
    # ROLE: neutral
    "Laundry":   {"role": "neutral", "label": "Laundry", "color": "#95a5a6"},
    "Gym":       {"role": "neutral", "label": "Gym", "color": "#95a5a6"},
    "Grocery":   {"role": "neutral", "label": "Grocery", "color": "#95a5a6"},
    "Park":      {"role": "neutral", "label": "Park", "color": "#95a5a6"},
    "Netflix":   {"role": "neutral", "label": "Netflix", "color": "#95a5a6"},
    "Library":   {"role": "neutral", "label": "Library", "color": "#95a5a6"},
    "Cafe":      {"role": "neutral", "label": "Cafe", "color": "#95a5a6"},
    "Bank":      {"role": "neutral", "label": "Bank", "color": "#95a5a6"}
}

SCENARIOS = [
  {
    "id": 1,
    "name": "The Triple Lane",
    "startNode": 0,
    "goalNode": 8,
    "revealSequence": [4, 2, 7, 0, 5, 3, 6, 1, 8],
    "assignments": {
      0: "HOME", 8: "WORK",
      4: "The Bar", 6: "Liquor",
      1: "Laundry", 2: "Gym", 3: "Grocery", 5: "Park", 7: "Netflix"
    },
    "edges": [
      [0, 4], [0, 1], [0, 2],
      [4, 6], [4, 5], [4, 3],
      [1, 3], [1, 7], [1, 5],
      [2, 5], [2, 7], [2, 6],
      [6, 8], [6, 7], [6, 3],
      [3, 8], [3, 5], [3, 2],
      [5, 8], [5, 7], [5, 1],
      [7, 8], [7, 4], [7, 2]
    ]
  },
  {
    "id": 2,
    "name": "Diagonal Drift",
    "startNode": 6,
    "goalNode": 2,
    "revealSequence": [2, 5, 8, 1, 4, 7, 0, 3, 6],
    "assignments": {
      6: "HOME", 2: "WORK",
      3: "The Bar", 1: "Liquor", 
      0: "Laundry", 4: "Gym", 5: "Grocery", 7: "Park", 8: "Netflix"
    },
    "edges": [
      [6, 3], [6, 7], [6, 0],
      [3, 1], [3, 4], [3, 8],
      [7, 4], [7, 8], [7, 5],
      [0, 5], [0, 4], [0, 8],
      [1, 2], [1, 5], [1, 7],
      [4, 2], [4, 1], [4, 0],
      [5, 2], [5, 8], [5, 7],
      [8, 2], [8, 1], [8, 3]
    ]
  },
  # ... Add other scenarios here
]

def get_scenario_data(scenario_id):
    """
    Returns a cleaner JSON structure for the frontend with calculated grid positions.
    """
    # Find scenario by ID
    data = next((s for s in SCENARIOS if s["id"] == scenario_id), None)
    if not data:
        return None

    nodes = []
    # Fixed 3x3 Grid Logic (Rows 0, 1, 2)
    # 0 1 2
    # 3 4 5
    # 6 7 8
    for i in range(9):
        # Determine Role/Label
        assignment_key = data["assignments"].get(i, "Park") # Default to Park
        # Lookup Metadata or fallback
        meta = NODE_METADATA.get(assignment_key, {"role": "neutral", "label": str(assignment_key), "color": "#95a5a6"})
        
        # Calculate Grid Position (0-2)
        grid_x = i % 3
        grid_y = i // 3
        
        nodes.append({
            "id": i,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "label": meta["label"],
            "role": meta["role"],
            "color": meta["color"]
        })

    return {
        "id": data["id"],
        "name": data["name"],
        "nodes": nodes,
        "edges": data["edges"],
        "revealSequence": data["revealSequence"],
        "startNode": data["startNode"],
        "goalNode": data["goalNode"]
    }
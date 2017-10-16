import json


def compute_numeric_solution(shapes):

    json_data = {
        "solver": "numeric",
        "shape_initial_values": [],
        "shape_ode_definitions": [],
        "shape_state_variables": [],
    }

    for shape in shapes:
        json_data["shape_initial_values"].extend([str(x) for x in shape.initial_values])
        json_data["shape_ode_definitions"].append(str(shape.ode_definition))
        json_data["shape_state_variables"].extend([str(x) for x in shape.state_variables])
        
    return json.dumps(json_data, indent=2)

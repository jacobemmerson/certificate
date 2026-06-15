import json
import math
import os

from dag import harm_tree, provision_tree


def _clean(value):
    """Recursively convert NaN/blank strings to None so the result is valid JSON."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def build_dashboard_data():
    return _clean({
        "harm_tree": harm_tree(),
        "provision_tree": provision_tree(),
    })


if __name__ == "__main__":
    data = build_dashboard_data()
    out_path = os.path.join(os.path.dirname(__file__), "..", "dashboard_data.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {out_path}")

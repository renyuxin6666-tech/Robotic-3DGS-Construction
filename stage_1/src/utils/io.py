import yaml
import json

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
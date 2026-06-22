import json, itertools, subprocess, copy
from pathlib import Path

BASE_CONFIG = "./templates/config_default.json"

SWEEP = {
    "train_config": {
        "lr_max": [3e-4, 3e-5, 3e-6],
        "lr_schedule_T_w": [512, 1024, 2048],
    },
    "model_config": {
        "num_layers": [4],
    },
    "dl_config": {"batch_size": [48]},
}


def flatten_sweep(sweep):
    keys, values = [], []
    for section, params in sweep.items():
        for k, vs in params.items():
            keys.append((section, k))
            values.append(vs)
    return keys, values


with open(BASE_CONFIG) as f:
    base = json.load(f)

keys, values = flatten_sweep(SWEEP)

# Generates a grid of param combinations
for combo in itertools.product(*values):
    cfg = copy.deepcopy(base)
    for (section, key), val in zip(keys, combo):
        cfg[section][key] = val

    cfg_path = Path("./experiments") / f"sweep_{'_'.join(str(v) for v in combo)}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # subprocess.run(["python", "train", str(cfg_path)])
    # subprocess.run(["python", "-m", "cs336_basics.train", str(cfg_path)])

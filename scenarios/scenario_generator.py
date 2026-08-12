import yaml
from pathlib import Path
from copy import deepcopy


import shutil


PROTECTION_PARAMS = {
    "RAND": [
        (2, 2), (3, 2), (4, 2),
        (2, 4), (3, 4), (4, 4),
        (2, 6), (3, 6), (4, 6),
        (2, 8), (3, 8), (4, 8),
    ],
    "WC": [
        (5, 2), (6, 2), (7, 2),
        (5, 4), (6, 4), (7, 4),
        (5, 6), (6, 6), (7, 6),
        (5, 8), (6, 8), (7, 8),
    ],
    "MAX_ELEV": [
        (1, 2), (2, 2), (3, 2),
        (1, 4), (2, 4), (3, 4),
        (1, 6), (2, 6), (3, 6),
        (1, 8), (2, 8), (3, 8),
    ],
}


def clear_directory_pathlib(dir_path):
    path = Path(dir_path)

    # Iterate through all items inside the directory
    for item in path.iterdir():
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)  # Remove subdirectories and their contents
        else:
            item.unlink()         # Remove files or symbolic links


def tuple_constructor(loader, node):
    """Load the sequence of values from the YAML node and returns a tuple constructed from the sequence."""
    values = loader.construct_sequence(node)
    return tuple(values)


yaml.SafeLoader.add_constructor(
    'tag:yaml.org,2002:python/tuple',
    tuple_constructor)

BASE_DIR = Path(__file__).parent.resolve()
INPUTS_DIR = BASE_DIR / "input/"


def main():
    with open(BASE_DIR / "scenario_base.yaml", "r") as f:
        raw_data = yaml.safe_load(f)
    added_loss = 3.  # db
    gso_links = []
    raw_data["ngso2gso"]["gso_links"] = gso_links

    raw_data["ngso2gso"]["ngso"]["tx_model"]["pfd_at_ref_bandwidth_dBW_m2"]

    for link_key, link in raw_data["gso_links_helpers"]["links"].items():
        for place_key, place in raw_data["gso_links_helpers"]["places"].items():
            gso_links.append({**place, **link, "added_loss": added_loss})

    to_create = []
    for selection_strategy in ["RAND", "WC", "MAX_ELEV"]:
        for arc_avoid, n_co in PROTECTION_PARAMS[selection_strategy]:
            scenario_name = f"{selection_strategy.lower()}_{arc_avoid}arc_avoid_{n_co}Nco"
            d = deepcopy(raw_data)
            d["ngso2gso"]["scenario_name"] = scenario_name
            d["ngso2gso"]["selection_strategy"] = selection_strategy
            d["ngso2gso"]["ngso"]["gso_protection_avoidance_angle"] = arc_avoid
            d["ngso2gso"]["ngso"]["n_co_channel"] = n_co
            to_create.append((d, scenario_name + ".yaml"))

    if INPUTS_DIR.exists():
        print(f"DELETING EXISTING FILES in {INPUTS_DIR}")
        clear_directory_pathlib(INPUTS_DIR)

    print(f"Creating {len(to_create)} files at {INPUTS_DIR}")
    INPUTS_DIR.mkdir(exist_ok=True)
    for par in to_create:
        with open(INPUTS_DIR / par[1], "w") as f:
            yaml.dump(par[0], f, sort_keys=False)


if __name__ == "__main__":
    main()

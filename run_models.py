"""Open this file in PyCharm and Run. Edit CONFIG_PATH to choose a saved comparison."""
from pathlib import Path
from atmosphere_compare.runner import run

CONFIG_PATH = Path(__file__).parent / "configs" / "real_models.json"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    run(args.config)

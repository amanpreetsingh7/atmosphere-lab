"""Run this file in PyCharm. Launches Streamlit using the same Python interpreter."""
from pathlib import Path
import importlib.util
import subprocess
import sys
import shlex

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    missing = [name for name in ("streamlit", "plotly", "pandas", "numpy", "matplotlib", "scipy", "astropy", "netCDF4") if importlib.util.find_spec(name) is None]
    if missing:
        print("Missing packages:", ", ".join(missing))
        print("Install requirements.txt in your PyCharm interpreter, then run this file again.")
        print("Terminal command:", shlex.join([sys.executable,"-m","pip","install","-r",str(root/"requirements.txt")]))
        raise SystemExit(1)
    print("Opening Atmosphere Lab. If no browser opens, visit http://localhost:8501")
    print("Stop the PyCharm run to close the local server.")
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(root / "app.py"),
                                    "--server.address=127.0.0.1", "--server.port=8501"], cwd=root))

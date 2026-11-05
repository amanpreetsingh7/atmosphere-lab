"""Run in PyCharm to install dependencies into this exact Python interpreter."""
from pathlib import Path
import subprocess
import sys

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    print('Installing Atmosphere Lab requirements into:',sys.executable,flush=True)
    code=subprocess.call([sys.executable,'-m','pip','install','-r',str(root/'requirements.txt')])
    if code==0:print('READY. Run launch_app.py in this project.',flush=True)
    else:print('Installation did not finish. Keep this output so the dependency error can be checked.',flush=True)
    raise SystemExit(code)

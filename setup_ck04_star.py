"""Background installer for the native CK04 node requested in the app."""
from pathlib import Path
import argparse
from atmosphere_compare.ck04_install import install_ck04

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--teff_K', type=float, required=True)
    parser.add_argument('--logg_cgs', type=float, required=True)
    parser.add_argument('--metallicity_dex', type=float, required=True)
    args = parser.parse_args()
    try:install_ck04(Path(__file__).resolve().parent, **vars(args))
    except Exception as exc:
        print('CK04 INSTALL STOPPED:', exc, flush=True)
        raise SystemExit(1)

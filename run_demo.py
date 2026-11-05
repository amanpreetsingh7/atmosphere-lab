"""Software demonstration only: analytic blackbodies, NOT atmosphere models.

Run directly in PyCharm. Does not need astropy or any model downloads.
Writes to demo_input/ and a fresh result directory; never edits real models.
"""
from pathlib import Path
import json
import numpy as np
from atmosphere_compare.runner import run


def surface_blackbody(wavelength_um, temperature):
    # pi * Planck spectral radiance; output W/m^2/micron.
    h, c, k = 6.62607015e-34, 299792458.0, 1.380649e-23
    m = wavelength_um * 1e-6
    return np.pi * 2*h*c*c / m**5 / np.expm1(h*c/(m*k*temperature)) * 1e-6


def prepare_demo(root):
    root.mkdir(parents=True, exist_ok=True)
    wave = np.geomspace(.3, 30, 30000)
    for name, temperature in (("star", 5000), ("planet", 1000)):
        np.savetxt(root / (name+".csv"), np.column_stack((wave, surface_blackbody(wave, temperature))),
                   delimiter=",", header="wavelength_um,surface_flambda_W_m2_um")
    # Deliberately named toy filter, not an approximation claimed to be J, L', or JWST.
    np.savetxt(root / "toy_filter.csv", [[3, 0], [3.3, 1], [3.8, 1], [4.1, 0]], delimiter=",")
    config = {"title": "SOFTWARE DEMO: blackbodies, not atmosphere grids", "bin_resolving_power": 100,
              "wavelength_range_um": [.5, 20], "distance_pc": 10, "output_directory": "../results",
              "filters": [{"name": "Toy 3.0-4.1 micron response (not a real filter)", "file": "toy_filter.csv",
                           "delimiter": ",", "wavelength_frame": "vacuum", "detector": "photon"}]}
    for name, temperature, unit in (("star", 5000, "R_sun"), ("planet", 1000, "R_jup")):
        config[name] = {"label": "%s: %d K blackbody demonstration" % (name, temperature),
                        "radius": {"value": 1.0, "unit": unit},
                        "spectrum": {"format": "ascii", "file": name+".csv", "delimiter": ",",
                                     "wavelength_unit": "um", "flux_unit": "W_m2_um", "flux_convention": "surface_flux",
                                     "wavelength_frame": "vacuum", "parameters": {"teff_K": temperature},
                                     "label": "DEMO blackbody"}}
    p = root / "demo.json"
    p.write_text(json.dumps(config, indent=2))
    return p


if __name__ == "__main__":
    print("This tests the software flow; it is NOT a scientific atmosphere comparison.")
    run(prepare_demo(Path(__file__).parent / "demo_input"))

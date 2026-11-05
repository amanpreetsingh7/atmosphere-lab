import copy
import json
import io
import unittest
import zipfile
from atmosphere_compare.workbench import blackbody_demo, snapshot, export_bundle


class WorkbenchChecks(unittest.TestCase):
    def test_export_snapshot_keeps_mode_and_config(self):
        star, planet = blackbody_demo(5000, "Star"), blackbody_demo(1000, "Planet")
        item = snapshot(star, planet, 6.957e8, 7.1492e7, 10, 20, [.8, 5], "Demonstration", {"mode": "Demonstration"})
        other = copy.deepcopy(item)
        other["label"] = "Second demo"
        with zipfile.ZipFile(io.BytesIO(export_bundle(item, [other]))) as z:
            report = json.loads(z.read("report.json"))
            self.assertEqual(report["mode"], "Demonstration")
            self.assertEqual(report["star"]["kind"], "blackbody_demonstration")
            self.assertIn("saved_comparisons.png", z.namelist())
            self.assertTrue(z.read("spectrum_comparison.pdf").startswith(b"%PDF"))

    def test_native_phoenix_reader_on_synthetic_fits(self):
        # Small synthetic FITS fixture, not a genuine atmosphere-grid spectrum.
        import tempfile
        from pathlib import Path
        import numpy as np
        from astropy.io import fits
        from atmosphere_compare.spectra import load_spectrum
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            h = fits.Header()
            h["PHXTEFF"], h["PHXLOGG"], h["PHXM_H"] = 5000, 4.5, 0
            h["BUNIT"] = "erg/s/cm^2/cm"
            fits.writeto(path/"flux.fits", np.array([1e7, 2e7, 3e7]), h)
            fits.writeto(path/"wave.fits", np.array([10000, 20000, 30000]), fits.Header({"UNIT":"Angstrom"}))
            settings = {"format": "phoenix_hires", "file": "flux.fits", "wavelength_file": "wave.fits",
                        "parameters": {"teff_K": 5000, "logg_cgs": 4.5}}
            result = load_spectrum(settings, path)
            np.testing.assert_allclose(result.wavelength_um, [1, 2, 3])
            np.testing.assert_allclose(result.surface_flux, [1, 2, 3])
            settings["parameters"]["teff_K"] = 6000
            with self.assertRaises(ValueError):
                load_spectrum(settings, path)

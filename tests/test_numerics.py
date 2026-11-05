"""Analytic software checks; no observed data or atmosphere-grid experiment."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from atmosphere_compare.spectra import Spectrum, to_surface_flambda, select_from_catalog, load_spectrum
from atmosphere_compare.compare import compare, band_signal, band_contrast


class NumericalChecks(unittest.TestCase):
    def test_fnu_matches_flambda_conversion(self):
        wave = np.geomspace(.5, 10, 100)
        flambda = np.linspace(1, 3, 100)
        fnu_cgs = flambda/(299792458/(wave*1e-6)**2*1e-6)/1e-3
        _, actual = to_surface_flambda(wave, fnu_cgs, "um", "erg_cm2_s_Hz")
        np.testing.assert_allclose(actual, flambda, rtol=1e-14)

    def test_cgs_wavelength_units(self):
        for unit, factor in (("erg_cm2_s_A", 10), ("erg_cm2_s_cm", 1e-7), ("W_m2_m", 1e-6)):
            _, flux = to_surface_flambda([10000, 20000, 30000], [1, 2, 3], "angstrom", unit)
            np.testing.assert_allclose(flux, np.array([1, 2, 3])*factor)

    def test_surface_and_eddington_flux(self):
        _, surface = to_surface_flambda([1, 2, 3], [1, 2, 3], "um", "W_m2_um")
        _, edd = to_surface_flambda([1, 2, 3], [1, 2, 3], "um", "W_m2_um", "eddington_flux")
        np.testing.assert_allclose(edd, surface*4*np.pi)

    def test_radius_and_distance_scaling(self):
        wave = np.linspace(1, 3, 1000)
        star, planet = Spectrum(wave, np.full(1000, 5.0)), Spectrum(wave, np.full(1000, 2.0))
        a = compare(star, planet, 10, 1, 20, distance_pc=10)
        b = compare(star, planet, 10, 2, 20, distance_pc=20)
        np.testing.assert_allclose(a["planet_star_contrast"], .004)
        np.testing.assert_allclose(b["planet_star_contrast"], a["planet_star_contrast"]*4)
        np.testing.assert_allclose(b["star_observed_flambda_W_m2_um"], a["star_observed_flambda_W_m2_um"]/4)
        np.testing.assert_allclose(b["planet_observed_flambda_W_m2_um"], a["planet_observed_flambda_W_m2_um"])

    def test_band_is_ratio_of_integrals(self):
        wave = np.linspace(1, 3, 1001)
        s, p = Spectrum(wave, wave), Spectrum(wave, np.ones_like(wave))
        band = band_contrast(s, p, 1, [1, 2, 3], [1, 1, 1], "energy")
        self.assertAlmostEqual(band["planet_star_contrast"], .5)
        self.assertNotAlmostEqual(band["planet_star_contrast"], np.mean(1/wave), places=3)

    def test_photon_weighting(self):
        wave = np.linspace(1, 3, 1001)
        s, p = Spectrum(wave, wave), Spectrum(wave, np.ones_like(wave))
        actual = band_contrast(s, p, 1, [1, 2, 3], [1, 1, 1], "photon")
        self.assertAlmostEqual(actual["planet_star_contrast"], 6/13, places=12)

    def test_no_extrapolation_or_partial_filter(self):
        spec = Spectrum([1, 2, 3], [1, 1, 1])
        with self.assertRaises(ValueError):
            compare(spec, spec, 1, 1, wavelength_range=[.5, 3])
        with self.assertRaises(ValueError):
            band_signal(spec, [.9, 2, 3], [0, 1, 0])

    def test_duplicate_and_bad_flux_rejected(self):
        with self.assertRaises(ValueError):
            Spectrum([1, 1, 2], [1, 2, 3])
        with self.assertRaises(ValueError):
            Spectrum([1, 2, 3], [1, -1, 3])

    def test_catalog_exact_selection(self):
        row = {"family": "test", "parameters": {"teff_K": 1000}, "spectrum": {"file": "sample.dat"}}
        self.assertEqual(select_from_catalog([row], "test", {"teff_K": 1000})["file"], "sample.dat")
        with self.assertRaises(ValueError):
            select_from_catalog([row], "test", {"teff_K": 1001})

    def test_air_vacuum_not_mixed(self):
        a = Spectrum([1, 2, 3], [1, 1, 1], {"wavelength_frame": "air"})
        v = Spectrum([1, 2, 3], [1, 1, 1], {"wavelength_frame": "vacuum"})
        with self.assertRaises(ValueError):
            compare(a, v, 1, 1)

    def test_bobcat_reader_and_header_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"synthetic_bobcat.dat"
            path.write_text("1000 100 0.28 0 0 0 1 0\n1 1D-15\n2 2D-15\n3 3D-15\n")
            settings = {"format": "sonora_bobcat", "file": path.name,
                        "parameters": {"teff_K": 1000, "logg_cgs": 4.0}}
            loaded = load_spectrum(settings, folder)
            _, expected = to_surface_flambda([1, 2, 3], [1e-15, 2e-15, 3e-15], "um", "erg_cm2_s_Hz")
            np.testing.assert_allclose(loaded.surface_flux, expected)
            self.assertEqual(loaded.metadata["parameters_verified_from_header"]["logg_cgs"], 4)
            settings["parameters"]["teff_K"] = 1200
            with self.assertRaises(ValueError):
                load_spectrum(settings, folder)

    def test_generic_csv_reader(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"linear.csv"
            path.write_text("wavelength_nm,flux\n1000,1\n2000,2\n3000,3\n")
            loaded = load_spectrum({"format": "ascii", "file": path.name, "delimiter": ",", "skiprows": 1,
                                    "wavelength_unit": "nm", "flux_unit": "W_m2_um", "flux_convention": "surface_flux"}, folder)
            np.testing.assert_allclose(loaded.wavelength_um, [1, 2, 3])
            np.testing.assert_allclose(loaded.surface_flux, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()

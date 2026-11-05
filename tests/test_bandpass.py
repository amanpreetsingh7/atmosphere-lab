"""Analytic tests for generic filter and sky integration."""
import unittest
import numpy as np
from numpy.polynomial import Polynomial
from atmosphere_compare.bandpass import Response, signal, predict_band_contrast
from atmosphere_compare.spectra import Spectrum
from atmosphere_compare.compare import H, C


class BandpassChecks(unittest.TestCase):
    def setUp(self):
        self.response = Response(np.array([9., 11., 14.]), np.array([.4, .5, .65]),
                                 np.array([8., 10., 12., 15.]), np.array([.4, .5, .6, .75]),
                                 (9., 14.), {'limits_um': [9., 14.], 'wavelength_frame': 'vacuum'})

    def test_polynomial_integrals_include_all_components_and_one_photon_factor(self):
        # F=2+3x; filter=.05x-.05; sky=.05x. Native knots deliberately differ.
        spectrum = Spectrum([8., 10.5, 12.7, 15.], 2+3*np.array([8., 10.5, 12.7, 15.]))
        polynomial = Polynomial([2, 3])*Polynomial([-.05, .05])*Polynomial([0, .05])
        for detector in ('energy', 'photon'):
            expected_poly = polynomial if detector == 'energy' else polynomial*Polynomial([0, 1e-6/(H*C)])
            integral = expected_poly.integ()
            expected = integral(14)-integral(9)
            for rule in ('gauss3', 'boole'):
                self.assertAlmostEqual(signal(spectrum, self.response, detector, rule)/expected, 1., places=13)

    def test_independent_adaptive_integral_with_nonmatching_piecewise_knots(self):
        from scipy.integrate import quad
        spectrum = Spectrum([8., 9.7, 11.8, 12.4, 15.], [2., 8., 1., 3., 4.])
        response = Response(np.array([9., 10.3, 12.1, 14.]), np.array([0., .7, 1., 0.]),
                            np.array([8., 9.8, 11.6, 13., 15.]), np.array([.5, .7, .1, .6, .3]), (9., 14.), {})
        knots = sorted({*spectrum.wavelength_um, *response.filter_wave_um, *response.sky_wave_um})
        points = [v for v in knots if 9 < v < 14]
        for detector in ('energy', 'photon'):
            def reference(q):
                v = np.interp(q, spectrum.wavelength_um, spectrum.surface_flux)*np.interp(q, response.filter_wave_um, response.filter_transmission)*np.interp(q, response.sky_wave_um, response.sky_transmission)
                return v if detector == 'energy' else v*q*1e-6/(H*C)
            expected = quad(reference, 9, 14, points=points, epsabs=0., epsrel=1e-12)[0]
            self.assertAlmostEqual(signal(spectrum, response, detector)/expected, 1., places=13)

    def test_radii_distance_and_delta_magnitude(self):
        a = Spectrum([8, 11, 15], [10, 10, 10]);b = Spectrum([8, 11, 15], [2, 2, 2])
        one = predict_band_contrast(a, b, 10, 1, self.response, distance_pc=5)
        two = predict_band_contrast(a, b, 10, 2, self.response, distance_pc=10)
        self.assertAlmostEqual(one['planet_star_contrast'], .002)
        self.assertAlmostEqual(two['planet_star_contrast'], .008)
        self.assertAlmostEqual(two['star_observer_weighted_signal']/one['star_observer_weighted_signal'], .25)
        self.assertAlmostEqual(two['planet_observer_weighted_signal']/one['planet_observer_weighted_signal'], 1.)
        self.assertAlmostEqual(one['delta_magnitude'], -2.5*np.log10(.002))
        np.testing.assert_allclose(one['spectral_weights']['star'], one['spectral_weights']['planet'])
        self.assertTrue(all(r['arithmetic_agrees'] for r in one['independent_arithmetic_checks']))

    def test_band_contrast_is_not_average_spectral_ratio_and_plot_sampling_does_not_control_integral(self):
        a = Spectrum([8, 11, 15], [8, 11, 15]);b = Spectrum([8, 11, 15], [1, 1, 1])
        one = predict_band_contrast(a, b, 1, 1, self.response, weight_samples=31)
        two = predict_band_contrast(a, b, 1, 1, self.response, weight_samples=301)
        self.assertEqual(one['planet_star_contrast'], two['planet_star_contrast'])
        self.assertGreater(abs(one['planet_star_contrast']-np.mean(1/np.linspace(9, 14, 301))), 1e-5)

    def test_no_extrapolation_or_wavelength_frame_mixing(self):
        for spectrum in (Spectrum([1, 3, 5.5], [1, 1, 1]), Spectrum([8, 11, 15], [1, 1, 1], {'wavelength_frame': 'air'})):
            with self.assertRaises(ValueError):signal(spectrum, self.response)
        with self.assertRaises(ValueError):self.response.at([8.9, 14])
        with self.assertRaises(ValueError):Response(np.array([10, 11, 13]), np.ones(3), np.array([9, 11, 15]), np.ones(3), (9, 14), {})

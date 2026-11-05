"""Generic filter and sky band predictions from atmosphere spectra.

All input spectra are emergent surface F_lambda in W/m2/micron. No native
flux renormalization, spectral extrapolation, telescope count calibration or
sensitivity estimation occurs here. Filter and sky are separate components.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import math
import numpy as np
from .compare import H, C, PC_M

def file_sha(path):
    out = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            out.update(chunk)
    return out.hexdigest()


def _component(wave, transmission, label):
    x, t = np.asarray(wave, float), np.asarray(transmission, float)
    if x.ndim != 1 or t.shape != x.shape or len(x) < 3:
        raise ValueError(label + ': expected matching 1D arrays with at least three samples.')
    if not np.isfinite(x).all() or not np.isfinite(t).all() or np.any(x <= 0) or np.any(np.diff(x) <= 0):
        raise ValueError(label + ': wavelengths must be finite, positive and increasing.')
    if np.any(t < 0) or np.any(t > 1) or not np.any(t > 0):
        raise ValueError(label + ': transmission must be in [0,1] with positive support.')
    return x, t


@dataclass
class Response:
    filter_wave_um: np.ndarray
    filter_transmission: np.ndarray
    sky_wave_um: np.ndarray
    sky_transmission: np.ndarray
    limits_um: tuple
    metadata: dict

    def __post_init__(self):
        self.filter_wave_um, self.filter_transmission = _component(
            self.filter_wave_um, self.filter_transmission, 'Filter')
        self.sky_wave_um, self.sky_transmission = _component(
            self.sky_wave_um, self.sky_transmission, 'Sky')
        lo, hi = map(float, self.limits_um)
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError('Response limits must be finite and increasing.')
        for x in (self.filter_wave_um, self.sky_wave_um):
            if lo < x[0] or hi > x[-1]:
                raise ValueError('The filter and sky must both cover the entire chosen band; extrapolation is disabled.')
        self.limits_um = (lo, hi)

    def at(self, wave):
        q = np.asarray(wave, float)
        lo, hi = self.limits_um
        if not np.isfinite(q).all() or np.any(q < lo) or np.any(q > hi):
            raise ValueError('Response query outside the declared band.')
        return np.interp(q, self.filter_wave_um, self.filter_transmission) * np.interp(q, self.sky_wave_um, self.sky_transmission)


def _nodes(spectrum, response):
    lo, hi = response.limits_um
    x = spectrum.wavelength_um
    if lo < x[0] or hi > x[-1]:
        raise ValueError('Spectrum must cover %.6g–%.6g um; this file covers %.6g–%.6g um. No extrapolation.' % (lo, hi, x[0], x[-1]))
    if spectrum.metadata.get('wavelength_frame', 'vacuum') != response.metadata.get('wavelength_frame', 'vacuum'):
        raise ValueError('Spectrum and response wavelength conventions differ.')
    arrays = (x, response.filter_wave_um, response.sky_wave_um)
    return np.unique(np.concatenate(([lo, hi], *[a[(a > lo) & (a < hi)] for a in arrays])))


def signal(spectrum, response, detector='photon', rule='gauss3'):
    """Integrate F_lambda * filter * sky, with one photon factor when requested.

    Three-point Gauss is exact on each segment for the specified piecewise
    linear components (degree four in photon mode). Boole is an independent
    degree-five rule used for arithmetic controls, not an independent physical
    reconstruction of unresolved lines.
    """
    if detector not in ('energy', 'photon'):
        raise ValueError('Choose photon or energy detection.')
    nodes = _nodes(spectrum, response)
    left, right = nodes[:-1], nodes[1:]
    def value(q):
        v = np.interp(q, spectrum.wavelength_um, spectrum.surface_flux) * response.at(q)
        return v*q*1e-6/(H*C) if detector == 'photon' else v
    if rule == 'gauss3':
        mid, half = (left+right)/2, (right-left)/2
        a = math.sqrt(3/5)
        return float(np.sum(half*(5/9*value(mid-a*half)+8/9*value(mid)+5/9*value(mid+a*half))))
    if rule == 'boole':
        step = (right-left)/4
        return float(np.sum((right-left)/90*(7*value(left)+32*value(left+step)+12*value(left+2*step)+32*value(left+3*step)+7*value(right))))
    raise ValueError('Unknown quadrature rule.')


def predict_band_contrast(star, planet, rstar_m, rplanet_m, response,
                          detector='photon', distance_pc=None, weight_samples=301):
    """Calculate brightness from emergent surface spectra and a declared response.

    Displayed photon weights are relative spectral-density samples, not
    absolute fluxes or bin probabilities.
    """
    if not all(np.isfinite(v) and v > 0 for v in (rstar_m, rplanet_m)):
        raise ValueError('Radii must be positive and finite, in metres.')
    if distance_pc is not None and (not np.isfinite(distance_pc) or distance_pc <= 0):
        raise ValueError('Distance must be positive and finite, in parsecs.')
    if isinstance(weight_samples, bool) or int(weight_samples) != weight_samples or not 3 <= weight_samples <= 10001:
        raise ValueError('Choose an integer from 3 to 10001 display samples.')
    ss, ps = signal(star, response, detector), signal(planet, response, detector)
    if not np.isfinite([ss, ps]).all() or ss <= 0 or ps <= 0:
        raise ValueError('Both spectra need positive signal in this response.')
    contrast = (rplanet_m/rstar_m)**2*ps/ss
    if not np.isfinite(contrast) or contrast <= 0:
        raise ValueError('Contrast is outside the representable positive range.')
    checks = []
    for body, spectrum, actual in (('star', star, ss), ('planet', planet, ps)):
        reference = signal(spectrum, response, detector, rule='boole')
        error = abs(actual/reference-1)
        checks.append({'body': body, 'gauss3_vs_boole_relative_difference': error,
                       'arithmetic_tolerance': 1e-10, 'arithmetic_agrees': error <= 1e-10})
        if not np.isfinite(error) or error > 1e-10:
            raise ValueError('Independent band-integral arithmetic check failed for ' + body)
    w = np.linspace(*response.limits_um, int(weight_samples))
    result = {'band': response.metadata, 'detector': detector,
        'rstar_m': float(rstar_m), 'rplanet_m': float(rplanet_m), 'distance_pc': distance_pc,
        'star_surface_band_signal': ss, 'planet_surface_band_signal': ps,
        'surface_signal_unit': 'photon m^-2 s^-1' if detector == 'photon' else 'W m^-2',
        'planet_star_contrast': contrast, 'delta_magnitude': float(-2.5*np.log10(contrast)),
        'source_spectra': {'star': star.metadata, 'planet': planet.metadata},
        'independent_arithmetic_checks': checks,
        'native_samples_inside_band': {body: int(np.count_nonzero((s.wavelength_um >= w[0]) & (s.wavelength_um <= w[-1])))
            for body, s in (('star', star), ('planet', planet))},
        'spectral_weights': {'wavelength_um': w.tolist(),
            'normalization': 'Each photon spectral density divided by its maximum; uniform wavelength grid; not bin probabilities.',
            'integration_uses_native_samples': True},
        'response_samples': {'wavelength_um': w.tolist(),
            'filter': np.interp(w, response.filter_wave_um, response.filter_transmission).tolist(),
            'sky': np.interp(w, response.sky_wave_um, response.sky_transmission).tolist(),
            'product': response.at(w).tolist()},
        'is_observation': False, 'is_sensitivity_curve': False,
        'scope': 'Native thermal brightness prediction through reference filter/sky. Imposed radii; no reflected light, irradiation solver or calibrated telescope count/noise model.'}
    for body, spectrum in (('star', star), ('planet', planet)):
        weight = np.interp(w, spectrum.wavelength_um, spectrum.surface_flux)*response.at(w)*w*1e-6/(H*C)
        if np.max(weight) <= 0:
            raise ValueError('Sampled photon weights have no positive support; increase samples.')
        result['spectral_weights'][body] = (weight/np.max(weight)).tolist()
    if distance_pc is not None:
        d = distance_pc*PC_M
        result['star_observer_weighted_signal'] = ss*(rstar_m/d)**2
        result['planet_observer_weighted_signal'] = ps*(rplanet_m/d)**2
        result['observer_signal_note'] = 'Declared response weighted flux at the observer, not a calibrated instrument count rate.'
    return result

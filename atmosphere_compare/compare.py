"""Radius-scaled thermal-emission contrasts, using integrated wavelength bins."""
import numpy as np
from .spectra import Spectrum

RADII_M = {"R_sun": 6.957e8, "R_jup": 7.1492e7, "R_earth": 6.3781e6, "m": 1.0}
PC_M = 3.085677581491367e16
H = 6.62607015e-34
C = 299792458.0


def radius_m(spec):
    value = float(spec["value"])
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Radius must be positive and finite.")
    return value * RADII_M[spec["unit"]]


def integral(spec, edges):
    """Integral of the piecewise-linear native F_lambda representation.

    No extrapolation. This is numerical binning, not an instrument LSF model.
    """
    x, y = spec.wavelength_um, spec.surface_flux
    e = np.asarray(edges, float)
    if e.ndim != 1 or len(e) < 2 or not np.all(np.isfinite(e)) or np.any(np.diff(e) <= 0):
        raise ValueError("Bin edges must be finite and strictly increasing.")
    if e[0] < x[0] or e[-1] > x[-1]:
        raise ValueError("Bin extends outside spectrum; extrapolation is disabled.")
    # Integrate locally inside each output bin. Differencing a global cumulative
    # primitive can erase faint bins after a bright region through cancellation.
    knots = np.unique(np.concatenate((e, x[(x > e[0]) & (x < e[-1])])))
    values = np.interp(knots, x, y)
    areas = np.diff(knots) * (values[:-1] / 2 + values[1:] / 2)
    bins = np.searchsorted(e, knots[:-1], side="right") - 1
    return np.bincount(bins, weights=areas, minlength=len(e)-1)


def compare(star, planet, rstar_m, rplanet_m, resolving_power=100, wavelength_range=None, distance_pc=None):
    if star.metadata.get("wavelength_frame", "vacuum") != planet.metadata.get("wavelength_frame", "vacuum"):
        raise ValueError("Air and vacuum wavelength spectra cannot be mixed without an explicit conversion.")
    if not all(np.isfinite(v) and v > 0 for v in (rstar_m, rplanet_m, resolving_power)):
        raise ValueError("Radii and bin resolving power must be positive and finite.")
    lo, hi = max(star.wavelength_um[0], planet.wavelength_um[0]), min(star.wavelength_um[-1], planet.wavelength_um[-1])
    if wavelength_range is not None:
        req_lo, req_hi = map(float, wavelength_range)
        if not np.isfinite(req_lo) or not np.isfinite(req_hi) or req_lo < lo or req_hi > hi:
            raise ValueError("Requested range exceeds common model coverage. Choose a covered range.")
        lo, hi = req_lo, req_hi
    if lo >= hi:
        raise ValueError("Spectra have no common wavelength interval.")
    n = int(np.ceil(np.log(hi/lo) / np.log1p(1.0/resolving_power)))
    if n > 200000:
        raise ValueError("Too many output bins. Use a lower bin resolving power.")
    edges = np.geomspace(lo, hi, n+1)
    s = integral(star, edges)/np.diff(edges)
    p = integral(planet, edges)/np.diff(edges)
    if np.any(s <= 0):
        raise ValueError("Zero stellar flux in an output bin; cannot form contrast.")
    result = {"wavelength_um": np.sqrt(edges[:-1]*edges[1:]), "bin_lower_um": edges[:-1], "bin_upper_um": edges[1:],
              "star_surface_flambda_W_m2_um": s, "planet_surface_flambda_W_m2_um": p,
              "surface_flux_ratio": p/s, "planet_star_contrast": (rplanet_m/rstar_m)**2*p/s}
    for label, spec in (("star", star), ("planet", planet)):
        result[label+"_native_samples_in_bin"] = np.diff(np.searchsorted(spec.wavelength_um, edges))
    if distance_pc is not None:
        if not np.isfinite(distance_pc) or distance_pc <= 0:
            raise ValueError("Distance must be positive and finite.")
        d = distance_pc*PC_M
        result["star_observed_flambda_W_m2_um"] = s*(rstar_m/d)**2
        result["planet_observed_flambda_W_m2_um"] = p*(rplanet_m/d)**2
    return result


def band_signal(spec, wavelength, throughput, detector="photon"):
    """Integrate a supplied response; photon -> photons/m^2/s, energy -> W/m^2."""
    w, t = np.asarray(wavelength, float), np.asarray(throughput, float)
    if w.ndim != 1 or t.shape != w.shape or len(w) < 3 or not np.all(np.isfinite(w)) or not np.all(np.isfinite(t)):
        raise ValueError("Filter needs at least three finite wavelength/throughput samples.")
    if np.any(np.diff(w) <= 0) or np.any(w <= 0) or np.any(t < 0) or np.any(t > 1) or not np.any(t > 0):
        raise ValueError("Filter wavelengths must increase; throughput must be in [0,1] with nonzero support.")
    x, y = spec.wavelength_um, spec.surface_flux
    if w[0] < x[0] or w[-1] > x[-1]:
        raise ValueError("Full supplied filter coverage is required; no partial-band result is reported.")
    if detector not in ("photon", "energy"):
        raise ValueError("Detector must be photon or energy. Do not photon-weight an already weighted response twice.")
    # Union knots, then 2-point Gauss quadrature: exact for linear F * linear T * lambda.
    nodes = np.unique(np.concatenate((w, x[(x > w[0]) & (x < w[-1])])))
    mid, half = (nodes[1:]+nodes[:-1])/2, np.diff(nodes)/2
    total = 0.0
    for sign in (-1, 1):
        q = mid + sign*half/np.sqrt(3)
        v = np.interp(q, x, y)*np.interp(q, w, t)
        if detector == "photon":
            v *= q*1e-6/(H*C)
        total += np.sum(v*half)
    return float(total)


def band_contrast(star, planet, radius_ratio, wavelength, throughput, detector="photon"):
    if star.metadata.get("wavelength_frame", "vacuum") != planet.metadata.get("wavelength_frame", "vacuum"):
        raise ValueError("Air and vacuum wavelength spectra cannot be mixed without an explicit conversion.")
    radius_ratio = float(radius_ratio)
    if not np.isfinite(radius_ratio) or radius_ratio <= 0:
        raise ValueError("Radius ratio must be positive and finite.")
    s = band_signal(star, wavelength, throughput, detector)
    p = band_signal(planet, wavelength, throughput, detector)
    if s <= 0:
        raise ValueError("No stellar signal in filter.")
    return {"star_surface_band_signal": s, "planet_surface_band_signal": p,
            "planet_star_contrast": radius_ratio**2*p/s,
            "delta_magnitude": float(-2.5*np.log10(radius_ratio**2*p/s)) if p > 0 else None,
            "detector": detector, "signal_unit": "photon m^-2 s^-1" if detector == "photon" else "W m^-2"}


def equivalent_flux_density_Jy(spectrum,radius,distance_pc,wavelength,throughput,detector='photon'):
    """Response-equivalent Fnu, normalized to a flat-Fnu reference, not monochromatic Fnu."""
    if not all(np.isfinite(v) and v>0 for v in (radius,distance_pc)):
        raise ValueError('Radius and distance must be positive and finite.')
    w=np.asarray(wavelength,float)
    # band_signal checks the response and native coverage before reference construction.
    signal=band_signal(spectrum,w,throughput,detector)
    nodes=np.unique(np.concatenate((np.geomspace(w[0],w[-1],6000),w)))
    flat=Spectrum(nodes,C/(nodes*1e-6)**2*1e-6,{'wavelength_frame':spectrum.metadata.get('wavelength_frame','vacuum')})
    normalization=band_signal(flat,w,throughput,detector)
    return float(signal*(radius/(distance_pc*PC_M))**2/normalization/1e-26)

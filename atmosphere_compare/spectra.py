"""Explicit-unit adapters for emergent surface spectra. Native file loading only."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import io
import math
import gzip
import warnings
import numpy as np

C = 299792458.0
FLUX_UNIT = "W m^-2 um^-1"
WAVE_FACTORS = {"um": 1.0, "nm": 1e-3, "angstrom": 1e-4, "m": 1e6, "cm": 1e4}


@dataclass
class Spectrum:
    wavelength_um: np.ndarray
    surface_flux: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        x, y = np.asarray(self.wavelength_um, float), np.asarray(self.surface_flux, float)
        if x.ndim != 1 or y.shape != x.shape or len(x) < 3:
            raise ValueError("A spectrum needs matching one-dimensional arrays with at least three samples.")
        if not np.all(np.isfinite(x)) or np.any(x <= 0) or not np.all(np.isfinite(y)) or np.any(y < 0):
            raise ValueError("Wavelengths must be positive and finite; flux must be finite and nonnegative.")
        order = np.argsort(x)
        x, y = x[order], y[order]
        if np.any(np.diff(x) <= 0):
            raise ValueError("Duplicate wavelengths are ambiguous. Resolve them in the source file.")
        self.wavelength_um, self.surface_flux = x, y


def to_surface_flambda(wavelength, flux, wavelength_unit, flux_unit, convention="surface_flux"):
    """Convert F_nu or F_lambda to emergent F_lambda in W/m^2/micron."""
    if wavelength_unit not in WAVE_FACTORS:
        raise ValueError("Supported wavelength units: " + ", ".join(WAVE_FACTORS))
    wave = np.asarray(wavelength, float) * WAVE_FACTORS[wavelength_unit]
    flux = np.asarray(flux, float)
    factors = {"W_m2_um": 1.0, "W_m2_m": 1e-6, "erg_cm2_s_um": 1e-3,
               "erg_cm2_s_A": 10.0, "erg_cm2_s_cm": 1e-7}
    if flux_unit in factors:
        converted = flux * factors[flux_unit]
    elif flux_unit in ("erg_cm2_s_Hz", "W_m2_Hz"):
        if np.any(wave <= 0):
            raise ValueError("Positive wavelengths are required for F_nu conversion.")
        converted = flux * (1e-3 if flux_unit == "erg_cm2_s_Hz" else 1.0) * C / (wave * 1e-6)**2 * 1e-6
    else:
        raise ValueError("Unknown flux unit: " + flux_unit)
    if convention == "eddington_flux":
        converted = converted * (4 * math.pi)
    elif convention != "surface_flux":
        raise ValueError("Use surface_flux or eddington_flux. Observed flux and intensity require different treatment.")
    return wave, converted


def file_record(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def load_spectrum(spec, base):
    """Load release-specific emission files; grid interpolation lives in grid.py."""
    path = (Path(base) / spec["file"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Model file missing: %s. See MODEL_DOWNLOADS.md; no substitute is inserted." % path)
    fmt = spec["format"]
    metadata = {"label": spec.get("label", path.name), "format": fmt,
                "parameters_declared": spec.get("parameters", {}),
                "parameters_verified_from_header": {}, "files": [file_record(path)],
                "wavelength_frame": spec.get("wavelength_frame", "vacuum"),
                "source_url": spec.get("source_url"), "flux_output_unit": FLUX_UNIT}
    if metadata["wavelength_frame"] not in ("air", "vacuum"):
        raise ValueError("Declare wavelength_frame as air or vacuum.")
    if fmt in ('phoenix_hires','sonora_bobcat','sonora_cholla','sonora_diamondback','atmo2020_ascii','atmo2020_stsci_fits','sonora_elf_owl_v2','sonora_flame_skimmer_v1_r3k') and metadata['wavelength_frame']!='vacuum':
        raise ValueError('This native emission release uses vacuum wavelengths. A converted export needs an explicit ASCII mapping, not a changed native-format label.')
    if fmt in ("ascii", "sonora_bobcat"):
        # Fortran D exponents occur in some model releases.
        text = (gzip.open(path, "rt").read() if path.suffix.lower() == ".gz" else path.read_text())
        text = text.replace("D+", "E+").replace("D-", "E-")
        if fmt == "sonora_bobcat":
            # The official archive uses a numeric+labels parameter line followed
            # by a units line. Older exports can omit the units line.
            lines = text.splitlines()
            skip = None
            for i, line in enumerate(lines[:30]):
                tokens = line.split()
                if len(tokens) == 2:
                    try:
                        a, b = map(float, tokens)
                    except ValueError:
                        continue
                    if a > 0 and b >= 0:
                        skip = i
                        break
            if skip is None:
                raise ValueError("No two-column Bobcat spectrum found after its header.")
        else:
            skip = spec.get("skiprows", 0)
        data = np.loadtxt(io.StringIO(text), skiprows=skip, delimiter=spec.get("delimiter"),
                          usecols=(spec.get("wavelength_column", 0), spec.get("flux_column", 1)), ndmin=2)
        wave, flux = data[:, 0], data[:, 1]
        if fmt == "sonora_bobcat":
            wave_unit, flux_unit, convention = "um", "erg_cm2_s_Hz", "surface_flux"
            metadata["original_header"] = text.splitlines()[:skip]
            metadata["header_lines_skipped"] = skip
            unit_header = "".join(" ".join(metadata["original_header"][1:]).lower().split()).replace("^", "")
            if unit_header and not ("micron" in unit_header and "erg/cm2/s/hz" in unit_header):
                raise ValueError("Bobcat native unit header must specify microns and erg/cm2/s/Hz. Use an explicit ASCII mapping for a converted file.")
            metadata["native_units_verified_from_header"] = bool(unit_header)
            if not unit_header:
                metadata["unit_note"] = "No unit line in this legacy export; units follow the explicitly selected Bobcat release convention."
            # Numeric first-line parameters: Teff, g(m/s^2), Y, fsed, kz, [M/H], C/O(relative solar), fhole.
            if skip >= 1:
                try:
                    vals = [float(v) for v in text.splitlines()[0].split()[:8]]
                except ValueError:
                    vals = []
                if len(vals) >= 8 and vals[0] > 0 and vals[1] > 0:
                    measured = {"teff_K": vals[0], "logg_cgs": math.log10(vals[1] * 100),
                                "metallicity_dex": vals[5], "co_relative_solar": vals[6]}
                    _check_parameters(spec.get("parameters", {}), measured)
                    metadata["parameters_verified_from_header"] = measured
                    from .bobcat_coordinates import grid_coordinates
                    coordinates = grid_coordinates(path.name, measured)
                    if coordinates is not None:
                        metadata['grid_coordinates'] = coordinates
        else:
            wave_unit, flux_unit = spec["wavelength_unit"], spec["flux_unit"]
            convention = spec["flux_convention"]
    elif fmt == "ck04_stsci_fits":
        if metadata['wavelength_frame'] != 'vacuum':
            raise ValueError('This integration adopts vacuum CK04 coordinates; converted files need an explicit adapter.')
        from .ck04 import read_ck04
        wave, flux, extra = read_ck04(path, spec)
        _check_parameters(spec.get('parameters', {}), extra['parameters_verified_from_header'])
        metadata.update(extra)
        wave_unit, flux_unit, convention = 'angstrom', 'erg_cm2_s_A', 'surface_flux'
    elif fmt == 'bt_settl_stsci_fits':
        from .bt_settl_stsci import read_bt_settl_stsci
        wave,flux,extra=read_bt_settl_stsci(path,spec)
        _check_parameters(spec.get('parameters',{}),extra['parameters_verified_from_filename'])
        _check_parameters(spec.get('parameters',{}),extra['parameters_verified_from_header'])
        metadata.update(extra)
        wave_unit,flux_unit,convention='angstrom','erg_cm2_s_A','surface_flux'
    elif fmt == "phoenix_hires":
        try:
            from astropy.io import fits
            from astropy import units as u
        except ImportError as exc:
            raise ImportError("PHOENIX FITS loading requires astropy. Install requirements.txt.") from exc
        wave_path = (Path(base) / spec["wavelength_file"]).expanduser().resolve()
        with fits.open(path, mode="readonly", memmap=False) as hdus:
            flux = np.asarray(hdus[0].data, float)
            header = hdus[0].header
            measured = {k: float(header[h]) for k, h in (("teff_K", "PHXTEFF"), ("logg_cgs", "PHXLOGG"),
                                                       ("metallicity_dex", "PHXM_H"), ("alpha_dex", "PHXALPHA")) if h in header}
        with fits.open(wave_path, mode="readonly", memmap=False) as hdus:
            wave = np.asarray(hdus[0].data, float)
            wave_header = hdus[0].header
        flux_unit_header, wave_unit_header = header.get("BUNIT"), wave_header.get("UNIT")
        if not flux_unit_header or not wave_unit_header:
            raise ValueError("Native PHOENIX high-resolution files need flux BUNIT and wavelength UNIT headers. Use an explicit ASCII mapping for converted data.")
        with warnings.catch_warnings():
            # The published BUNIT uses several slashes. Accept that documented
            # syntax while still rejecting incompatible units or scale factors.
            warnings.filterwarnings("ignore",message=".*multiple slashes.*",category=u.UnitsWarning)
            flux_factor=u.Unit(flux_unit_header).to(u.erg/u.s/u.cm**2/u.cm)
            wave_factor=u.Unit(wave_unit_header).to(u.AA)
        if not (math.isclose(flux_factor, 1., rel_tol=1e-12) and math.isclose(wave_factor, 1., rel_tol=1e-12)):
            raise ValueError("PHOENIX high-resolution units must be erg/s/cm2/cm and Angstrom. This file has a different convention.")
        metadata["unit_headers_checked"] = {"BUNIT": flux_unit_header, "wavelength_UNIT": wave_unit_header}
        metadata["files"].append(file_record(wave_path))
        _check_parameters(spec.get("parameters", {}), measured)
        metadata["parameters_verified_from_header"] = measured
        wave_unit, flux_unit, convention = "angstrom", "erg_cm2_s_cm", "surface_flux"
    elif fmt in ('sonora_cholla','sonora_diamondback','atmo2020_ascii','atmo2020_stsci_fits','bt_settl_svo','sonora_elf_owl_v2','sonora_flame_skimmer_v1_r3k'):
        from .native_formats import load_extra
        wave,flux,wave_unit,flux_unit,extra=load_extra(path,fmt,spec)
        _check_parameters(extra['parameters_verified_from_filename'],extra['parameters_verified_from_header'])
        _check_parameters(spec.get('parameters',{}),extra['parameters_verified_from_filename'])
        _check_parameters(spec.get('parameters',{}),extra['parameters_verified_from_header'])
        metadata.update(extra);convention='surface_flux'
    else:
        raise ValueError("Unknown format %s. Use an explicit ASCII adapter for other exported grids." % fmt)
    metadata.update(input_wavelength_unit=wave_unit, input_flux_unit=flux_unit, input_flux_convention=convention)
    wave, flux = to_surface_flambda(wave, flux, wave_unit, flux_unit, convention)
    result = Spectrum(wave, flux, metadata)
    result.metadata["wavelength_range_um"] = [float(result.wavelength_um[0]), float(result.wavelength_um[-1])]
    result.metadata["native_samples"] = len(result.wavelength_um)
    return result


def _check_parameters(declared, measured):
    for key in declared.keys() & measured.keys():
        equal=(math.isclose(float(declared[key]),float(measured[key]),rel_tol=1e-6,abs_tol=2e-3)
               if isinstance(measured[key],(int,float)) else declared[key]==measured[key])
        if not equal:
            raise ValueError("Declared %s=%s disagrees with model header value %s." % (key, declared[key], measured[key]))


def select_from_catalog(catalog, family, parameters):
    """Exact selection; never silently snap parameters to a nearest grid point."""
    matches = [row for row in catalog if row["family"] == family and row["parameters"] == parameters]
    if len(matches) != 1:
        raise ValueError("Expected one exact model match; found %d for %s %s. No extrapolation or nearest model was used." % (len(matches), family, parameters))
    return dict(matches[0]["spectrum"], parameters=matches[0]["parameters"])

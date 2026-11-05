"""Property-driven atmosphere selection, spectra and band predictions.

Native exact selection is the default. Optional interpolation mixes surface
F_lambda with bounded temperature/gravity coordinate weights, holding every
other grid parameter fixed. It is not a new atmosphere solution.
"""
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
import copy
import csv
import io
import json
import hashlib
import math
import numpy as np
from . import __version__
from .spectra import Spectrum, load_spectrum, WAVE_FACTORS
from .compare import radius_m
from .bandpass import Response, predict_band_contrast, file_sha
from .grid import BandGrid


def _same(first, second):
    if isinstance(first, bool) or isinstance(second, bool):return type(first) is type(second) and first == second
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return math.isclose(float(first), float(second), rel_tol=1e-10, abs_tol=1e-10)
    return first == second


class ModelLibrary:
    """One catalog snapshot, with a small per-request cache of loaded nodes."""
    def __init__(self, catalog, rows=None):
        self.path = Path(catalog).expanduser().resolve()
        self.kind = 'catalog_file' if rows is None else 'explicit_in_memory_rows'
        content = self.path.read_bytes() if rows is None else json.dumps(rows, sort_keys=True).encode()
        self.rows = json.loads(content) if rows is None else copy.deepcopy(rows)
        self.sha256 = hashlib.sha256(content).hexdigest()
        if not isinstance(self.rows, list) or not self.rows:raise ValueError('Model catalog must be a nonempty list.')
        for row in self.rows:
            if row.get('role') not in ('star', 'planet') or not isinstance(row.get('family'), str) or not isinstance(row.get('parameters'), dict) or not isinstance(row.get('spectrum'), dict):
                raise ValueError('Each catalog row needs role, family, parameters and spectrum mapping.')
        self.cache = OrderedDict()

    def inventory(self):
        result = []
        for role, family in sorted({(r['role'], r['family']) for r in self.rows}):
            rows = [r for r in self.rows if r['role'] == role and r['family'] == family]
            keys = sorted({key for r in rows for key in r['parameters']})
            result.append({'role': role, 'family': family, 'registered_models': len(rows),
                'installed_models': sum((self.path.parent/Path(r['spectrum']['file']).expanduser()).is_file() for r in rows),
                'parameters': {key: sorted({json.dumps(r['parameters'][key]) for r in rows if key in r['parameters']}) for key in keys}})
        return result

    def _load(self, row):
        token = [row]
        for key in ('file', 'wavelength_file'):
            if key in row['spectrum']:
                p = (self.path.parent/Path(row['spectrum'][key]).expanduser()).resolve()
                if not p.is_file():raise FileNotFoundError('Selected model is not installed: '+str(p))
                st = p.stat();token.append([str(p), st.st_size, st.st_mtime_ns])
        key = json.dumps(token, sort_keys=True)
        if key not in self.cache:
            mapping = dict(row['spectrum'], parameters=row['parameters'])
            for field in ('file', 'wavelength_file'):
                if field in mapping:mapping[field] = str((self.path.parent/Path(mapping[field]).expanduser()).resolve())
            self.cache[key] = load_spectrum(mapping, self.path.parent)
            if len(self.cache) > 8:self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        s = self.cache[key]
        # Metadata is private to this selection; cached native arrays are not mutated.
        result = Spectrum(s.wavelength_um, s.surface_flux, copy.deepcopy(s.metadata))
        result.metadata.update(family=row['family'], catalog_path=str(self.path), catalog_kind=self.kind, catalog_sha256=self.sha256, construction='native_exact')
        if row.get('grid_coordinates'):
            actual = result.metadata.get('grid_coordinates', {})
            for field in ('nominal_teff_K', 'filename_gravity_m_s2', 'reported_teff_K', 'reported_gravity_m_s2'):
                if field in row['grid_coordinates'] and not _same(row['grid_coordinates'][field], actual.get(field)):
                    raise ValueError('Catalog connectivity differs from the native source: '+field)
        return result

    def resolve(self, role, selection, response):
        allowed = {'family', 'parameters', 'radius', 'selection'}
        if not isinstance(selection, dict) or set(selection)-allowed:raise ValueError('Model input accepts family, parameters, radius and selection.')
        family, query = selection.get('family'), selection.get('parameters')
        if not isinstance(family, str) or not isinstance(query, dict) or not query:raise ValueError('Supply a model family and nonempty parameter object.')
        for key, value in query.items():
            if not isinstance(value, (str, int, float, bool)) or (isinstance(value, (int, float)) and not isinstance(value, bool) and not np.isfinite(value)):
                raise ValueError('Grid parameters must be finite scalars or native categorical strings.')
            if key in ('teff_K', 'logg_cgs', 'metallicity_dex') and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ValueError(key+' must be numeric.')
            if key == 'teff_K' and value <= 0:raise ValueError('Temperature must be positive.')
        rows = [r for r in self.rows if r['role'] == role and r['family'] == family]
        if not rows:raise ValueError('No registered %s models in family %s. Inspect the catalog inventory.' % (role, family))
        known = {key for r in rows for key in r['parameters']}
        if set(query)-known:raise ValueError('Unsupported parameters for '+family+': '+', '.join(sorted(set(query)-known)))
        mode = selection.get('selection', 'exact')
        if mode not in ('exact', 'linear_surface_flux'):raise ValueError('Selection must be exact or linear_surface_flux.')
        exact = [r for r in rows if all(key in r['parameters'] and _same(value, r['parameters'][key]) for key, value in query.items())]
        if len(exact) > 1:raise ValueError('Parameters are ambiguous; specify all distinguishing cloud/composition/chemistry parameters.')
        if len(exact) == 1:
            spectrum = self._load(exact[0]);spectrum.metadata['parameters_requested'] = copy.deepcopy(query)
            return spectrum, copy.deepcopy(exact[0])
        if mode == 'exact':
            raise ValueError('No exact installed catalog selection for %s %s. No nearest model or extrapolation was used.' % (family, query))
        axes = ('teff_K', 'logg_cgs')
        if not set(axes).issubset(query):raise ValueError('Temperature/gravity interpolation requires teff_K and logg_cgs.')
        held_query = {k: v for k, v in query.items() if k not in axes}
        rows = [r for r in rows if all(k in r['parameters'] and _same(v, r['parameters'][k]) for k, v in held_query.items())]
        if not rows:raise ValueError('No models at the requested fixed composition/cloud/chemistry values.')
        groups = {json.dumps({k: v for k, v in r['parameters'].items() if k not in axes}, sort_keys=True) for r in rows}
        if len(groups) != 1:raise ValueError('Interpolation is ambiguous; hold every non-temperature/gravity parameter fixed.')
        if not all(set(axes).issubset(r['parameters']) for r in rows):raise ValueError('Catalog does not provide both interpolation coordinates.')
        # BandGrid is used only for bounded coordinate weights, not its log-band-flux rule.
        mesh = BandGrid(rows, np.ones(len(rows)), axes)
        _, corners = mesh.evaluate(query)
        sources = []
        for corner in corners:
            candidates = [r for r in rows if r['parameters'] == corner['parameters']]
            if len(candidates) != 1:raise ValueError('Duplicate interpolation corner.')
            spectrum = self._load(candidates[0]);lo, hi = response.limits_um
            if spectrum.wavelength_um[0] > lo or spectrum.wavelength_um[-1] < hi:raise ValueError('Every interpolation corner must cover the full response band.')
            sources.append((spectrum, corner['weight']))
        frames = {s.metadata['wavelength_frame'] for s, weight in sources}
        if len(frames) != 1 or next(iter(frames)) != response.metadata.get('wavelength_frame', 'vacuum'):raise ValueError('Interpolation sources and response must share a wavelength convention.')
        lo, hi = response.limits_um
        nodes = np.unique(np.concatenate(([lo, hi], *[s.wavelength_um[(s.wavelength_um > lo)&(s.wavelength_um < hi)] for s, weight in sources])))
        redundant_midpoint = len(nodes) < 3
        if redundant_midpoint:nodes = np.unique(np.append(nodes, (lo+hi)/2))
        flux = sum(weight*np.interp(nodes, s.wavelength_um, s.surface_flux) for s, weight in sources)
        params = json.loads(next(iter(groups)));params.update({a: query[a] for a in axes})
        files = {record['sha256']: record for s, weight in sources for record in s.metadata['files']}
        metadata = {'label': family+' (bounded surface-flux interpolation)', 'family': family, 'construction': 'linear_surface_flux_interpolation',
            'parameters_requested': copy.deepcopy(query), 'parameters_declared': params, 'parameters_verified_from_header': {},
            'wavelength_frame': next(iter(frames)), 'wavelength_range_um': [lo, hi], 'flux_output_unit': 'W m^-2 um^-1',
            'catalog_path': str(self.path), 'catalog_sha256': self.sha256, 'files': list(files.values()),
            'interpolation': {'redundant_midpoint_inserted': redundant_midpoint, 'method': 'Linear mixture of emergent surface F_lambda on the union of corner wavelength knots; log(Teff)/linear logg coordinate weights.',
                'corners': [{'weight': weight, 'source': s.metadata} for s, weight in sources],
                'scope': 'Within a complete bracketing cell of cataloged nodes; no extrapolation, composition interpolation or flux renormalization. Accuracy is grid/band dependent and not certified by algebraic tests.'},
            'band_limited_reconstruction': True}
        selection_row = {'role': role, 'family': family, 'parameters': params, 'spectrum': {'label': metadata['label']}}
        return Spectrum(nodes, flux, metadata), selection_row


def _read_component(mapping, base):
    if not isinstance(mapping, dict):raise ValueError('Each response component needs a mapping.')
    if set(mapping)-{'file', 'wavelength_unit', 'transmission_unit', 'wavelength_frame'}:raise ValueError('Unknown response component option.')
    path = (Path(base)/Path(mapping['file']).expanduser()).resolve()
    records = list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8-sig'))))
    if not records or not {'wavelength', 'transmission'}.issubset(records[0]):raise ValueError('Response CSV requires wavelength,transmission columns.')
    unit = mapping.get('wavelength_unit');scale = mapping.get('transmission_unit')
    if unit not in WAVE_FACTORS or scale not in ('fraction', 'percent'):raise ValueError('Declare wavelength_unit and transmission_unit (fraction or percent).')
    frame = mapping.get('wavelength_frame')
    if frame not in ('air', 'vacuum'):raise ValueError('Declare response air/vacuum wavelength convention.')
    wave = np.array([float(r['wavelength']) for r in records])*WAVE_FACTORS[unit]
    transmission = np.array([float(r['transmission']) for r in records])/(100 if scale == 'percent' else 1)
    return wave, transmission, {'path': str(path), 'sha256': file_sha(path), 'input_units': mapping, 'wavelength_frame': frame}


def response_from_request(mapping, base):
    if isinstance(mapping, Response):return mapping
    if not isinstance(mapping, dict):raise ValueError('Supply a response object or explicit response mapping.')
    if mapping.get('kind') != 'filter_and_sky_csv':raise ValueError('Response kind must be filter_and_sky_csv.')
    if set(mapping)-{'kind', 'filter', 'sky', 'limits_um', 'name'}:raise ValueError('Unexpected response option.')
    fw, ft, fm = _read_component(mapping['filter'], base);sw, st, sm = _read_component(mapping['sky'], base)
    if fm['wavelength_frame'] != sm['wavelength_frame']:raise ValueError('Filter and sky conventions differ.')
    return Response(fw, ft, sw, st, tuple(mapping['limits_um']), {'name': mapping.get('name', 'Declared filter and sky'),
        'limits_um': list(mapping['limits_um']), 'wavelength_frame': fm['wavelength_frame'], 'filter_source': fm, 'sky_source': sm,
        'calibration_scope': 'Caller-declared filter and sky transmission; no implied count calibration, PWV/airmass validation or sky emission model.'})


def predict_from_properties(catalog, star, companion, response, *, base=None, distance_pc=None, detector='photon', samples=301, library=None):
    """Public API: model families + physical/grid properties -> spectra/contrast.

    Radius mappings use value/unit (R_sun, R_jup, R_earth or m). Native exact
    selection is default; optional interpolation is explicit per body.
    """
    base = Path(base) if base is not None else Path(catalog).expanduser().resolve().parent
    response = response_from_request(response, base)
    library = library or ModelLibrary(catalog)
    if library.path != Path(catalog).expanduser().resolve():raise ValueError('Library/catalog mismatch.')
    rstar, rplanet = radius_m(star['radius']), radius_m(companion['radius'])
    ss, sr = library.resolve('star', star, response);ps, pr = library.resolve('planet', companion, response)
    result = predict_band_contrast(ss, ps, rstar, rplanet, response, detector, distance_pc, samples)
    result.update(software_version=__version__, created_local=datetime.now().isoformat(), catalog_selections={'star': sr, 'planet': pr},
        property_inputs={'star': copy.deepcopy(star), 'companion': copy.deepcopy(companion)},
        selection_provenance={'catalog': str(library.path), 'catalog_kind': library.kind, 'catalog_sha256': library.sha256})
    result['scope'] = 'Thermal model brightness from the declared native or bounded interpolated surface spectra, response and imposed radii. No observed sensitivity, reflected light, flux renormalization or self-consistent irradiated atmosphere is inferred.'
    result['sampling_counts'] = {body: {'construction': s.metadata['construction'], 'samples_inside_band': result['native_samples_inside_band'][body]} for body, s in (('star', ss), ('planet', ps))}
    result['spectral_weights']['integration_representation_by_body'] = {body: s.metadata['construction'] for body, s in (('star', ss), ('planet', ps))}
    result['spectral_weights']['integration_uses_native_samples'] = all(s.metadata['construction'] == 'native_exact' for s in (ss, ps))
    if distance_pc is not None:result['observer_signal_note'] = 'Declared filter/sky response weighted flux at the observer; not a calibrated instrument count rate.'
    for body, s in (('star', ss), ('planet', ps)):
        if s.metadata['construction'] != 'native_exact':result['native_samples_inside_band'][body] = None
    w = np.array(result['spectral_weights']['wavelength_um'])
    a = np.interp(w, ss.wavelength_um, ss.surface_flux);b = np.interp(w, ps.wavelength_um, ps.surface_flux)
    ratio = np.divide((rplanet/rstar)**2*b, a, out=np.full(len(w), np.nan), where=a > 0)
    result['spectral_prediction'] = {'wavelength_um': w.tolist(), 'planet_star_contrast': [float(v) if np.isfinite(v) else None for v in ratio],
        'sampling_note': 'Plotting samples; integration uses the full piecewise-linear native/corner-union representation.'}
    result['energy_diagnostics'] = {}
    for body, s, row in (('star', ss, sr), ('planet', ps, pr)):
        t = row['parameters'].get('teff_K')
        if t and s.metadata['construction'] == 'native_exact':
            integral = float(np.sum(np.diff(s.wavelength_um)*(s.surface_flux[:-1]+s.surface_flux[1:])/2))
            result['energy_diagnostics'][body] = {'native_coverage_flux_over_sigma_Teff4': integral/(5.670374419e-8*t**4),
                'coverage_um': s.metadata['wavelength_range_um'], 'interpretation': 'Coverage-limited diagnostic, not a normalization correction or accuracy certificate.'}
        else:result['energy_diagnostics'][body] = {'not_computed': 'Reconstruction is limited to the response band; no bolometric diagnostic is inferred.'}
    return result


def predict_request(request, base):
    """JSON request facade with explicit paths; contains no fixed atmosphere."""
    allowed = {'description', 'catalog', 'star', 'companion', 'response', 'distance_pc', 'detector', 'samples'}
    if not isinstance(request, dict) or set(request)-allowed:raise ValueError('Unknown request fields: '+str(set(request)-allowed if isinstance(request, dict) else type(request)))
    result = predict_from_properties(Path(base)/request['catalog'], request['star'], request['companion'], request['response'], base=base,
        distance_pc=request.get('distance_pc'), detector=request.get('detector', 'photon'), samples=request.get('samples', 301))
    result['request'] = copy.deepcopy(request)
    return result


def predict_many(catalog, star, companions, response, **options):
    if not companions:raise ValueError('Supply at least one companion specification.')
    library = ModelLibrary(catalog)
    return [predict_from_properties(catalog, star, body, response, library=library, **options) for body in companions]


def radius_at_contrast(prediction, contrast):
    contrast = float(contrast)
    if not np.isfinite(contrast) or contrast <= 0:raise ValueError('Contrast must be a positive linear flux ratio.')
    return {'radius_m': float(prediction['rplanet_m']*np.sqrt(contrast/prediction['planet_star_contrast'])),
            'scope': 'Conditional radius at fixed surface atmosphere. Not a mass/age fit, completeness or confidence bound.'}


def predict_separation(catalog, star, companion, response, separation_arcsec, distance_pc, *, heating_options=None, **options):
    """Fixed atmosphere or explicit energy-budget heating with bounded resolution.

    Heating substitutes an isolated atmosphere at effective temperature. It is
    not an irradiated-atmosphere calculation or a reflected-light model.
    """
    from .separation import heating
    x = np.asarray(separation_arcsec, float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or np.any(x <= 0) or np.any(np.diff(x) <= 0):raise ValueError('Provide increasing positive separations.')
    if not np.isfinite(distance_pc) or distance_pc <= 0:raise ValueError('Distance must be positive.')
    library = ModelLibrary(catalog);predictions = []
    if heating_options is None:
        result = predict_from_properties(catalog, star, companion, response, distance_pc=distance_pc, library=library, **options)
        predictions = [result]*len(x)
        effective = [result['catalog_selections']['planet']['parameters']['teff_K']]*len(x)
    else:
        if set(heating_options)-{'intrinsic_teff_K', 'bond_albedo', 'true_to_projected'}:raise ValueError('Unknown heating option.')
        first = predict_from_properties(catalog, star, companion, response, distance_pc=distance_pc, library=library, **options)
        st = first['catalog_selections']['star']['parameters']['teff_K']
        intrinsic = float(heating_options['intrinsic_teff_K'])
        if not _same(intrinsic, companion['parameters'].get('teff_K')):raise ValueError('Companion starting teff_K must equal intrinsic_teff_K.')
        _, true_au, teq, effective = heating(x, distance_pc, first['rstar_m'], st, intrinsic,
            heating_options.get('bond_albedo', .3), heating_options.get('true_to_projected', 1.))
        for temperature in effective:
            selected = copy.deepcopy(companion);selected['parameters']['teff_K'] = float(temperature)
            predictions.append(predict_from_properties(catalog, star, selected, response, distance_pc=distance_pc, library=library, **options))
    return {'separation_arcsec': x.tolist(), 'projected_separation_au': (x*distance_pc).tolist(),
        'effective_temperature_K': list(map(float, effective)), 'contrast': [p['planet_star_contrast'] for p in predictions],
        'predictions': predictions, 'heating_options': heating_options,
        'scope': 'Model brightness versus projected separation; no observed sensitivity is generated. Heating, when requested, is an energy-budget approximation using isolated atmospheres. Albedo is Bond albedo; no reflected light or self-consistent irradiation/chemistry solution.'}

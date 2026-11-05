"""Generalized star/companion property entry backed by the public catalog API."""
from pathlib import Path
from datetime import datetime
import copy
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from atmosphere_compare.properties import ModelLibrary, predict_request, predict_separation, response_from_request, radius_at_contrast
from model_library_ui import jobs
from prediction_exports import model_access, export_result
from contrast_series_ui import render as render_series


def _body(library, role):
    prefix = 'general_'+role
    rows = [r for r in library.rows if r['role'] == role]
    if not rows:raise ValueError('No '+role+' entries in this catalog.')
    families = sorted({r['family'] for r in rows})
    family = st.selectbox(('Stellar' if role == 'star' else 'Companion')+' model family', families, key=prefix+'_family')
    rows = [r for r in rows if r['family'] == family]
    # All controls come from this family; no model family/property is fixed here.
    params = {}
    labels = {'teff_K': 'Temperature [K]', 'logg_cgs': 'log g [cgs]', 'metallicity_dex': 'Metallicity [dex]', 'co_relative_solar': 'C/O relative to solar'}
    for key in sorted({k for row in rows for k in row['parameters']}):
        values = [r['parameters'][key] for r in rows if key in r['parameters']]
        unique = sorted({json.dumps(v) for v in values}, key=lambda v: (0, float(json.loads(v))) if isinstance(json.loads(v), (int, float)) else (1, v))
        widget_key = prefix+'_'+family+'_'+key
        if key in ('teff_K', 'logg_cgs') or (key == 'metallicity_dex' and family == 'ck04_atlas9_stsci'):
            params[key] = st.number_input(('Star ' if role == 'star' else 'Companion ')+labels[key], value=float(json.loads(unique[0])), key=widget_key)
        else:
            params[key] = json.loads(st.selectbox(('Star ' if role == 'star' else 'Companion ')+labels.get(key, key), unique, format_func=lambda v: str(json.loads(v)), key=widget_key))
    mode = st.selectbox(('Star' if role == 'star' else 'Companion')+' model selection', ['exact', 'linear_surface_flux'],
                        format_func=lambda v: 'Exact native model' if v == 'exact' else 'Bounded temperature/gravity interpolation', key=prefix+'_mode')
    value = st.number_input(('Star' if role == 'star' else 'Companion')+' radius', min_value=.000001, value=1., key=prefix+'_radius')
    units = ['R_sun', 'R_jup', 'R_earth', 'm']
    unit = st.selectbox(('Star' if role == 'star' else 'Companion')+' radius unit', units, index=0 if role == 'star' else 1, key=prefix+'_runit')
    return {'family': family, 'parameters': params, 'radius': {'value': value, 'unit': unit}, 'selection': mode}


def _fingerprint(request, library, root):
    tokens = [request, library.sha256]
    families = [('star', request['star']['family']), ('planet', request['companion']['family'])]
    for row in library.rows:
        if (row['role'], row['family']) in families:
            for key in ('file', 'wavelength_file'):
                if key not in row['spectrum']:continue
                p = (library.path.parent/Path(row['spectrum'][key]).expanduser()).resolve()
                stat = p.stat() if p.is_file() else None
                tokens.append([str(p), stat.st_size if stat else None, stat.st_mtime_ns if stat else None])
    mapping = request['response']
    paths = [Path(root)/Path(mapping[k]['file']).expanduser() for k in ('filter', 'sky')]
    for p in paths:
        stat = p.stat() if p.is_file() else None;tokens.append([str(p), stat.st_size if stat else None, stat.st_mtime_ns if stat else None])
    return hashlib.sha256(json.dumps(tokens, sort_keys=True).encode()).hexdigest()


def render(root):
    root = Path(root)
    with st.expander('Predict from stellar and companion properties', expanded=False):
        st.write('Choose model families and physical properties directly. The same request can be used in the app, PyCharm runner or Python API.')
        try:
            catalog = Path(st.text_input('Property API catalog', str(root/'configs/app_catalog.json'), key='general_catalog')).expanduser().resolve()
            library = ModelLibrary(catalog)
            with st.expander('Registered parameter ranges and installed files'):
                st.json(library.inventory())
            left, right = st.columns(2)
            with left:star = _body(library, 'star')
            with right:companion = _body(library, 'planet')
            distance = st.number_input('Common distance [pc]', min_value=.000001, value=10., key='general_distance')
            response_text = st.text_area('Filter and sky response JSON', (root/'templates/response_demo.json').read_text(), key='general_response')
            st.caption('The default uses a synthetic response for demonstration. For science, supply your own filter and sky CSVs with explicit units, wavelength convention and limits_um. Both components and spectra must cover the entire band.')
            request = {'catalog': str(catalog), 'star': star, 'companion': companion, 'response': json.loads(response_text),
                       'distance_pc': distance, 'detector': 'photon', 'samples': 301}
            st.caption('Interpolation is an explicit linear surface-flux blend within a complete cataloged temperature/gravity cell. Other parameters stay fixed; no nearest-node substitution or extrapolation is used.')
            manager = jobs(str(root))
            if star['family'] == 'ck04_atlas9_stsci':
                if st.button('Install requested native CK04 star', disabled=manager.status()['running'], key='general_install_star'):
                    manager.start('ck04_star', {k: star['parameters'][k] for k in ('teff_K', 'logg_cgs', 'metallicity_dex')});st.rerun()
                st.caption('This installs one actual native node. For an intermediate temperature, install its native bracketing nodes and choose interpolation. A nonexistent node stops explicitly.')
                state = manager.status()
                if state['name'] == 'ck04_star':
                    st.code(state['log'], language=None)
                    if st.button('Refresh property model choices', key='general_refresh'):st.rerun()
            if st.button('Save these inputs for the PyCharm request runner', key='general_save_request'):
                path = root/'configs/atmosphere_request.json';path.parent.mkdir(parents=True, exist_ok=True)
                if path.is_file():shutil.copy2(path, path.with_name('atmosphere_request.before_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json'))
                path.write_text(json.dumps(request, indent=2));st.success('Saved: '+str(path))
            signature = _fingerprint(request, library, root)
            current = st.session_state.get('general_prediction')
            if current and current['signature'] != signature:
                st.session_state.pop('general_prediction', None);st.session_state.pop('general_export', None)
                st.info('Inputs or source files changed. Calculate again to refresh the result.')
            if st.button('Predict from properties', disabled=manager.status()['running'], key='general_predict'):
                st.session_state.pop('general_prediction', None);st.session_state.pop('general_export', None)
                with st.spinner('Resolving native models and integrating their spectra…'), model_access(root):
                    result = predict_request(request, root)
                    if signature != _fingerprint(request, library, root):raise ValueError('Source files changed during calculation; calculate again.')
                    st.session_state['general_prediction'] = {'signature': signature, 'result': result}
            current = st.session_state.get('general_prediction')
            if not current:return
            result = current['result'];a, b = st.columns(2)
            interpolated=[body for body,source in result.get('source_spectra',{}).items() if source.get('band_limited_reconstruction')]
            if interpolated:
                st.warning('Interpolated '+', '.join(interpolated)+' spectrum: this is a surface-flux blend, not a new atmosphere solution. Accuracy depends on the grid, band and cell; cold and sparse cells can have substantial errors. Exact native selection bypasses this interpolation.')
                with st.expander('Interpolation source nodes and weights'):
                    st.json({body:result['source_spectra'][body].get('interpolation',{}) for body in interpolated})
            a.metric('Predicted band contrast', '%.6g' % result['planet_star_contrast']);b.metric('Delta magnitude', '%.4f' % result['delta_magnitude'])
            st.json({'resolved_parameters': {body: row['parameters'] for body, row in result['catalog_selections'].items()}, 'sampling': result['sampling_counts']})
            fig = go.Figure(go.Scatter(x=result['spectral_prediction']['wavelength_um'], y=result['spectral_prediction']['planet_star_contrast'], name='Spectral contrast'))
            fig.add_hline(y=result['planet_star_contrast'], line_dash='dash', annotation_text='Band contrast')
            fig.update_layout(xaxis_title='Wavelength [µm]', yaxis_title='Companion / star contrast', yaxis_type='log')
            st.plotly_chart(fig, use_container_width=True)
            if st.button('Export property prediction', key='general_export_button'):
                output = export_result(result, root/'results'/('property_prediction_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
                st.session_state['general_export'] = {'signature': signature, 'path': str(output), 'bytes': output.read_bytes()}
            exported = st.session_state.get('general_export')
            if exported and exported['signature'] == signature:
                st.download_button('Download property result', exported['bytes'], Path(exported['path']).name, 'application/zip', key='general_download')
            render_series(root, result, prefix='property')
            with st.expander('Separation, heating and a conditional brightness limit'):
                lower = st.number_input('Minimum separation [arcsec]', min_value=.00001, value=.1, key='general_sep_min')
                upper = st.number_input('Maximum separation [arcsec]', min_value=.00001, value=2., key='general_sep_max')
                count = int(st.number_input('Separation samples', min_value=2, max_value=200, value=12, key='general_sep_count'))
                heated = st.checkbox('Include the stellar energy-budget heating approximation', key='general_heating')
                albedo = st.number_input('Bond albedo', min_value=0., max_value=1., value=.3, key='general_albedo')
                geometry = st.number_input('True / projected separation ratio', min_value=1., value=1., key='general_geometry')
                st.caption('Heating combines intrinsic and equilibrium temperatures in fourth powers and selects isolated atmospheres at the resulting temperature. It does not calculate self-consistent irradiated spectra or reflected light. Use bounded interpolation for intermediate temperatures.')
                if st.button('Calculate model contrast versus separation', key='general_separation'):
                    if lower >= upper:raise ValueError('Minimum separation must be below maximum.')
                    heat = {'intrinsic_teff_K': companion['parameters']['teff_K'], 'bond_albedo': albedo, 'true_to_projected': geometry} if heated else None
                    with model_access(root):
                        curve = predict_separation(catalog, star, companion, request['response'], np.geomspace(lower, upper, count), distance,
                            heating_options=heat, base=root, samples=31)
                    st.line_chart(pd.DataFrame({'separation_arcsec': curve['separation_arcsec'], 'model_contrast': curve['contrast']}).set_index('separation_arcsec'))
                    st.download_button('Download model separation calculation', json.dumps(curve, indent=2, allow_nan=False), 'model_separation.json', 'application/json', key='general_curve_download')
                limit = st.number_input('Measured contrast or sensitivity boundary', min_value=1e-12, value=.001, format='%.8f', key='general_limit')
                st.json(radius_at_contrast(result, limit))
        except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
            st.session_state.pop('general_prediction', None);st.session_state.pop('general_export', None)
            st.error('Property prediction could not run: '+str(exc))

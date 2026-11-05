"""Model downloads and registration inside the app, before a spectrum run exists."""
from pathlib import Path
import json
import zipfile
import pandas as pd
import streamlit as st
from atmosphere_compare.model_library import ModelJob,register_model,register_folder
from atmosphere_compare.example import load_validation
from validation_ui import render_report

ADAPTERS=['phoenix_hires','sonora_bobcat','sonora_cholla','sonora_diamondback','sonora_flame_skimmer_v1_r3k','sonora_elf_owl_v2','atmo2020_ascii','atmo2020_stsci_fits','bt_settl_svo','bt_settl_stsci_fits','ascii']


@st.cache_resource
def jobs(root):
    return ModelJob(root)


def reset_catalog_controls():
    for key in list(st.session_state):
        if key.startswith(('star_param_','planet_param_')) or key in ('star_family','planet_family','sep_temperatures'):
            st.session_state.pop(key,None)


def render(root):
    root=Path(root)
    with st.expander('Model library - downloads and imports',expanded=False):
        downloads,imports,bulk,status,validation=st.tabs(['Download models','Import a model','Import a grid folder','Available models','Validate installed models'])
        with downloads:
            st.write('Install official spectra on this laptop and reuse the model archive cache.')
            manager=jobs(str(root))
            @st.fragment(run_every='2s')
            def progress():
                state=manager.status()
                left,right=st.columns(2)
                if left.button('Install PHOENIX + Bobcat starter models',disabled=state['running'],key='install_starter'):
                    try:manager.start('starter');st.rerun()
                    except (OSError,ValueError) as exc:st.error(str(exc))
                if right.button('Add Bobcat temperature sequence',disabled=state['running'],key='install_sequence'):
                    try:manager.start('temperature_sequence');st.rerun()
                    except (OSError,ValueError) as exc:st.error(str(exc))
                a,b=st.columns(2)
                if a.button('Install Bobcat mass/age tracks (~1 MB)',disabled=state['running'],key='install_evolution'):
                    try:manager.start('evolution');st.rerun()
                    except (OSError,ValueError) as exc:st.error(str(exc))
                if b.button('Add Bobcat temperature + gravity grid',disabled=state['running'],key='install_gravity'):
                    try:manager.start('gravity_grid');st.rerun()
                    except (OSError,ValueError) as exc:st.error(str(exc))
                if st.button('Install Cholla chemistry grid (443 MB archive)',disabled=state['running'],key='install_cholla'):
                    try:manager.start('cholla');st.rerun()
                    except (OSError,ValueError) as exc:st.error(str(exc))
                if st.button('Install ATMO 2020 equilibrium spectra (STScI)',disabled=state['running'],key='install_atmo'):
                    try:manager.start('atmo');st.rerun()
                    except (OSError,ValueError) as exc:st.error(str(exc))
                a,b=st.columns(2)
                dt=a.selectbox('Diamondback subset temperature [K]',list(range(900,2401,100)),index=1,key='diamondback_temperature')
                dg=a.selectbox('Diamondback subset gravity [m/s²]',[31,100,316,1000,3160],index=1,key='diamondback_gravity')
                ft=b.selectbox('Flame Skimmer subset temperature [K]',[50,100,150,200,250,300,350,400,450,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2200,2400],index=14,key='flame_temperature')
                fg=b.selectbox('Flame Skimmer subset gravity [m/s²]',[1,3.1,10,31,100,316,1000,3160],index=4,key='flame_gravity')
                if a.button('Add Diamondback clouds + metallicity',disabled=state['running'],key='install_diamondback'):
                    try:manager.start('diamondback_parameters',{'temperature':dt,'gravity':dg});st.rerun()
                    except (OSError,ValueError) as exc:st.error(str(exc))
                if b.button('Add Flame Skimmer metallicity + C/O + mixing',disabled=state['running'],key='install_flame'):
                    try:manager.start('flame_parameters',{'temperature':ft,'gravity':fg});st.rerun()
                    except (OSError,ValueError) as exc:st.error(str(exc))
                st.caption('Each button adds models at the temperature and gravity selected above; defaults are 1000 K, g=100 m/s² (log g=4.0). Repeat at other native coordinates to extend your library. Diamondback: eight models, about 44 MB transfer / 100 MB storage at the default. Flame Skimmer v1 R3000: eleven models, about 7 MB including index. Sizes appear in the log. Missing native subsets stop explicitly. The multi-gigabyte ZIP archives are not downloaded.')
                st.caption('The STScI ATMO subset has log g=5.0. The Bobcat gravity extension selects available g=31–1000 m/s² nodes (approximately log g=3.5–5.0), including overlap for matched model comparisons. Actual installed cells determine the choices.')
                st.caption('Bobcat selections retain reported header temperatures/gravity. Five verified files report temperatures 1 K above their filename labels. Native connectivity is recorded separately, so interpolation does not invent missing rectangular nodes or rename those spectra.')
                st.caption('Cholla adds cloud-free disequilibrium models with log Kzz controls. Extracted spectra need additional disk space. Other native grid folders can be registered in Import a grid folder.')
                state=manager.status()
                st.caption('Starter selection: PHOENIX 5000 K, log g 4.5, solar composition; Bobcat log g 4.0, solar composition. These are selectable starting models. First archive transfer is about 1.20 GB.')
                if state['running']:
                    st.info('Model installation is running. Progress refreshes automatically; the plots remain available.')
                    if st.button('Stop model installation',key='stop_install'):
                        manager.stop();st.rerun()
                elif state['started']:
                    if state['cancelled']:st.info('Installation stopped. Downloaded partial files are retained for a later retry.')
                    elif state['exit_code']==0:
                        st.success('Checks finished. Refresh the report in Validate installed models.' if state['name'] in ('validation','software_checks') else 'Models installed. Refresh the model selectors to use the new entries.')
                    else:st.error('Installation stopped with an error. Details are in the progress log below.')
                    if st.button('Refresh model selectors',key='refresh_models'):
                        reset_catalog_controls();st.rerun()
                if state['log']:
                    with st.expander('Installation progress log',expanded=state['running']):st.code(state['log'],language=None)
            progress()
        with imports:
            st.write('Register a local spectrum in the model library. Additional families become selectable after import.')
            role=st.selectbox('Spectrum role',['star','planet'],key='import_role')
            family=st.text_input('Model family name',key='import_family')
            kind=st.selectbox('Spectrum adapter',ADAPTERS,key='import_adapter')
            file=st.text_input('Local spectrum file path',key='import_file')
            wave_file=st.text_input('Separate wavelength FITS path (PHOENIX)',key='import_wave_file') if kind=='phoenix_hires' else None
            frame=st.selectbox('Imported wavelength convention',['vacuum','air'],key='import_frame')
            if kind=='sonora_elf_owl_v2':st.caption('Use files from the corrected v2 release. The file content does not independently prove the release version; the adapter choice is recorded as a declaration.')
            if kind=='bt_settl_svo':st.caption('This adapter accepts SVO BT-Settl CIFIST emission VOTables with air wavelengths. It does not accept raw logarithmic .spec.7 files.')
            parameters=st.text_area('Atmospheric parameters JSON','{}',key='import_parameters')
            st.caption('Use the grid\'s parameter names and conventions: teff_K, logg_cgs, metallicity_dex, C/O, cloud or chemistry parameters. Native headers fill verified parameters; ASCII parameters are declarations.')
            mapping=st.text_area('ASCII unit/column settings JSON','{"wavelength_unit":"um","flux_unit":"W_m2_um","flux_convention":"surface_flux","skiprows":0}',key='import_mapping') if kind=='ascii' else None
            replace=st.checkbox('Replace this exact role/family/parameter selection',key='import_replace')
            if st.button('Add spectrum to model library',key='import_submit',disabled=jobs(str(root)).status()['running']):
                try:
                    params=json.loads(parameters)
                    spec={'format':kind,'file':file,'wavelength_frame':frame,'label':family.strip() or 'Atmosphere'}
                    if kind=='phoenix_hires':spec['wavelength_file']=wave_file
                    elif kind=='ascii':
                        extra=json.loads(mapping)
                        if not isinstance(extra,dict):raise ValueError('Unit/column settings must be an object.')
                        spec.update(extra)
                        spec.update(format=kind,file=file,wavelength_frame=frame)
                    row,meta=register_model(root,role,family,spec,params,replace)
                    st.success('Model registered. Refresh selectors to choose it.')
                    st.json({'family':row['family'],'parameters':row['parameters'],'header_verified':meta['parameters_verified_from_header']})
                except (OSError,ValueError,KeyError,TypeError,ImportError) as exc:st.error('Model import could not run: '+str(exc))
            if st.button('Refresh imported model selectors',key='refresh_imports'):
                reset_catalog_controls();st.rerun()
        with bulk:
            st.write('Import native spectra from an extracted grid folder. All selected files are validated before the catalog changes.')
            folder=st.text_input('Extracted grid folder',key='grid_import_folder')
            pattern=st.text_input('File pattern','*.gz',key='grid_import_pattern')
            adapter=st.selectbox('Grid-folder adapter',ADAPTERS,key='grid_import_adapter')
            role=st.selectbox('Grid-folder role',['planet','star'],key='grid_import_role')
            family=st.text_input('Grid-folder family',key='grid_import_family')
            frame=st.selectbox('Grid-folder wavelengths',['vacuum','air'],key='grid_import_frame')
            params=st.text_area('Common parameters JSON','{}',key='grid_import_parameters')
            settings=st.text_area('Additional spectrum settings JSON','{}',key='grid_import_settings')
            st.caption('Native headers/filenames supply each temperature and gravity. Additional settings can include wavelength_file for PHOENIX, source_url, or explicit ASCII units. Keep different cloud/chemistry/composition cases identified.')
            replace=st.checkbox('Replace exact grid selections',key='grid_import_replace')
            if st.button('Validate and register grid folder',key='grid_import_submit',disabled=jobs(str(root)).status()['running']):
                try:
                    count=register_folder(root,role,family,folder,pattern,adapter,json.loads(params),dict(json.loads(settings),wavelength_frame=frame),replace)
                    st.success('%d native spectra registered. Refresh selectors.'%count)
                except (OSError,ValueError,KeyError,TypeError,ImportError) as exc:st.error('Grid import could not run: '+str(exc))
            st.link_button('ATMO 2020 official downloads','https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/atmo-2020-atmosphere-models')
            st.link_button('Sonora Cholla official archive','https://zenodo.org/records/4450269')
            st.link_button('Sonora Diamondback official archive','https://zenodo.org/records/12735103')
            st.link_button('Sonora Flame Skimmer v1 R3000 quickstart','https://zenodo.org/records/20030439')
            st.link_button('Sonora Elf Owl v2','https://zenodo.org/records/15150874')
            st.link_button('BT-Settl SVO catalog','https://svo2.cab.inta-csic.es/theory/newov2/index.php')
        with status:
            path=root/'configs/app_catalog.json'
            try:
                rows=json.loads(path.read_text()) if path.exists() else []
                st.dataframe(pd.DataFrame([{'Role':r.get('role'),'Family':r['family'],'Parameters':str(r['parameters']),
                    'Installed':(path.parent/r['spectrum']['file']).is_file()} for r in rows]),width='stretch')
            except (OSError,ValueError,KeyError,TypeError) as exc:st.error('Catalog could not be read: '+str(exc))
            st.caption('Additional adapters cover Cholla, Diamondback emission text, original ATMO 2020 emission ASCII, STScI ATMO FITS and SVO BT-Settl CIFIST emission VOTables. Each adapter is release-specific; synthetic reader tests do not validate every real model release.')
        with validation:
            st.write('Run diagnostics on this laptop using the installed model catalog and the observing responses you calculated in the app.')
            current=st.session_state.get('current');bands=[] if not current else [b for b in current['report']['filter_results'] if 'response' in b]
            st.caption('Checks native file loading, recorded units and parameters, band coverage, coverage-limited bolometric flux and leave-one-temperature-out band interpolation. This does not validate a self-consistent irradiation solver.')
            state=jobs(str(root)).status()
            @st.fragment(run_every='2s')
            def diagnostic_progress():
                manager=jobs(str(root));status=manager.status()
                if status['name'] not in ('validation','software_checks'):return
                if status['running']:
                    st.info('Checks are running on this laptop. Progress refreshes automatically.')
                    if st.button('Stop checks',key='stop_validation'):
                        manager.stop();st.rerun()
                elif status['cancelled']:st.info('Checks stopped. The displayed report may be from an earlier run.')
                elif status['exit_code']==0:st.success('Checks finished. Click Refresh validation report to display the results.')
                else:st.error('Checks ended with an error. Inspect the log and refresh the report for any recorded load errors.')
                if status['log']:
                    with st.expander('Diagnostic progress log',expanded=status['running']):st.code(status['log'],language=None)
            diagnostic_progress()
            if st.button('Run synthetic software checks',disabled=state['running'],key='validate_software'):
                try:jobs(str(root)).start('software_checks');st.rerun()
                except (OSError,ValueError) as exc:st.error(str(exc))
            if st.button('Run installed-model diagnostics',disabled=state['running'],key='validate_models'):
                try:
                    catalogs=[str(root/'configs/app_catalog.json')]
                    if current:
                        catalogs=[m['catalog_path'] for m in current['report']['settings']['model_sources'].values() if m.get('catalog_path')] or catalogs
                    request=root/'configs/validation_request.json';request.parent.mkdir(exist_ok=True)
                    request.write_text(json.dumps({'catalogs':catalogs,'bands':bands},indent=2))
                    jobs(str(root)).start('validation');st.rerun()
                except (OSError,ValueError,KeyError,TypeError) as exc:st.error(str(exc))
            latest=root/'results/latest_validation.json'
            if st.button('Refresh validation report',key='validation_refresh'):st.rerun()
            if latest.is_file():
                try:
                    record=json.loads(latest.read_text());st.json(record)
                    loaded=load_validation(root)
                    render_report(loaded['report'],key='library_diagnostics')
                    output=Path(loaded['zip_path'])
                    if output.is_file():st.download_button('Download installed-model validation ZIP',output.read_bytes(),output.name,'application/zip',key='validation_zip')
                except (OSError,ValueError,KeyError,TypeError,zipfile.BadZipFile) as exc:st.error('Validation report could not be loaded: '+str(exc))
            software=root/'results/latest_software_checks.json'
            if software.is_file():
                try:
                    record=json.loads(software.read_text());st.json(record['summary']);output=Path(record['path'])
                    if output.is_file():st.download_button('Download software-check ZIP',output.read_bytes(),output.name,'application/zip',key='software_validation_zip')
                except (OSError,ValueError,KeyError) as exc:st.error('Software-check report could not be loaded: '+str(exc))
            if not bands:st.info('For band-interpolation tests, run a comparison and calculate at least one response in Filters & limits first. File and energy diagnostics can run without a response.')

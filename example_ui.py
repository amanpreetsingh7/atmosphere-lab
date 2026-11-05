"""A guided example on selected spectra, with results and diagnostics in one ZIP."""
from pathlib import Path
import io
import json
import zipfile
import pandas as pd
import streamlit as st
from atmosphere_compare.example import run_example,load_validation,validation_coverage
from atmosphere_compare.compare import RADII_M
from atmosphere_compare.separation import export_separation
from atmosphere_compare.workbench import export_bundle
from validation_ui import render_report


def configure_separation(item,index):
    st.session_state.update(sep_band='Calculated filter',sep_calculated_band=index,sep_sensitivity_source='None',sep_evolution=False)


def render(root,item,native):
    st.write('Use your selected spectrum files to calculate a band brightness and check the numerical scaling. Run the installed-model diagnostics on this laptop, then include their results in the example export.')
    st.markdown('1. Select your star and planet, set their radii/distance, and **Run comparison**.\n2. In **Filters & limits**, calculate a supplied response or explicitly choose an **Idealized top-hat (method check)**.\n3. In **Model library → Validate installed models**, run diagnostics and refresh the report.\n4. Run this example, then build a separation plot through the same response.')
    bands=[b for b in item['report']['filter_results'] if 'response' in b]
    if not bands:st.info('Calculate a response in Filters & limits first.');return
    index=st.selectbox('Worked-example response',range(len(bands)),format_func=lambda i:bands[i]['name'],key='example_band')
    band=bands[index]
    if band.get('source',{}).get('kind')=='idealized_top_hat':st.caption('This response is an explicitly idealized method check. It is not an instrument calibration or an observation.')
    st.button('Use this response in separation tab',on_click=configure_separation,args=(item,index),key='example_separation')
    st.caption('Open Contrast vs separation and Build separation plot. You can select temperature/radius sequences there; heating is an explicit optional approximation.')
    if st.button('Run worked example and consistency checks',key='example_run',type='primary'):
        st.session_state.pop('example_result',None);st.session_state.pop('example_export',None)
        try:
            with st.spinner('Checking the selected spectra on this laptop…'):result=run_example(item,native,band)
            st.session_state.example_result=result
        except (OSError,ValueError,TypeError,KeyError,ImportError) as exc:st.error('Example could not run: '+str(exc))
    result=st.session_state.get('example_result')
    if not result or result['source_comparison_id']!=item['id']:return
    st.caption('Showing the last calculated example through '+result['selected_band']['name']+'. Run again after changing the response.')
    metrics=st.columns(3);prediction=result['band_prediction']
    metrics[0].metric('Predicted band contrast','%.6g'%prediction['planet_star_contrast'])
    metrics[1].metric('Predicted planet brightness [band-equivalent mJy]','%.6g'%prediction['planet_band_equivalent_mJy'])
    metrics[2].metric('Numerical checks passed','%d / %d'%(sum(r['passed'] for r in result['checks']),len(result['checks'])))
    st.dataframe(pd.DataFrame(result['checks']),width='stretch');st.caption(result['interpretation'])
    validation=None;coverage=None
    try:
        validation=load_validation(root)
        if validation:
            coverage=validation_coverage(validation,native,result['selected_band'])
            st.write('Latest diagnostic coverage:',{k:v for k,v in coverage.items() if k!='meaning'})
            if not all(coverage[k] for k in ('star','planet','response')):st.warning('The latest report does not cover every current example input. Re-run installed-model diagnostics with this response before treating it as validation of this example.')
            with st.expander('Latest installed-model diagnostic results'):render_report(validation['report'],key='example_diagnostics')
        else:st.info('Installed-model diagnostics have not been recorded for this project. Run them in Model library; the example can still demonstrate numerical scaling.')
    except (OSError,ValueError,KeyError,TypeError,zipfile.BadZipFile) as exc:st.error('Diagnostic report could not be read: '+str(exc))
    separation=st.session_state.get('separation_result')
    if separation and separation.get('source_comparison_id')!=item['id']:separation=None
    if separation:
        same_response=separation['bandpass']['wavelength_um']==result['selected_band']['response']['wavelength_um'] and separation['bandpass']['throughput']==result['selected_band']['response']['throughput'] and separation['bandpass']['detector']==result['selected_band']['detector'] and separation['bandpass']['wavelength_frame']==result['selected_band']['wavelength_frame']
        if not same_response:separation=None;st.info('The last separation plot uses another response, so it will not be added to this example.')
    token=json.dumps([result,validation['report_sha256'] if validation else None,coverage,separation],sort_keys=True)
    if st.button('Prepare worked-example ZIP',key='example_prepare'):
        extras={}
        if separation:extras['separation']=separation
        out=io.BytesIO()
        with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('example.json',json.dumps(dict(result,diagnostic_coverage=coverage),indent=2,allow_nan=False))
            archive.writestr('comparison_and_figures.zip',export_bundle(item,[],extras))
            archive.writestr('numerical_checks.csv',pd.DataFrame(result['checks']).to_csv(index=False))
            if validation:archive.writestr('installed_model_validation.json',json.dumps(validation['report'],indent=2,allow_nan=False))
            if separation:archive.writestr('separation_curves_and_figures.zip',export_separation(separation))
            archive.writestr('README.txt','This is a worked prediction using selected spectrum files, not an observed detection.\nThe response may be idealized; see its recorded source.\nMatching diagnostics are indicated by diagnostic_coverage, not by a blanket physical pass.\nOriginal model files are not bundled; retain their hashes and release provenance.\n')
        st.session_state.example_export={'token':token,'zip':out.getvalue()}
    prepared=st.session_state.get('example_export')
    if prepared and prepared['token']==token:st.download_button('Download worked-example ZIP',prepared['zip'],'atmosphere_worked_example.zip','application/zip',key='example_zip')

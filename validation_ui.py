"""Readable laptop diagnostic results, preserving missing and failed checks."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def interpolation_summary(temperatures,gravities):
    """Summarize measured withheld-model errors without assigning a pass cutoff."""
    rows=[]
    families=sorted({r['family'] for r in temperatures+gravities})
    for family in families:
        row={'family':family}
        for label,tests in (('temperature',temperatures),('gravity',gravities)):
            group=[r for r in tests if r['family']==family]
            row[label+' checks']=len(group)
            row['largest absolute '+label+' error [%]']=max(
                (100*abs(float(r['fractional_error'])) for r in group),default=None)
        rows.append(row)
    return rows


def render_report(result,key='validation'):
    if result.get('software_version'):
        st.caption('Diagnostic software version: '+result['software_version']+' · Python '+result.get('python_version','unrecorded'))
    else:
        st.caption('Historical diagnostic report: software version was not recorded. Rerun diagnostics for a current versioned report.')
    reports=result.get('reports',[])
    records=[r for p in reports for r in p.get('records',[])]
    errors=[r for p in reports for r in p.get('load_errors',[])]
    temperatures=[r for p in reports for r in p.get('heldout_temperature_band_tests',[])]
    gravities=[r for p in reports for r in p.get('heldout_gravity_band_tests',[])]
    unsupported_gravity=[r for p in reports for r in p.get('unsupported_gravity_holdouts',[])]
    unsupported=[{'family':r['family'],'parameters':str(r['parameters']),'band':name,'reason':b['unsupported']} for r in records for name,b in r.get('bands',{}).items() if 'unsupported' in b]
    columns=st.columns(4)
    for c,label,value in zip(columns,('Loaded spectra','Load errors','Temperature checks','Gravity checks'),(len(records),len(errors),len(temperatures),len(gravities))):c.metric(label,value)
    if errors:st.error('Some source files could not be loaded. These files were not validated.');st.dataframe(pd.DataFrame([{'model':str(e.get('model',{})),'error':e['error']} for e in errors]),width='stretch')
    if unsupported:st.warning('Some spectra do not support the selected response.');st.dataframe(pd.DataFrame(unsupported),width='stretch')
    if unsupported_gravity:
        st.warning('Some gravity-track holdouts were unsupported; they are not passed tests.')
        st.dataframe(pd.DataFrame(unsupported_gravity),width='stretch')
    if not temperatures:st.info('No temperature holdout tests ran. Each test needs at least three supported temperatures at the same gravity/composition and a calculated response.')
    if not gravities:st.info('No gravity holdout tests ran. Each test needs at least three supported gravity tracks with compatible temperature coverage/composition and a calculated response.')
    summary=interpolation_summary(temperatures,gravities)
    if summary:
        st.subheader('Measured interpolation limits')
        st.dataframe(pd.DataFrame(summary),width='stretch')
        largest=max(abs(float(r['fractional_error'])) for r in temperatures+gravities)*100
        st.info('The largest withheld-model difference in this report is %.2f%%. These tests remove native models and predict them from remaining neighbors. This is a measured error on the tested subset, not an uncertainty bound for every interpolated query.'%largest)
        st.caption('Exact native-model comparisons use the source spectrum directly. For interpolated fits or separation curves, inspect the errors in the relevant band and parameter range before interpreting precision. No universal accuracy threshold is assumed.')
    for axis,tests in (('omitted_temperature_K',temperatures),('omitted_logg_cgs',gravities)):
        if not tests:continue
        fig=go.Figure()
        for family in sorted({r['family'] for r in tests}):
            group=[r for r in tests if r['family']==family]
            fig.add_trace(go.Scatter(x=[r[axis] for r in group],y=[100*r['fractional_error'] for r in group],mode='markers',name=family,
                text=[r['band']+'; '+str(r['held_parameters']) for r in group]))
        fig.add_hline(y=0,line_dash='dot');fig.update_layout(template='plotly_white',xaxis_title='Omitted temperature [K]' if axis.endswith('_K') else 'Omitted log g [cgs]',yaxis_title='Interpolated/native signal − 1 [%]')
        st.plotly_chart(fig,use_container_width=True,key=key+'_'+axis)
        st.dataframe(pd.DataFrame(tests),width='stretch')
    energy=[{'family':r['family'],'role':r.get('role'),'parameters':str(r['parameters']),'native wavelength range [µm]':str(r['metadata'].get('wavelength_range_um')),
             'native flux / σ Teff⁴':r['native_coverage_flux_over_sigma_Teff4']} for r in records if 'native_coverage_flux_over_sigma_Teff4' in r]
    if energy:
        st.subheader('Native integrated-flux comparison')
        summaries=[]
        for family in sorted({r['family'] for r in energy}):
            group=[r['native flux / σ Teff⁴'] for r in energy if r['family']==family]
            summaries.append({'family':family,'spectra':len(group),
                'minimum difference from σ Teff⁴ [%]':100*(min(group)-1),
                'maximum difference from σ Teff⁴ [%]':100*(max(group)-1)})
        st.dataframe(pd.DataFrame(summaries),width='stretch')
        largest_ratio=max(r['native flux / σ Teff⁴'] for r in energy)
        if largest_ratio>1:
            st.info('The largest native integral is %.2f%% above nominal σ Teff⁴. Missing wavelength tails cannot explain an excess. This diagnostic alone does not establish its cause; preserve the supplied normalization while investigating.'%(100*(largest_ratio-1)))
        st.caption('Only the native wavelength coverage is integrated. A deficit can depend on omitted wavelength regions as well as the supplied spectrum; this is not a complete bolometric measurement. The displayed differences are diagnostics, not pass/fail tolerances.')
        with st.expander('Coverage-limited energy diagnostics'):st.dataframe(pd.DataFrame(energy),width='stretch')
    st.caption('The energy ratio depends on native wavelength coverage. Interpolation errors require scientific review; neither completed status nor zero load errors establishes physical accuracy.')

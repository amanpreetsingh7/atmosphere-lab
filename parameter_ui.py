"""In-app controlled cloud, chemistry and composition comparisons."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from atmosphere_compare.parameter_exploration import PARAMETER_NAMES,ORDER,value_order,value_label,candidates,explore,export_bundle


def render(item,native):
    st.write('Compare exact native models while changing one atmosphere parameter. The current stellar spectrum and both radii stay fixed.')
    path_text=item['report']['settings'].get('model_sources',{}).get('planet',{}).get('catalog_path')
    if not path_text:
        st.info('Select a companion from the model catalog and run a comparison first.');return
    try:
        path=Path(path_text);content=path.read_bytes();all_rows=json.loads(content)
        rows=[r for r in all_rows if r.get('role','planet')=='planet']
        families=sorted({r['family'] for r in rows})
        if not families:st.info('Install or import a companion grid in Model library.');return
        family=st.selectbox('Parameter comparison grid',families,key='parameter_family')
        subset=[r for r in rows if r['family']==family]
        keys=set(k for r in subset for k in r['parameters'])
        axes=[k for k in keys if k!='chemistry' and len({json.dumps(r['parameters'].get(k)) for r in subset if k in r['parameters']})>1]
        axes.sort(key=lambda k:(k not in ('fsed','metallicity_dex','co_relative_solar','logKzz_cgs'),ORDER.index(k) if k in ORDER else 100,k))
        if not axes:st.info('This installed subset has one native selection. Add models with different parameter values.');return
        axis=st.selectbox('Parameter to vary',axes,format_func=lambda k:PARAMETER_NAMES.get(k,k),key='parameter_axis_'+family)
        remaining=[r for r in subset if axis in r['parameters']]
        other=sorted(set(k for r in remaining for k in r['parameters'])-{axis},key=lambda k:(ORDER.index(k) if k in ORDER else 100,k))
        held={}
        preferred={'teff_K':1000,'logg_cgs':4,'metallicity_dex':0,'co_relative_solar':1,'chemistry':'equilibrium','fsed':2,'logKzz_cgs':4}
        source=item['report']['planet']
        if source.get('family')==family:preferred.update(source.get('parameters_declared',{}))
        # A sparse starter grid is a set of intersecting parameter sequences.
        # Start with a group that actually has multiple values for this axis.
        groups={}
        for row in remaining:
            fixed={k:v for k,v in row['parameters'].items() if k!=axis}
            identity=json.dumps(fixed,sort_keys=True)
            groups.setdefault(identity,[]).append(row['parameters'][axis])
        best=max(groups,key=lambda key:(len({json.dumps(v) for v in groups[key]}),sum(preferred.get(k)==v for k,v in json.loads(key).items())))
        preferred.update(json.loads(best))
        with st.expander('Fixed atmosphere parameters',expanded=True):
            for key in other:
                options=sorted({json.dumps(r['parameters'].get(key)) for r in remaining},key=lambda v:value_order(json.loads(v)))
                default=next((i for i,v in enumerate(options) if json.loads(v)==preferred.get(key)),0)
                encoded=st.selectbox(PARAMETER_NAMES.get(key,key),options,index=default,format_func=lambda v:value_label(json.loads(v)) if json.loads(v) is not None else 'not applicable',key='parameter_hold_'+family+'_'+axis+'_'+key)
                value=json.loads(encoded)
                if value is not None:held[key]=value
                remaining=[r for r in remaining if r['parameters'].get(key)==value]
        matched=candidates(rows,family,axis,held)
        values=sorted({json.dumps(r['parameters'][axis]) for r in matched},key=lambda v:value_order(json.loads(v)))
        chosen=st.multiselect('Native values to compare',values,default=values[:12],format_func=lambda v:value_label(json.loads(v)),key='parameter_values_'+family+'_'+axis+'_'+hashlib.sha256(json.dumps(held,sort_keys=True).encode()).hexdigest()[:12])
        selected=[r for r in matched if json.dumps(r['parameters'][axis]) in chosen]
        report=item['report']
        bands=report['filter_results']
        include_bands=st.checkbox('Include responses calculated in Filters & limits',value=True,key='parameter_include_bands')
        responses=bands if include_bands else []
        request={'source_id':item['id'],'catalog_sha256':hashlib.sha256(content).hexdigest(),'family':family,'axis':axis,'held':held,'values':chosen,'responses':responses}
        signature=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        # Old output is not exported after a control, source or response change.
        old=st.session_state.get('parameter_result')
        if old and old.get('request_signature')!=signature:st.session_state.pop('parameter_result',None);st.session_state.pop('parameter_export',None)
        st.caption('C/O controls use each grid’s published convention. Sonora C/O here is relative to solar (1 = solar). Cloud-free is discrete; f_sed is a sedimentation parameter, not a measured cloud fraction. Use saved comparisons for comparisons between different families.')
        if len(selected)<2:st.info('At least two native values are needed at the same fixed settings. Choose another parameter or add native models.')
        if st.button('Compare native parameter values',key='parameter_build',disabled=not 2<=len(selected)<=12):
            st.session_state.pop('parameter_result',None);st.session_state.pop('parameter_export',None)
            with st.spinner('Loading the selected native models on this laptop…'):
                result=explore(native[0],selected,path.parent,axis,report['rstar_m'],report['rplanet_m'],report['bin_resolving_power'],report['wavelength_range_um'],responses)
            result.update(source_comparison_id=item['id'],request_signature=signature,catalog_path=str(path),catalog_sha256=request['catalog_sha256'])
            st.session_state.parameter_result=result
        result=st.session_state.get('parameter_result')
        if not result:return
        st.caption('Shared plotted coverage: %.6g–%.6g µm. All curves use identical wavelength bins; full native coverage is required for each response.'%tuple(result['shared_wavelength_range_um']))
        view=st.radio('Parameter plot',['Companion / star contrast','Companion surface spectrum'],horizontal=True,key='parameter_plot_view')
        quantity='planet_star_contrast' if view.startswith('Companion /') else 'planet_surface_flambda_W_m2_um'
        fig=go.Figure()
        for curve in result['curves']:
            data=curve['result'];fig.add_trace(go.Scatter(x=data['wavelength_um'],y=data[quantity],name=curve['label'],mode='lines'))
        fig.update_layout(height=480,template='plotly_dark',xaxis={'type':'log','title':'Wavelength [µm]'},yaxis={'type':'log','title':view},legend_title=result['parameter_label'])
        st.plotly_chart(fig,use_container_width=True,key='parameter_plot')
        if result['band_rows']:
            table=pd.DataFrame(result['band_rows'])
            if any(isinstance(value,str) for value in table['parameter_value']):
                table['parameter_value']=table['parameter_value'].map(value_label)
            st.dataframe(table,width='stretch')
        else:st.caption('Calculate a response in Filters & limits, then repeat this comparison to add native band-integrated predictions.')
        if st.button('Prepare parameter figures and results',key='parameter_prepare'):
            with st.spinner('Preparing the figures and reproducibility records…'):st.session_state.parameter_export={'signature':signature,'zip':export_bundle(result)}
        bundle=st.session_state.get('parameter_export')
        if bundle and bundle['signature']==signature:st.download_button('Download parameter comparison ZIP',bundle['zip'],'atmosphere_parameter_comparison.zip','application/zip',key='parameter_download')
        with st.expander('Fixed settings and native model provenance'):st.json({k:v for k,v in result.items() if k!='curves'});st.json([{'value':c['value'],'spectrum':c['spectrum']} for c in result['curves']])
    except (OSError,ValueError,KeyError,TypeError,ImportError) as exc:
        st.session_state.pop('parameter_result',None);st.session_state.pop('parameter_export',None)
        st.error('Parameter comparison could not run: '+str(exc))

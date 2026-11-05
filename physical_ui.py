"""Band-contrast trends, conditional physical constraints and evolutionary inputs."""
import io
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from atmosphere_compare.compare import band_signal,equivalent_flux_density_Jy,RADII_M
from atmosphere_compare.separation import catalog_band_sequence
from atmosphere_compare.evolution import read_table,read_bobcat_tracks,evaluate
from inference_ui import render as render_inference
from atmosphere_compare.workbench import physical_figure_bytes


def render(item,native):
    trends,evolution,inference=st.tabs(['Brightness and sensitivity','Mass and age','Measurements and uncertainties'])
    report=item['report'];star,planet=native
    with trends:
        st.write('Translate a band brightness or sensitivity threshold into conditional planet properties for the selected atmosphere sequence.')
        bands=[b for b in report['filter_results'] if 'response' in b]
        if not bands:
            st.info('Calculate an observing response in Filters & limits first.')
        else:
            index=st.selectbox('Physical-properties response',list(range(len(bands))),format_func=lambda i:bands[i]['name'],key='physical_band')
            radii_text=st.text_input('Radii to compare [R_jup], comma separated','1,1.5',key='physical_radii')
            kind=st.radio('Brightness input',['Model predictions','Measured contrast','Contrast sensitivity threshold'],key='physical_kind')
            observed=st.number_input('Planet/star contrast',min_value=1e-15,value=1e-5,format='%.5e',key='physical_observed',disabled=kind=='Model predictions')
            if st.button('Calculate physical-properties plots',key='physical_build'):
                st.session_state.pop('physical_result',None)
                try:
                    band=bands[index];response=band['response'];w,t=response['wavelength_um'],response['throughput'];detector=band['detector']
                    if planet.metadata.get('catalog_path'):
                        temperatures,signals,source=catalog_band_sequence(planet.metadata,w,t,detector)
                    else:
                        temperature=(planet.metadata.get('parameters_verified_from_header') or planet.metadata.get('parameters_declared',{})).get('teff_K')
                        if temperature is None:raise ValueError('Declare a planet temperature or use a local catalog sequence.')
                        temperatures,signals=np.array([temperature]),np.array([band_signal(planet,w,t,detector)])
                        source={'spectra':[planet.metadata],'held_parameters':planet.metadata.get('parameters_declared',{})}
                    radii=[float(v.strip()) for v in radii_text.split(',')]
                    if not radii or len(radii)>10 or any(not np.isfinite(r) or r<=0 for r in radii):raise ValueError('Use up to ten positive radii.')
                    stellar=band_signal(star,w,t,detector);rs=report['rstar_m']
                    if stellar<=0 or np.any(signals<=0):raise ValueError('Positive stellar and planet band signals are required for physical-property conversion.')
                    jy=equivalent_flux_density_Jy(star,rs,report['distance_pc'],w,t,detector)
                    rows=[]
                    for temp,signal in zip(temperatures,signals):
                        for radius in radii:
                            contrast=float(signal/stellar*(radius*RADII_M['R_jup']/rs)**2)
                            row={'temperature_K':float(temp),'radius_Rjup':radius,'planet_star_contrast':contrast,'band_equivalent_mJy':contrast*jy*1000}
                            if kind!='Model predictions':
                                row['radius_to_match_input_Rjup']=float(rs*np.sqrt(observed*stellar/signal)/RADII_M['R_jup'])
                                row['model_above_input']=bool(contrast>=observed)
                            rows.append(row)
                    result={'source_comparison_id':item['id'],'band':band,'kind':kind,'input_contrast':observed if kind!='Model predictions' else None,
                        'rows':rows,'temperature_sequence':source,'model_stellar_equivalent_Jy':jy,
                        'interpretation':'Conditional radius at fixed atmospheric parameters. A sensitivity threshold is a brightness boundary, not a confidence bound, completeness or occurrence exclusion.'}
                    st.session_state.physical_result=result
                except (OSError,ValueError,TypeError,KeyError,ImportError) as exc:st.error('Physical-properties calculation could not run: '+str(exc))
            result=st.session_state.get('physical_result')
            if result and result['source_comparison_id']==item['id']:
                st.caption('Showing the last calculated result through '+result['band']['name']+'. Apply changed inputs with Calculate physical-properties plots.')
                frame=pd.DataFrame(result['rows'])
                xaxis=st.radio('Physical-property plot axis',['Temperature','Radius'],horizontal=True,key='physical_axis')
                fig=go.Figure()
                if xaxis=='Temperature':
                    for radius,group in frame.groupby('radius_Rjup'):
                        fig.add_trace(go.Scatter(x=group['temperature_K'],y=group['planet_star_contrast'],mode='lines+markers',name='%g Rjup'%radius))
                else:
                    for temp,group in frame.groupby('temperature_K'):
                        group=group.sort_values('radius_Rjup')
                        fig.add_trace(go.Scatter(x=group['radius_Rjup'],y=group['planet_star_contrast'],mode='lines+markers',name='%g K'%temp))
                if result['input_contrast']:fig.add_hline(y=result['input_contrast'],line_dash='dash',annotation_text=result['kind'])
                fig.update_layout(xaxis_title='Planet temperature [K]' if xaxis=='Temperature' else 'Planet radius [R_jup]',yaxis_title='Band planet/star contrast',yaxis_type='log',template='plotly_white')
                st.plotly_chart(fig,use_container_width=True,key='temperature_contrast_plot')
                st.dataframe(frame,width='stretch')
                st.caption(result['interpretation'])
                st.download_button('Physical-properties CSV',frame.to_csv(index=False).encode(),'physical_properties.csv','text/csv',key='physical_csv')
                st.download_button('Physical-properties report',json.dumps(result,indent=2).encode(),'physical_properties.json','application/json',key='physical_json')
                if st.button('Prepare physical-properties figures',key='physical_prepare'):
                    st.session_state.physical_figures={'source_id':item['id'],'result_json':json.dumps(result,sort_keys=True),
                        'png':physical_figure_bytes(result),'pdf':physical_figure_bytes(result,'pdf')}
                prepared=st.session_state.get('physical_figures')
                if prepared and prepared['source_id']==item['id'] and prepared['result_json']==json.dumps(result,sort_keys=True):
                    st.download_button('Temperature/radius PNG',prepared['png'],'temperature_and_radius.png','image/png',key='physical_png')
                    st.download_button('Temperature/radius PDF',prepared['pdf'],'temperature_and_radius.pdf','application/pdf',key='physical_pdf')
    with evolution:
        st.write('Use a verified evolutionary table to link a fixed planet mass and age to intrinsic temperature and radius.')
        from pathlib import Path
        installed=sorted((Path(__file__).parent/'models/evolution').glob('nc*_co1.0_mass'))
        selected_track=st.selectbox('Installed evolutionary tracks',['Upload / local file']+[str(p) for p in installed],format_func=lambda p:Path(p).name,key='evolution_installed')
        adapter=st.selectbox('Evolution file format',['Explicit-unit CSV','Native Bobcat mass tracks'],key='evolution_adapter')
        file=st.file_uploader('Evolution file',key='evolution_file')
        st.caption('Columns: mass_Mjup, age_Myr, intrinsic_temperature_K, radius_Rjup. Use temperatures for non-irradiated intrinsic evolution. No evolutionary data are invented.')
        path=st.text_input('Or local evolution CSV path',key='evolution_path')
        content=None;name=None
        try:
            if selected_track!='Upload / local file':
                p=Path(selected_track);content,name=p.read_bytes(),p.name
            elif file is not None:content,name=file.getvalue(),file.name
            elif path.strip():
                p=Path(path).expanduser();content,name=p.read_bytes(),p.name
            if content:
                if selected_track!='Upload / local file' or adapter=='Native Bobcat mass tracks':
                    rows,meta=read_bobcat_tracks(content,name,'https://zenodo.org/records/5063476' if selected_track!='Upload / local file' else None)
                    import re
                    match=re.fullmatch(r'nc([+-][\d.]+)_co1\.0_mass',name)
                    if match:meta.update(metallicity_dex=float(match.group(1)),co_relative_solar=1.)
                else:rows,meta=read_table(content,name)
                st.caption(meta['source_validation'])
                masses=[r['mass_Mjup'] for r in rows];ages=[r['age_Myr'] for r in rows]
                for key,values in (('evolution_mass',masses),('evolution_age',ages)):
                    if key in st.session_state and not min(values)<=st.session_state[key]<=max(values):st.session_state.pop(key,None)
                mass=st.number_input('Planet mass [M_jup]',min_value=float(min(masses)),max_value=float(max(masses)),value=float(min(masses)),key='evolution_mass')
                age=st.number_input('Planet age [Myr]',min_value=float(min(ages)),max_value=float(max(ages)),value=float(min(ages)),key='evolution_age')
                if st.button('Evaluate mass and age',key='evolution_build'):
                    st.session_state.pop('evolution_planet',None)
                    st.session_state.pop('evolution_table',None)
                    point=evaluate(rows,mass,age,meta)
                    st.session_state.evolution_table={'rows':rows,'metadata':meta}
                    st.session_state.evolution_planet=point
            record=st.session_state.get('evolution_planet')
            if record:
                st.json(record)
                st.caption('Use this planet in Contrast vs separation. Temperature and gravity are bracketed in the installed atmosphere grid; composition compatibility is checked. The displayed result is the last evaluated table selection.')
                st.download_button('Evolution planet report',json.dumps(record,indent=2).encode(),'evolution_planet.json','application/json',key='evolution_json')
        except (OSError,ValueError,TypeError,KeyError,UnicodeError) as exc:st.error('Evolution table could not be used: '+str(exc))
    with inference:render_inference(item,native)

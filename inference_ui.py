"""Measurement-error inference entirely inside the physical-properties app tab."""
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from atmosphere_compare.compare import band_signal,RADII_M
from atmosphere_compare.grid import catalog_band_grid
from atmosphere_compare.inference import likelihood_grid
from atmosphere_compare.workbench import inference_figure_bytes


def render(item,native):
    bands=[b for b in item['report']['filter_results'] if 'response' in b]
    st.write('Compare measured band contrasts with atmosphere predictions, including measurement errors. Results are conditional likelihoods over the tested parameters.')
    if not bands:st.info('Calculate the observing responses in Filters & limits first.');return
    if len({b['name'] for b in bands})!=len(bands):st.error('Use distinct response names for multi-band inference.');return
    observations=st.data_editor(pd.DataFrame([{'band':b['name'],'kind':'detection','contrast':1e-5,'sigma':1e-6} for b in bands]),
        column_config={'band':st.column_config.TextColumn(disabled=True),'kind':st.column_config.SelectboxColumn(options=['detection','upper_limit']),
        'contrast':st.column_config.NumberColumn(format='%.5e'),'sigma':st.column_config.NumberColumn(format='%.5e',min_value=1e-15)},
        hide_index=True,key='inference_observations')
    st.caption('sigma is the one-sigma measurement noise in contrast. upper_limit means the noisy measurement was below the supplied threshold. An injection-recovery sensitivity curve alone does not supply this statistical model.')
    route=st.radio('Inference parameters',['Temperature and radius','Mass and age from loaded evolution tracks'],key='inference_route')
    radii=st.text_input('Trial radii [R_jup], comma separated','0.7,1,1.3,1.6',key='inference_radii',disabled=route!='Temperature and radius')
    extra_error=st.number_input('Declared fractional model error (independent per band)',min_value=0.,value=0.,max_value=1.,key='inference_model_error')
    covariance=st.text_area('Optional detection covariance matrix JSON (contrast squared)','',key='inference_covariance')
    if st.button('Calculate measurement likelihoods',key='inference_build'):
        st.session_state.pop('inference_result',None)
        try:
            star,planet=native;grids=[];star_signals=[]
            for band in bands:
                response=band['response'];w,t=response['wavelength_um'],response['throughput']
                if response.get('wavelength_frame',band.get('wavelength_frame','vacuum'))!=planet.metadata.get('wavelength_frame','vacuum'):raise ValueError('Response and atmosphere wavelength conventions differ.')
                grids.append(catalog_band_grid(planet.metadata,w,t,band['detector']))
                star_signals.append(band_signal(star,w,t,band['detector']))
            rs=item['report']['rstar_m'];rows=[];predictions=[];skipped=[]
            if route=='Temperature and radius':
                radius_values=[float(v.strip()) for v in radii.split(',')]
                if not radius_values or len(radius_values)>50 or any(not np.isfinite(r) or r<=0 for r in radius_values):raise ValueError('Use up to fifty positive radii.')
                gravity=float(planet.metadata['parameters_declared']['logg_cgs'])
                trials=[{'temperature_K':float(temp),'logg_cgs':gravity,'radius_Rjup':radius} for temp in grids[0].temperatures_at_gravity(gravity) for radius in radius_values]
            else:
                table=st.session_state.get('evolution_table')
                if not table:raise ValueError('Load and evaluate an evolution table in Mass and age first.')
                meta=table['metadata'];held=grids[0].metadata['held_parameters']
                for key in ('metallicity_dex','co_relative_solar'):
                    if key in meta and (key not in held or not np.isclose(float(meta[key]),float(held[key]),atol=1e-6)):raise ValueError('Evolution and atmosphere '+key+' must match.')
                from atmosphere_compare.evolution import G,MJUP_KG
                trials=[{'mass_Mjup':r['mass_Mjup'],'age_Myr':r['age_Myr'],'temperature_K':r['intrinsic_temperature_K'],
                         'radius_Rjup':r['radius_Rjup'],'logg_cgs':float(np.log10(G*r['mass_Mjup']*MJUP_KG/(r['radius_Rjup']*RADII_M['R_jup'])**2*100))} for r in table['rows']]
            if len(trials)>20000:raise ValueError('Use at most 20,000 trial points per run.')
            for trial in trials:
                try:
                    signals=[g.evaluate({'teff_K':trial['temperature_K'],'logg_cgs':trial['logg_cgs']})[0] for g in grids]
                    contrast=[(trial['radius_Rjup']*RADII_M['R_jup']/rs)**2*s/f for s,f in zip(signals,star_signals)]
                    predictions.append(contrast);rows.append(dict(trial,**{'contrast_'+b['name']:c for b,c in zip(bands,contrast)}))
                except ValueError as exc:skipped.append({'parameters':trial,'reason':str(exc)})
            likelihood=likelihood_grid(predictions,observations.to_dict('records'),json.loads(covariance) if covariance.strip() else None,extra_error)
            for i,row in enumerate(rows):row.update(log_likelihood=likelihood['log_likelihood'][i],relative_likelihood=likelihood['relative_likelihood'][i])
            result={'source_comparison_id':item['id'],'route':route,'rows':rows,'skipped_outside_supported_grid':skipped,
                'observations':observations.to_dict('records'),'covariance':json.loads(covariance) if covariance.strip() else None,
                'fractional_model_error':extra_error,'likelihood':{k:v for k,v in likelihood.items() if k not in ('log_likelihood','relative_likelihood')},
                'bands':bands,'grids':[g.metadata for g in grids],
                'evolution_source':st.session_state['evolution_table']['metadata'] if route!='Temperature and radius' else None}
            json.dumps(result,allow_nan=False);st.session_state.inference_result=result
        except (ValueError,OSError,KeyError,TypeError,ImportError,np.linalg.LinAlgError) as exc:st.error('Likelihood calculation could not run: '+str(exc))
    result=st.session_state.get('inference_result')
    if result and result['source_comparison_id']==item['id']:
        frame=pd.DataFrame(result['rows']);mass='mass_Mjup' in frame
        x,y=('mass_Mjup','age_Myr') if mass else ('temperature_K','radius_Rjup')
        fig=go.Figure(go.Scatter(x=frame[x],y=frame[y],mode='markers',marker={'color':frame['relative_likelihood'],'colorscale':'Viridis','showscale':True,'colorbar':{'title':'Relative likelihood'}},text=frame['relative_likelihood']))
        fig.update_layout(xaxis_title=x,yaxis_title=y,template='plotly_white')
        if mass:fig.update_xaxes(type='log');fig.update_yaxes(type='log')
        st.plotly_chart(fig,use_container_width=True,key='inference_plot')
        st.write('Best tested point:',result['rows'][result['likelihood']['best_model_index']])
        st.caption(result['likelihood']['interpretation'])
        st.caption('%d tested points supported; %d omitted outside complete installed cells. Showing the last calculated inputs.'%(len(frame),len(result['skipped_outside_supported_grid'])))
        st.dataframe(frame,width='stretch')
        st.download_button('Likelihood grid CSV',frame.to_csv(index=False).encode(),'measurement_likelihoods.csv','text/csv',key='inference_csv')
        st.download_button('Likelihood provenance JSON',json.dumps(result,indent=2).encode(),'measurement_likelihoods.json','application/json',key='inference_json')
        if st.button('Prepare likelihood figures',key='inference_prepare'):
            st.session_state.inference_figures={'result':json.dumps(result,sort_keys=True),'png':inference_figure_bytes(result),'pdf':inference_figure_bytes(result,'pdf')}
        prepared=st.session_state.get('inference_figures')
        if prepared and prepared['result']==json.dumps(result,sort_keys=True):
            st.download_button('Likelihood PNG',prepared['png'],'measurement_likelihoods.png','image/png',key='inference_png')
            st.download_button('Likelihood PDF',prepared['pdf'],'measurement_likelihoods.pdf','application/pdf',key='inference_pdf')

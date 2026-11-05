"""Manual native-series calculation, live plots and portable figure export."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from atmosphere_compare.properties import ModelLibrary
from atmosphere_compare.contrast_series import available_nodes,source_signature,calculate_series,calculate_heated_series,export_series
from atmosphere_compare.heated_series import heated_csv_rows
from atmosphere_compare.compare import RADII_M
from prediction_exports import model_access


def render(root,prediction,prefix):
    key=lambda name:prefix+'_series_'+name
    with st.expander('Theoretical contrast curves - temperature and radius'):
        st.caption('Calculate from installed native spectra using the current star, observing response and companion family.')
        try:
            inputs=prediction['property_inputs'];star=inputs['star'];companion=inputs['companion']
            catalog=Path(prediction['selection_provenance']['catalog'])
            library=ModelLibrary(catalog)
            response=prediction['request']['response']
            nodes=available_nodes(library,companion)
            if not nodes:
                st.info('No exact temperature sequence exists at the current companion gravity/composition. Select an installed native gravity and fixed parameter combination first. No gravity snapping is applied.')
                return
            st.json({'companion_family':companion['family'],'held_parameters':{k:v for k,v in companion['parameters'].items() if k!='teff_K'}})
            ts=sorted(nodes)
            heated=st.checkbox('Include stellar heating in separation curves',key=key('heated'))
            defaults=[t for t in (200.,300.,500.,1000.,1500.,2000.) if t in ts] or ts[:min(5,len(ts))]
            all_nodes=st.checkbox('Use all matching native temperatures',disabled=heated,key=key('all'))
            selected=st.multiselect('Intrinsic temperatures Tmin [K]' if heated else 'Native companion temperatures [K]',ts,default=defaults,disabled=all_nodes and not heated,key=key('temperatures'))
            unit=st.selectbox('Series radius unit',['R_jup','R_earth','R_sun','m'],key=key('unit'))
            radius=prediction['rplanet_m']/RADII_M[unit]
            text=st.text_input('Companion radii (comma separated, maximum four)','%g,%g,%g'%(radius*.8,radius,radius*1.2),key=key('radii_'+unit))
            low=st.number_input('Series display: minimum separation [arcsec]',min_value=.00001,value=.1,key=key('lo'))
            high=st.number_input('Series display: maximum separation [arcsec]',min_value=.00002,value=1.,key=key('hi'))
            heat_options={}
            if heated:
                distance=st.number_input('Heating distance [pc]',min_value=.000001,value=float(prediction['distance_pc']),key=key('distance'))
                albedo=st.number_input('Heating Bond albedo',min_value=0.,max_value=1.,value=.3,key=key('albedo'))
                geometry=st.number_input('Heating true / projected separation ratio',min_value=1.,value=1.,key=key('geometry'))
                heat_options=dict(distance_pc=distance,bond_albedo=albedo,true_to_projected=geometry,separations=(low,high),samples=301)
                st.caption('Tmin is the temperature without stellar heating. Teff^4 = Tmin^4 + Teq^4, with full heat redistribution. The full matching native temperature sequence supplies bounded log band-contrast interpolation at heated Teff. Fixed gravity/composition; no extrapolation.')
                st.warning('Approximate heating using isolated atmospheres, not a self-consistent irradiated atmosphere. Interpolation errors depend on the grid and band. Unsupported heated temperatures leave gaps; no reflected light or measured sensitivity is included.')
            radii=[float(v.strip()) for v in text.split(',')]
            chosen=ts if all_nodes and not heated else selected
            if not 0<low<high:raise ValueError('Minimum display separation must be below maximum.')
            if not 1<=len(radii)<=4:raise ValueError('Choose one to four radii.')
            if heated and not 1<=len(chosen)<=8:raise ValueError('Choose one to eight Tmin values for a readable heated plot.')
            source=source_signature(library,response,root)
            identity=hashlib.sha256(json.dumps([prediction,chosen,radii,unit,low,high,source,heated,heat_options],sort_keys=True).encode()).hexdigest()
            current=st.session_state.get(key('result'))
            if current and current['identity']!=identity:
                st.session_state.pop(key('result'),None);st.info('Series inputs or model files changed. Calculate the series again.')
            if st.button('Calculate native contrast series',key=key('calculate')):
                st.session_state.pop(key('result'),None)
                progress=st.progress(0.,text='Integrating installed spectra on this computer')
                with model_access(root):
                    calculator=calculate_heated_series if heated else calculate_series
                    data=calculator(catalog,star,companion,response,chosen,radii,radius_unit=unit,base=root,**heat_options,
                        progress=lambda done,total:progress.progress(done/total,text='Calculated %d / %d native temperature nodes'%(done,total)))
                    archive=export_series(root,data,(low,high))
                progress.empty()
                st.session_state[key('result')]={'identity':identity,'data':data,'archive':archive.read_bytes(),'name':archive.name,'path':str(archive)}
            current=st.session_state.get(key('result'))
            if not current or current['identity']!=identity:return
            data=current['data'];table=pd.DataFrame(data['rows'])
            fig=go.Figure()
            for r in sorted(table.radius_value.unique()):
                points=table[table.radius_value==r]
                fig.add_trace(go.Scatter(x=points.temperature_K,y=points.contrast,mode='lines+markers',name='%g %s'%(r,unit)))
            fig.update_layout(template='plotly_white',title='Theoretical band contrast versus temperature',xaxis_title='Companion effective temperature [K]',yaxis_title='Band-integrated companion / star contrast',yaxis_type='log')
            st.plotly_chart(fig,use_container_width=True,key=key('temperature_plot'))
            separation=go.Figure()
            styles=['dot','solid','dash','dashdot']
            if heated:
                h=data['heated_separation'];temps=sorted({c['Tmin_K'] for c in h['curves']});rs=sorted({c['radius_value'] for c in h['curves']})
                colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f']
                for curve in h['curves']:
                    i,j=temps.index(curve['Tmin_K']),rs.index(curve['radius_value'])
                    separation.add_trace(go.Scatter(x=h['separation_arcsec'],y=curve['contrast'],mode='lines',connectgaps=False,
                        name='Tmin=%g K, %g %s'%(curve['Tmin_K'],curve['radius_value'],unit),line={'color':colors[i],'dash':styles[j]},
                        customdata=np.column_stack([curve['effective_temperature_K'],h['equilibrium_temperature_K']]),
                        hovertemplate='Separation: %{x:.4g} arcsec<br>Contrast: %{y:.6g}<br>Heated Teff: %{customdata[0]:.2f} K<br>Equilibrium temperature: %{customdata[1]:.2f} K<extra>%{fullData.name}</extra>'))
                    if curve['omitted_out_of_grid_samples']:
                        st.warning('Tmin=%g K, %g %s: %d heated samples exceed the grid and are omitted.'%(curve['Tmin_K'],curve['radius_value'],unit,curve['omitted_out_of_grid_samples']))
                settings=h['settings']
                st.caption('Heating assumptions: distance %g pc; Bond albedo %g; true/projected separation %g; full-surface heat redistribution.'%(settings['distance_pc'],settings['bond_albedo'],settings['true_to_projected_ratio']))
            else:
                temperatures=sorted(table.temperature_K.unique())
                displayed=[temperatures[i] for i in np.unique(np.linspace(0,len(temperatures)-1,min(5,len(temperatures))).round().astype(int))]
                for t in displayed:
                    points=table[table.temperature_K==t].sort_values('radius_value')
                    for i,row in enumerate(points.itertuples()):
                        separation.add_trace(go.Scatter(x=[low,high],y=[row.contrast,row.contrast],mode='lines',
                            name='%g K, %g %s'%(t,row.radius_value,unit),line={'color':'dodgerblue','dash':styles[i%4]}))
            separation.update_layout(template='plotly_white',height=620,title='Approximate stellar heating - curves labeled by Tmin' if heated else 'Fixed atmosphere contrasts versus separation',
                xaxis_title='Angular separation [arcsec]',yaxis_title='Band-integrated companion / star contrast',yaxis_type='log',legend={'font':{'size':9}})
            st.plotly_chart(separation,use_container_width=True,key=key('separation_plot'))
            if heated:
                with st.expander('Heated contrasts and temperatures'):
                    st.dataframe(pd.DataFrame(list(heated_csv_rows(data))),hide_index=True)
                import io,zipfile
                with zipfile.ZipFile(io.BytesIO(current['archive'])) as package:
                    columns=st.columns(3)
                    for column,label,filename,mime in zip(columns,['Heated figure PNG','Heated figure PDF','Heated numerical CSV'],
                            ['03_contrast_vs_separation_heated.png','03_contrast_vs_separation_heated.pdf','heated_contrasts.csv'],['image/png','application/pdf','text/csv']):
                        column.download_button(label,package.read(filename),filename,mime,key=key('heated_download_'+mime))
            st.dataframe(table,hide_index=True)
            st.caption('The temperature plot and native table are unheated exact-node calculations. Radii are imposed at fixed gravity; these are not fixed-mass or fixed-age evolution tracks. Heated exports add a separate Tmin figure and a table of separation, Teq, Teff and contrast.' if heated else 'Companion nodes are native exact selections. Connecting lines are visual guides. Radii are imposed at fixed gravity; these are not fixed-mass or fixed-age evolution tracks. The export also includes a compact separation display without irradiation or measured sensitivity.')
            st.download_button('Download series and figures',current['archive'],current['name'],'application/zip',key=key('download'))
            st.caption('Saved: '+current['path'])
        except (OSError,ValueError,KeyError,TypeError,ImportError,RuntimeError) as exc:
            st.session_state.pop(key('result'),None);st.error('Contrast series could not run: '+str(exc))

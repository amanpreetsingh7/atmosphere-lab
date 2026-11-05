"""General atmosphere-based contrast versus separation controls."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import io
import json
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from atmosphere_compare import __version__
from atmosphere_compare.compare import band_signal, band_contrast, equivalent_flux_density_Jy, RADII_M
from atmosphere_compare.grid import catalog_band_grid
from atmosphere_compare.separation import (load_sensitivity, heating, catalog_band_sequence,
    interpolate_band_signal, finite_list, export_separation, plot_bytes, simulated_sensitivity)


def available_temperatures(metadata):
    path = Path(metadata.get("catalog_path", ""))
    if not path.is_file():
        return []
    held = {k: v for k, v in metadata.get("parameters_declared", {}).items() if k != "teff_K"}
    rows = json.loads(path.read_text())
    return sorted({float(row["parameters"]["teff_K"]) for row in rows
        if row.get("role", "planet") == "planet" and row.get("family") == metadata.get("family")
        and {k: v for k, v in row["parameters"].items() if k != "teff_K"} == held
        and (path.parent/row["spectrum"]["file"]).is_file()})


def test_controls(item,metadata):
    lo,hi=item['report']['wavelength_range_um']
    a,b=(3.5,4.0) if lo<=3.5<4<=hi else (lo*(hi/lo)**.35,lo*(hi/lo)**.65)
    st.session_state.update(sep_band='Custom top-hat',sep_band_lower=float(a),sep_band_upper=float(b),
        sep_sensitivity_source='Simulated sensitivity (layout test)',sep_evolution=False,sep_use_jy=True)
    nodes=available_temperatures(metadata)
    if len(nodes)>1:
        n=min(5,len(nodes)-1)
        chosen=[nodes[i] for i in np.unique(np.round(np.linspace(0,len(nodes)-2,n)).astype(int))]
        st.session_state.update(sep_route='Catalog temperature sequence',sep_temperatures=chosen,sep_heating=True)
        radius=item['report']['rplanet_m']/RADII_M['R_jup']
        st.session_state['sep_radii_'+item['id']]='%g,%g'%(radius,1.5*radius)


def render(item, native):
    st.write("Generate band-integrated atmosphere predictions versus separation, optionally overlaid on observed sensitivity.")
    report = item["report"]
    star, planet = native
    st.button('Set up an atmosphere plot test',on_click=test_controls,args=(item,planet.metadata),key='sep_test_setup')
    st.caption('Uses your selected atmospheres and available temperature nodes. The optional test response and simulated sensitivity are labeled; stellar inputs remain yours.')
    left, right = st.columns(2)
    with left:
        route = st.selectbox("Planet predictions", ["Selected planet atmosphere", "Catalog temperature sequence"], key="sep_route")
        evolved=st.session_state.get('evolution_planet')
        use_evolution=st.checkbox('Use the evaluated mass/age planet',disabled=evolved is None,key='sep_evolution')
        if use_evolution:
            route='Catalog temperature sequence'
            st.caption('Evolution input: %g Mjup, %g Myr, %g K intrinsic, %g Rjup.'%(evolved['mass_Mjup'],evolved['age_Myr'],evolved['intrinsic_temperature_K'],evolved['radius_Rjup']))
        target = st.text_input("Target label", "Target", key="sep_target")
        lower = st.number_input("Minimum angular separation [arcsec]", min_value=.00001, value=.05, key="sep_lower")
        upper = st.number_input("Maximum angular separation [arcsec]", min_value=.00002, value=5.0, key="sep_upper")
        use_heating = st.checkbox("Apply approximate stellar heating", key="sep_heating")
        albedo = st.number_input("Bond albedo", min_value=0.0, max_value=1.0, value=.3, disabled=not use_heating, key="sep_albedo")
        geometry = st.number_input("True distance / projected separation", min_value=1.0, value=1.0, disabled=not use_heating, key="sep_geometry")
        selected_temperatures = []
        if route == "Catalog temperature sequence":
            nodes = available_temperatures(planet.metadata)
            current_t = planet.metadata.get("parameters_declared", {}).get("teff_K")
            default = [float(current_t)] if current_t is not None and float(current_t) in nodes else nodes[:1]
            selected_temperatures = st.multiselect("Planet temperature nodes [K]", nodes, default=default, disabled=use_evolution,key="sep_temperatures")
            if not nodes:
                st.info("This selection has no installed catalog temperature sequence. Use a local catalog with temperature nodes at matching gravity/composition.")
        radius_values = st.text_input("Planet radii [R_jup], comma separated", "%.8g" % (report["rplanet_m"]/RADII_M["R_jup"]),
            disabled=route == "Selected planet atmosphere" or use_evolution, key="sep_radii_"+item["id"])
        use_jy = st.checkbox("Add flux-density axis",value=True,key="sep_use_jy")
        flux_source=st.selectbox('Stellar flux normalization',['Selected stellar atmosphere','Supplied stellar flux'],disabled=not use_jy,key='sep_flux_source')
        jy = st.number_input("Stellar flux density [Jy; same observing band]", min_value=.000001, value=1.0, disabled=not use_jy or flux_source!='Supplied stellar flux', key="sep_jy")
    with right:
        band_source = st.selectbox("Separation bandpass", ["Calculated filter", "Upload throughput", "Local throughput file", "Custom top-hat"], key="sep_band")
        band_file, band_path, selected_band = None, None, None
        skip = 1
        if band_source == "Calculated filter":
            bands = [b for b in report["filter_results"] if "response" in b]
            if bands:
                index = st.selectbox("Use calculated filter", list(range(len(bands))), format_func=lambda i: bands[i]["name"], key="sep_calculated_band")
                selected_band = bands[index]
            else:
                st.info("Calculate a filter in Filters & limits first, or choose another bandpass source.")
        elif band_source == "Upload throughput":
            band_file = st.file_uploader("Throughput CSV: wavelength_um,throughput", type=["csv", "txt"], key="sep_throughput")
            skip = int(st.number_input("Throughput header lines", 0, 100, 1, key="sep_skip"))
        elif band_source == "Local throughput file":
            band_path = st.text_input("Separation throughput file path", key="sep_band_path")
            skip = int(st.number_input("Throughput header lines", 0, 100, 1, key="sep_skip"))
        else:
            band_lower = st.number_input("Top-hat start [µm]", min_value=.01, value=float(report["wavelength_range_um"][0]), key="sep_band_lower")
            band_upper = st.number_input("Top-hat end [µm]", min_value=.02, value=float(report["wavelength_range_um"][1]), key="sep_band_upper")
            st.caption("Custom top-hats are idealized responses. Enter the band you intend to examine.")
        detector = st.selectbox("Separation response convention", ["photon", "energy"], key="sep_detector", disabled=band_source == "Calculated filter")
        frame = st.selectbox("Separation band wavelength convention", ["vacuum", "air"], key="sep_frame", disabled=band_source == "Calculated filter")
        sensitivity_source=st.radio('Sensitivity overlay',['None','Upload observed sensitivity','Simulated sensitivity (layout test)'],key='sep_sensitivity_source')
        sensitivity_file = st.file_uploader("Observed sensitivity: IDL .sav or CSV", type=["sav", "csv"], key="sep_sensitivity") if sensitivity_source=='Upload observed sensitivity' else None
        observation_band = st.text_input("Observed sensitivity band", key="sep_observed_band")
        st.caption("Sensitivity CSV columns: separation_arcsec,contrast; optional pa_deg for individual position-angle curves.")
    if use_heating:
        st.warning("Approximate heating uses a non-irradiated temperature sequence at fixed gravity/composition. It does not solve an irradiated atmosphere. Unsupported heated temperatures leave gaps.")
        st.caption("Full redistribution: Teq = Tstar sqrt(Rstar/(2a)) (1−A)^¼; Teffective⁴ = Tintrinsic⁴ + Teq⁴. The geometry ratio sets true distance relative to projected separation.")
    if sensitivity_source=='Simulated sensitivity (layout test)':
        st.warning('LAYOUT TEST: gray and black sensitivity curves will be simulated. They are not observations or detection limits.')
    if st.button("Build separation plot", type="primary", key="sep_build"):
        st.session_state.pop("separation_result", None)
        st.session_state.pop("separation_export", None)
        try:
            if lower >= upper:
                raise ValueError("Maximum separation must exceed the minimum.")
            theta = np.linspace(lower, upper, 250)
            if band_source == "Calculated filter":
                if selected_band is None:
                    raise ValueError("Calculate a filter first or choose a supplied response.")
                w, tr = np.asarray(selected_band["response"]["wavelength_um"]), np.asarray(selected_band["response"]["throughput"])
                detector, frame = selected_band["detector"], selected_band["wavelength_frame"]
                band_meta = {"kind": "calculated_filter", "label": selected_band["name"], "source": selected_band["source"]}
            elif band_source in ("Upload throughput", "Local throughput file"):
                if band_source == "Upload throughput":
                    if band_file is None:
                        raise ValueError("Upload the throughput CSV first.")
                    content, name = band_file.getvalue(), band_file.name
                else:
                    if not band_path.strip():
                        raise ValueError("Supply a throughput file path.")
                    path = Path(band_path).expanduser()
                    content, name = path.read_bytes(), path.name
                data = np.loadtxt(io.BytesIO(content), delimiter=",", skiprows=skip, ndmin=2)
                w, tr = data[:, 0], data[:, 1]
                band_meta = {"kind": "supplied_throughput", "label": name, "sha256": hashlib.sha256(content).hexdigest(), "header_lines": skip}
            else:
                if band_lower >= band_upper:
                    raise ValueError("Bandpass end must exceed its start.")
                w, tr = np.linspace(band_lower, band_upper, 101), np.ones(101)
                band_meta = {"kind": "custom_top_hat", "label": "Custom %.3f–%.3f µm top-hat" % (band_lower, band_upper), "validation": "Idealized, user-selected response."}
            band_meta.update(detector=detector, wavelength_frame=frame, wavelength_um=w.tolist(), throughput=tr.tolist())
            if frame != star.metadata.get("wavelength_frame", "vacuum") or frame != planet.metadata.get("wavelength_frame", "vacuum"):
                raise ValueError("Bandpass and both spectra must share the air/vacuum convention.")
            distance, rs, rp = report["distance_pc"], report["rstar_m"], report["rplanet_m"]
            models, sequence, tstar = [], None, None
            if route == "Selected planet atmosphere" and not use_heating:
                b = band_contrast(star, planet, rp/rs, w, tr, detector)
                if b["planet_star_contrast"] <= 0:
                    raise ValueError("The selected planet has no positive signal in this band.")
                models.append({"label": item["label"], "contrast": [b["planet_star_contrast"]]*len(theta), "band_calculation": b, "rplanet_m": rp})
                scope = "selected atmosphere prediction"
            else:
                evolution_grid=None
                if use_evolution:
                    evolution_grid=catalog_band_grid(planet.metadata,w,tr,detector)
                    grid_t=evolution_grid.nodes['teff_K'];grid_f=None;sequence=evolution_grid.metadata
                else:
                    grid_t, grid_f, sequence = catalog_band_sequence(planet.metadata, w, tr, detector)
                temperatures = selected_temperatures
                radii = [float(v.strip())*RADII_M["R_jup"] for v in radius_values.split(",")]
                if route == "Selected planet atmosphere":
                    tint = (planet.metadata.get("parameters_verified_from_header") or planet.metadata.get("parameters_declared", {})).get("teff_K")
                    if tint is None:
                        raise ValueError("The selected planet needs a declared or header-verified temperature for heating.")
                    temperatures, radii = [float(tint)], [rp]
                if use_evolution:
                    temperatures=[evolved['intrinsic_temperature_K']]
                    radii=[evolved['radius_Rjup']*RADII_M['R_jup']]
                    for key in ('metallicity_dex','co_relative_solar'):
                        source=evolved['source'];held=sequence['held_parameters']
                        if key in source and (key not in held or not np.isclose(float(source[key]),float(held[key]),atol=1e-6)):
                            raise ValueError('Evolution and atmosphere '+key+' must match.')
                    # Verify the intrinsic point before plotting; no arbitrary gravity snapping.
                    evolution_grid.evaluate({'teff_K':temperatures[0],'logg_cgs':evolved['logg_cgs']})
                    sequence['evaluated_logg_cgs']=evolved['logg_cgs']
                if not temperatures or not radii or len(temperatures)*len(radii) > 30:
                    raise ValueError("Choose temperatures and radii, with at most 30 curves.")
                if not all(np.isfinite(r) and r > 0 for r in radii):
                    raise ValueError("Planet radii must be positive and finite.")
                stellar_signal = band_signal(star, w, tr, detector)
                if stellar_signal <= 0:
                    raise ValueError("No stellar signal in this band.")
                if use_heating:
                    if len(grid_t) < 2:
                        raise ValueError("Approximate heating needs at least two installed temperature nodes to bracket the heated signals.")
                    tstar = (star.metadata.get("parameters_verified_from_header") or star.metadata.get("parameters_declared", {})).get("teff_K")
                    if tstar is None:
                        raise ValueError("Stellar temperature is required for heating.")
                    tstar = float(tstar)
                scope = "approximate heating with atmosphere grids" if use_heating else "atmosphere temperature sequence"
                for tint in temperatures:
                    if tint < grid_t[0] or tint > grid_t[-1]:
                        raise ValueError("Requested intrinsic temperature lies outside the installed sequence.")
                    if use_heating:
                        _, true_au, teq, effective = heating(theta, distance, rs, tstar, tint, albedo, geometry)
                        if use_evolution:
                            signals,unsupported=evolution_grid.evaluate_many(effective,evolved['logg_cgs'])
                            sequence['unsupported_query_points']=unsupported
                        else:signals = interpolate_band_signal(grid_t, grid_f, effective)
                    else:
                        # UI selects exact native nodes; interpolation is not used in this option.
                        if use_evolution:
                            signal,corners=evolution_grid.evaluate({'teff_K':tint,'logg_cgs':evolved['logg_cgs']})
                            signals=np.full(len(theta),signal);sequence['intrinsic_interpolation_corners']=corners
                        else:
                            indices = np.flatnonzero(np.isclose(grid_t, tint, rtol=0, atol=1e-6))
                            if len(indices) != 1:raise ValueError("Choose an installed native temperature node.")
                            signals = np.full(len(theta), grid_f[indices[0]])
                        effective, teq, true_au = np.full(len(theta), tint), None, None
                    for rplanet in radii:
                        contrast = (rplanet/rs)**2*signals/stellar_signal
                        temperature_label = "T_int" if use_heating or use_evolution else "Teff"
                        curve = {"label": "%s: %s=%g K, %g Rjup" % (planet.metadata.get("family", "Atmosphere"), temperature_label, tint, rplanet/RADII_M["R_jup"]),
                            "intrinsic_temperature_K": tint if use_heating else None, "rplanet_m": rplanet,
                            "effective_temperature_K": effective.tolist(), "contrast": finite_list(contrast),
                            "omitted_out_of_grid_samples": int(np.sum(~np.isfinite(contrast)))}
                        if use_heating:
                            curve.update(equilibrium_temperature_K=teq.tolist(), assumed_true_distance_au=true_au.tolist())
                        models.append(curve)
                if not any(any(c is not None and c > 0 for c in model["contrast"]) for model in models):
                    raise ValueError("None of the requested curves is supported in this separation range.")
            observation = None
            if sensitivity_file is not None or sensitivity_source=='Simulated sensitivity (layout test)':
                obs = load_sensitivity(sensitivity_file.getvalue(), sensitivity_file.name) if sensitivity_file is not None else simulated_sensitivity(lower,upper)
                mean, counts = obs.summary()
                observation = {"separation_arcsec": obs.separation_arcsec.tolist(), "curves": finite_list(obs.curves),
                    "mean_contrast": finite_list(mean), "finite_curve_count": counts.tolist(), "metadata": obs.metadata,
                    "observing_band_declared": observation_band, "band_matching": "User-supplied label; observing/model response compatibility is not independently verified."}
            if use_jy and flux_source=='Selected stellar atmosphere':
                jy=equivalent_flux_density_Jy(star,rs,distance,w,tr,detector)
            settings = {"target_label": target, "theta_min_arcsec": lower, "theta_max_arcsec": upper,
                "distance_pc": distance, "rstar_m": rs, "stellar_teff_K": tstar,
                "bond_albedo": albedo if use_heating else None, "true_to_projected_distance_ratio": geometry if use_heating else None,
                "star_band_flux_Jy": jy if use_jy else None, "flux_normalization_source":flux_source,"route": route, "approximate_heating": use_heating,
                "scope": "Supplied or tabulated evolutionary radii; no self-consistent irradiation, mass posterior, reflected light or completeness calculation."}
            if observation and observation['metadata'].get('kind')=='simulated_layout_test':
                settings['target_label']=target+' - SIMULATED sensitivity layout test'
            result = {"source_comparison_id": item["id"], "settings": settings, "bandpass": band_meta,
                "software_version": __version__, "created_utc": datetime.now(timezone.utc).isoformat(),
                "model_scope": scope, "separation_arcsec": theta.tolist(), "projected_separation_au": (theta*distance).tolist(),
                "models": models, "temperature_sequence": sequence, "observation": observation,
                "evolution_planet":evolved if use_evolution else None,
                "source_spectra": {"star": star.metadata, "planet": planet.metadata}}
            positive = [c for m in models for c in m["contrast"] if c is not None and c > 0]
            if observation:
                positive += [c for c in observation["mean_contrast"] if c is not None and c > 0]
            settings.update(contrast_min=min(positive)/2, contrast_max=max(positive)*2)
            json.dumps(result, allow_nan=False)
            st.session_state.separation_result = result
        except (OSError, ValueError, KeyError, TypeError, IndexError, ImportError, UnicodeError) as exc:
            st.error("Separation plot could not run: " + str(exc))
    result = st.session_state.get("separation_result")
    if not result:
        st.info("Select a response and build the plot. Observed sensitivity is optional.")
        return
    if result["source_comparison_id"] != item["id"]:
        st.info("The spectrum comparison changed. Build the separation plot again.")
        return
    st.caption("Showing the last successful separation calculation. Apply changed controls with Build separation plot.")
    fig = go.Figure()
    obs = result.get("observation")
    if obs:
        for y in obs["curves"]:
            fig.add_trace(go.Scatter(x=obs["separation_arcsec"], y=y, mode="lines", showlegend=False,
                line={"color": "rgba(130,140,150,.35)", "width": 1}, connectgaps=False))
        fig.add_trace(go.Scatter(x=obs["separation_arcsec"], y=obs["mean_contrast"], mode="lines",
            name="SIMULATED sensitivity mean" if obs['metadata'].get('kind')=='simulated_layout_test' else "Imported sensitivity: mean", line={"color": "black", "width": 3}, connectgaps=False))
        if obs["metadata"].get("partial_pa_run"):
            st.warning("Imported file contains %d of %d planned position-angle curves." % (obs["metadata"]["saved_pa_count"],obs["metadata"]["planned_pa_count"]))
        if obs['metadata'].get('kind')=='simulated_layout_test':st.warning('The gray/black curves in this result are simulated for a plot-layout test; do not use them as measured limits.')
        else:st.caption("Verify that the imported observing band matches the modeling response. A curve crossing is a brightness comparison, not a detection probability.")
    else:
        st.caption("Atmosphere predictions only. Upload observed sensitivity when you want a comparison with measured limits.")
    for model in result["models"]:
        fig.add_trace(go.Scatter(x=result["separation_arcsec"], y=model["contrast"], mode="lines", name=model["label"],line={'color':'dodgerblue'},connectgaps=False))
        if model.get("omitted_out_of_grid_samples"):
            st.warning("%s: %d heated temperatures are outside the grid; those points are omitted." % (model["label"],model["omitted_out_of_grid_samples"]))
    settings = result["settings"]
    ticks = np.linspace(settings["theta_min_arcsec"],settings["theta_max_arcsec"],6)
    yrange = [np.log10(settings["contrast_min"]),np.log10(settings["contrast_max"])]
    layout = dict(template="plotly_white",height=620,margin={"t":100,"b":50},title=settings["target_label"]+" — "+result["model_scope"],
        xaxis={"title":"Angular separation [arcsec]","range":[settings["theta_min_arcsec"],settings["theta_max_arcsec"]]},
        xaxis2={"title":"Projected separation [au]","overlaying":"x","side":"top","matches":"x","tickvals":ticks.tolist(),
            "ticktext":["%.3g" % (x*settings["distance_pc"]) for x in ticks]},
        yaxis={"title":"Band-integrated planet / star contrast","type":"log","range":yrange},legend={"font":{"size":10}})
    if settings.get("star_band_flux_Jy"):
        yticks = np.geomspace(settings["contrast_min"],settings["contrast_max"],6)
        layout["yaxis2"]={"title":"Equivalent flux density [mJy]","type":"log","overlaying":"y","side":"right","range":yrange,
            "tickvals":yticks.tolist(),"ticktext":["%.3g" % (v*settings["star_band_flux_Jy"]*1000) for v in yticks]}
    fig.add_trace(go.Scatter(x=[settings["theta_min_arcsec"],settings["theta_max_arcsec"]],
        y=[settings["contrast_min"],settings["contrast_max"]],xaxis="x2",yaxis="y2" if settings.get("star_band_flux_Jy") else "y",
        opacity=0,mode="lines",showlegend=False,hoverinfo="skip"))
    fig.update_layout(**layout)
    st.plotly_chart(fig,use_container_width=True,key="sep_plot")
    st.caption(result["bandpass"]["label"]+" · "+result["bandpass"]["detector"]+" response")
    if st.button('Check this plot calculation',key='sep_check'):
        checks=[]
        normalizations={}
        for model in result['models']:
            values=np.asarray([np.nan if v is None else v for v in model['contrast']],float)
            check={'Curve':model['label'],'Finite positive points':int(np.sum(np.isfinite(values)&(values>0))),'Out-of-grid omissions':model.get('omitted_out_of_grid_samples',0)}
            if 'equilibrium_temperature_K' in model:
                tint=model['intrinsic_temperature_K']
                check['Heating identity passed']=bool(np.allclose(np.asarray(model['effective_temperature_K'])**4,tint**4+np.asarray(model['equilibrium_temperature_K'])**4,rtol=1e-10))
            if model.get('effective_temperature_K'):
                key=tuple(model['effective_temperature_K'])
                normalized=values/model['rplanet_m']**2
                if key in normalizations:check['Radius scaling passed']=bool(np.allclose(normalized,normalizations[key],rtol=1e-10,atol=0,equal_nan=True))
                else:normalizations[key]=normalized
            checks.append(check)
        result['numerical_checks']=checks
        st.session_state['separation_checks']={'source_id':result['created_utc'],'rows':checks}
    checked=st.session_state.get('separation_checks')
    if checked and checked['source_id']==result['created_utc']:
        import pandas as pd
        st.dataframe(pd.DataFrame(checked['rows']),width='stretch')
        st.caption('Numerical checks of this calculation; they do not validate an atmosphere grid or observing response.')
    if settings.get("star_band_flux_Jy"):
        st.caption("Equivalent mJy = contrast × supplied stellar Jy × 1000; the supplied calibration must match the response.")
    if st.button("Prepare separation export",key="sep_prepare"):
        st.session_state.separation_export={"signature":hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest(),
            "zip":export_separation(result),"png":plot_bytes(result),"pdf":plot_bytes(result,"pdf")}
    prepared=st.session_state.get("separation_export")
    if prepared and prepared["signature"]==hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest():
        cols=st.columns(3)
        cols[0].download_button("Separation PNG",prepared["png"],"contrast_vs_separation.png","image/png",key="sep_png")
        cols[1].download_button("Separation PDF",prepared["pdf"],"contrast_vs_separation.pdf","application/pdf",key="sep_pdf")
        cols[2].download_button("Separation results ZIP",prepared["zip"],"separation_results.zip","application/zip",key="sep_zip")
    with st.expander("Separation settings and provenance"):
        st.json({"settings":settings,"model_scope":result["model_scope"],"temperature_sequence":result["temperature_sequence"],
            "observation_metadata":obs["metadata"] if obs else None})

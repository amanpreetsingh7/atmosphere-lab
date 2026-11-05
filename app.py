"""Atmosphere Lab: a local Streamlit workbench inspired by the VORTEX workflow."""
from pathlib import Path
import copy
import hashlib
import io
import json
import tempfile
import uuid
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from atmosphere_compare.spectra import load_spectrum
from atmosphere_compare import __version__
from atmosphere_compare.compare import radius_m, band_contrast
from atmosphere_compare.workbench import snapshot, csv_bytes, figure_bytes, export_bundle
from separation_ui import render as render_separation
from model_library_ui import render as render_model_library
from general_inputs_ui import render as render_general_inputs
from physical_ui import render as render_physical
from example_ui import render as render_example
from parameter_ui import render as render_parameters
from atmosphere_compare.parameter_exploration import PARAMETER_NAMES

ROOT = Path(__file__).resolve().parent
REAL = "Atmosphere models"
st.set_page_config(page_title="Atmosphere Lab", page_icon="◉", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.lab-kicker {color:#5dd3c6;font-size:.78rem;letter-spacing:.2em;font-weight:600;}
.lab-description {color:#a8b9ce;max-width:850px;line-height:1.6;}
div[data-testid="stMetric"] {background:#152136;border:1px solid #263651;border-radius:12px;padding:16px;}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="lab-kicker">STELLAR + PLANETARY ATMOSPHERES</div>', unsafe_allow_html=True)
st.title("Atmosphere Lab")
st.markdown('<p class="lab-description">Compare emergent spectra, explore planet–star contrast, and test predictions through a filter. Save several comparisons to see how models and radii change the result.</p>', unsafe_allow_html=True)
if "saved" not in st.session_state:
    st.session_state.saved = []

render_model_library(ROOT)
render_general_inputs(ROOT)


def catalog_choice(role):
    path_text = st.text_input("Catalog JSON path", str(ROOT / "configs" / "app_catalog.json"), key=role+"_catalog_path")
    try:
        path = Path(path_text).expanduser().resolve()
        rows = json.loads(path.read_text())
        rows = [r for r in rows if r.get("role", role) == role]
        if not rows:
            st.info("This catalog has no %s entries." % role)
            return None
        family = st.selectbox("Model grid", sorted({r["family"] for r in rows}), key=role+"_family")
        available = [r for r in rows if r["family"] == family]
        params = sorted(set(k for r in available for k in r["parameters"]))
        selected = {}
        names = {"teff_K": "Temperature [K]", "logg_cgs": "log g [cgs]", "metallicity_dex": "Metallicity [dex]",
                 "co_relative_solar": "C/O relative to solar", "alpha_dex": "Alpha abundance [dex]"}
        names.update(PARAMETER_NAMES)
        for key in params:
            def parameter_order(encoded):
                value = json.loads(encoded)
                return (0, float(value)) if isinstance(value, (int, float)) else (1, str(value))
            values = sorted({json.dumps(r["parameters"].get(key)) for r in available}, key=parameter_order)
            choice = st.selectbox(names.get(key, key), values, format_func=lambda v: str(json.loads(v)), key=role+"_param_"+key)
            value = json.loads(choice)
            selected[key] = value
            available = [r for r in available if r["parameters"].get(key) == value]
        if len(available) != 1:
            st.error("The selected parameters do not identify one unique spectrum. Resolve duplicate catalog entries.")
            return None
        chosen = available[0]
        spec = dict(chosen["spectrum"], parameters=chosen["parameters"])
        model_path = (path.parent / spec["file"]).expanduser().resolve()
        if not model_path.is_file():
            st.caption("Selected model file is not installed yet. See MODEL_DOWNLOADS.md.")
        return {"kind": "catalog", "spec": spec, "base": str(path.parent), "catalog_path": str(path),
                "catalog_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "family": family}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        st.error("Cannot read catalog: " + str(exc))
        return None


def upload_choice(role):
    kind = st.selectbox("File format", ["Sonora Bobcat", "PHOENIX high resolution", "Linear ASCII export"],
                        index=1 if role == "star" else 0, key=role+"_format")
    uploaded = st.file_uploader("Spectrum file", type=["fits", "fit", "dat", "txt", "csv", "spec", "gz"], key=role+"_upload")
    label = st.text_input("Spectrum label", role.title()+" atmosphere", key=role+"_label")
    frame = st.selectbox("Wavelength convention", ["vacuum", "air"], key=role+"_frame")
    spec = {"label": label, "wavelength_frame": frame, "parameters": {}}
    parameter_text=st.text_area("Parameter declarations JSON (optional)","{}",key=role+"_declared_parameters")
    try:
        spec["parameters"]=json.loads(parameter_text)
        if not isinstance(spec["parameters"],dict):raise ValueError("Parameters must be a JSON object.")
    except ValueError as exc:
        st.error(str(exc));return None
    wave_upload = None
    if kind == "PHOENIX high resolution":
        spec["format"] = "phoenix_hires"
        wave_upload = st.file_uploader("PHOENIX wavelength FITS", type=["fits", "fit"], key=role+"_wave_upload")
    elif kind == "Sonora Bobcat":
        spec.update(format="sonora_bobcat")
        st.caption("Bobcat headers are detected automatically; native .gz files are accepted.")
        st.caption("Native Bobcat Fν: erg cm⁻² s⁻¹ Hz⁻¹; wavelength in microns.")
    else:
        spec.update(format="ascii", wavelength_unit=st.selectbox("Wavelength unit", ["um", "nm", "angstrom", "m", "cm"], key=role+"_wunit"),
                    flux_unit=st.selectbox("Linear flux unit", ["W_m2_um", "erg_cm2_s_Hz", "erg_cm2_s_cm", "erg_cm2_s_A", "erg_cm2_s_um", "W_m2_m", "W_m2_Hz"], key=role+"_funit"),
                    flux_convention=st.selectbox("Flux convention", ["surface_flux", "eddington_flux"], key=role+"_convention"),
                    skiprows=int(st.number_input("Header lines to skip", 0, 100, 0, key=role+"_skip")))
        delimiter = st.selectbox("Column separator", ["Whitespace", "Comma"], key=role+"_separator")
        spec["delimiter"] = "," if delimiter == "Comma" else None
        spec["wavelength_column"] = int(st.number_input("Wavelength column (zero-based)", 0, 100, 0, key=role+"_wcol"))
        spec["flux_column"] = int(st.number_input("Flux column (zero-based)", 0, 100, 1, key=role+"_fcol"))
        st.caption("Supply linear emergent flux, not continuum-normalized flux or logarithmic native values.")
    ready = uploaded is not None and (kind != "PHOENIX high resolution" or wave_upload is not None)
    return {"kind": "upload", "spec": spec, "upload": uploaded, "wave_upload": wave_upload} if ready else None


def load_choice(choice):
    if choice is None:
        raise ValueError("Choose both model files before running.")
    if choice["kind"] == "catalog":
        result = load_spectrum(choice["spec"], choice["base"])
        result.metadata.update(family=choice["family"], catalog_path=choice["catalog_path"], catalog_sha256=choice["catalog_sha256"])
        return result
    with tempfile.TemporaryDirectory(prefix="atmosphere_upload_") as directory:
        base = Path(directory)
        spec = copy.deepcopy(choice["spec"])
        file = choice["upload"]
        spec["file"] = "spectrum" + Path(file.name).suffix
        (base/spec["file"]).write_bytes(file.getvalue())
        if choice["wave_upload"] is not None:
            spec["wavelength_file"] = "wavelength.fits"
            (base/spec["wavelength_file"]).write_bytes(choice["wave_upload"].getvalue())
        result = load_spectrum(spec, base)
        for index, record in enumerate(result.metadata["files"]):
            record.pop("path", None)
            record["uploaded_name"] = file.name if index == 0 else choice["wave_upload"].name
        return result


with st.sidebar:
    st.subheader("Build a comparison")
    mode = REAL
    st.caption("Stellar and planetary atmosphere spectra")
    choices, radii = {}, {}
    for role in ("star", "planet"):
        with st.expander(role.title(), expanded=True):
            route = st.radio("Load from", ["Local catalog", "Upload files"], key=role+"_route")
            choices[role] = catalog_choice(role) if route == "Local catalog" else upload_choice(role)
            unit = st.selectbox("Radius unit", ["R_sun", "R_jup", "R_earth"], index=0 if role == "star" else 1, key=role+"_runit")
            value = st.number_input("Radius", min_value=.000001, max_value=10000.0, value=1.0, key=role+"_radius")
            radii[role] = {"value": value, "unit": unit}
    with st.expander("Wavelength and distance", expanded=False):
        custom = st.checkbox("Choose wavelength range", value=False, key="custom_range")
        lower = st.number_input("Lower wavelength [µm]", min_value=.01, value=.8, key="wave_lower")
        upper = st.number_input("Upper wavelength [µm]", min_value=.02, value=5.0, key="wave_upper")
        bin_r = st.selectbox("Output bin resolving power", [20, 50, 100, 200, 500], index=2, key="bin_r")
        distance = st.number_input("Common distance [pc]", min_value=.0001, value=10.0, key="distance")
    run_clicked = st.button("Run comparison", type="primary", key="run")
    st.caption("Apply changed controls with Run comparison.")

st.info("Select a stellar atmosphere and a planetary atmosphere, then Run comparison.")

if run_clicked:
    try:
        with st.spinner("Loading spectra and integrating common bins…"):
            star = load_choice(choices["star"])
            planet = load_choice(choices["planet"])
            settings = {"mode": mode, "radii": radii,
                        "bin_resolving_power": bin_r, "custom_wavelength_range_um": [lower, upper] if custom else None,
                        "distance_pc": distance,
                        "model_sources": {name: bodies.metadata for name, bodies in (("star", star), ("planet", planet))}}
            item = snapshot(star, planet, radius_m(radii["star"]), radius_m(radii["planet"]), distance,
                            bin_r, [lower, upper] if custom else None, mode, settings)
            item["id"] = uuid.uuid4().hex
            st.session_state.current = item
            st.session_state.native = (star, planet)
            # Avoid stale results after a failed run.
            st.session_state.run_error = None
    except (OSError, ValueError, KeyError, ImportError, TypeError) as exc:
        st.session_state.run_error = str(exc)
        st.error("Comparison could not run: " + str(exc))
        st.session_state.pop("current", None)
        st.session_state.pop("native", None)

item = st.session_state.get("current")
if item is None or item["mode"] != mode:
    st.info("Choose inputs and run a comparison to populate the plots.")
    st.caption("Install or import spectra using Model library above. Existing downloads are reused.")
    st.caption("Atmosphere Lab v"+__version__+" · Designed & developed by Amanpreet Singh")
    st.stop()

r = item["result"]
report = item["report"]
cols = st.columns(3)
cols[0].metric("Radius area ratio", "%.3g" % report["area_ratio"])
cols[1].metric("Shared wavelengths", "%.2f–%.2f µm" % tuple(report["wavelength_range_um"]))
cols[2].metric("Median binned contrast", "%.3g" % np.median(r["planet_star_contrast"]))
st.caption("The median spectral contrast is a display summary, not a filter-integrated contrast. Plots show the last successful run.")


def chart(traces, ylabel):
    fig = go.Figure()
    colors = ["#5dd3c6", "#ffb16b", "#9aa9ff", "#ee9dd9", "#b3d986", "#62b5ee"]
    for i, (x, y, label) in enumerate(traces):
        positive = np.asarray(y) > 0
        fig.add_trace(go.Scatter(x=np.asarray(x)[positive], y=np.asarray(y)[positive], name=label,
                                mode="lines", line={"width": 2, "color": colors[i % len(colors)]}))
    fig.update_layout(template="plotly_dark", height=440, margin={"l": 20, "r": 20, "t": 20, "b": 20},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend={"orientation": "h", "y": 1.16}, xaxis={"type": "log", "title": "Wavelength [µm]"},
                      yaxis={"type": "log", "title": ylabel})
    return fig


spectra_tab, contrast_tab, grid_tab, parameter_tab, filter_tab, separation_tab, physical_tab, example_tab, export_tab = st.tabs(["Spectra", "Contrast", "Model comparison", "Parameter exploration", "Filters & limits", "Contrast vs separation", "Physical properties", "Worked example", "Export & provenance"])
with spectra_tab:
    view = st.radio("Flux scale", ["Emergent surface flux", "Flux at common distance"], horizontal=True, key="flux_view")
    suffix = "surface_flambda_W_m2_um" if view.startswith("Emergent") else "observed_flambda_W_m2_um"
    st.plotly_chart(chart([(r["wavelength_um"], r[name+"_"+suffix], report[name]["label"]) for name in ("star", "planet")],
                          "Fλ [W m⁻² µm⁻¹]"), use_container_width=True, key="spectra_plot")
    left, right = st.columns(2)
    for container, name in ((left, "star"), (right, "planet")):
        with container:
            st.markdown("**%s spectrum**" % name.title())
            meta = report[name]
            verified=dict(meta.get('parameters_verified_from_filename',{}));verified.update(meta.get('parameters_verified_from_header',{}))
            st.write(verified or meta.get("parameters_declared") or "No atmospheric parameters declared.")
            if meta.get('parameters_verified_from_filename'):st.caption('Native filename parameters: '+str(meta['parameters_verified_from_filename']))
            if mode == REAL and not verified:
                st.caption("Parameter labels are declarations; they were not verified from this file's header.")
            extra_declarations={k:v for k,v in meta.get('parameters_declared',{}).items() if k not in verified}
            if extra_declarations and verified:
                st.write('Additional declared grid parameters:',extra_declarations)
                st.caption('These additional parameters were not verified from the native header.')
    st.caption("Output bins summarize native samples. Instrument convolution and unresolved spectral structure require separate validation.")
with contrast_tab:
    st.latex(r"C_\lambda=\left(\frac{R_p}{R_\star}\right)^2\frac{F_{\lambda,p}}{F_{\lambda,\star}}")
    st.plotly_chart(chart([(r["wavelength_um"], r["planet_star_contrast"], "Radius-scaled contrast")], "Planet / star contrast"),
                    use_container_width=True, key="contrast_plot")
    for note in report["sampling_notes"]:
        st.warning(note)
    st.caption("This spectral comparison describes emergent thermal emission at a shared distance, without reflected light. Approximate stellar heating is available separately in Contrast vs separation.")
with grid_tab:
    st.write("Save a run, change the model or radius, and run again to overlay the predictions.")
    label = st.text_input("Comparison label", item["label"], key="comparison_label_"+item["id"])
    save_col, clear_col = st.columns(2)
    if save_col.button("Save current comparison", key="save"):
        if len(st.session_state.saved) >= 12:
            st.warning("Twelve comparisons are already saved. Clear them before adding more.")
        else:
            saved = copy.deepcopy(item)
            saved["label"] = label.strip() or item["label"]
            saved["id"] = uuid.uuid4().hex
            st.session_state.saved.append(saved)
            st.success("Comparison saved in this session.")
    if clear_col.button("Clear saved comparisons", key="clear"):
        st.session_state.saved = []
    compatible = [s for s in st.session_state.saved if s["mode"] == mode]
    if compatible:
        st.plotly_chart(chart([(s["result"]["wavelength_um"], s["result"]["planet_star_contrast"], s["label"]) for s in compatible],
                              "Planet / star contrast"), use_container_width=True, key="comparison_plot")
        st.dataframe(pd.DataFrame([{"Label": s["label"], "Mode": s["mode"], "Rp/Rstar squared": s["report"]["area_ratio"],
                                    "Wavelength range [µm]": str(s["report"]["wavelength_range_um"]),
                                    "Star parameters":str(s["report"]["star"].get("parameters_declared",{})),
                                    "Planet parameters":str(s["report"]["planet"].get("parameters_declared",{}))} for s in compatible]), width="stretch")
    else:
        st.info("No saved comparisons in this mode yet.")
    st.caption("Saved runs can have different radii or stellar inputs; inspect their settings before attributing differences to the model grid.")
    st.caption("For a controlled comparison between planet model families, keep the star, radii, temperature, gravity and composition comparable. A star and its planet do not need the same gravity. Solar abundance references can differ between grids.")
with filter_tab:
    fsource = st.radio("Filter source", ["Upload throughput", "Local throughput file", "Idealized top-hat (method check)"], key="filter_source")
    fupload = st.file_uploader("Filter CSV: wavelength in µm, throughput 0–1", type=["csv", "txt", "dat"], key="filter_upload") if fsource == "Upload throughput" else None
    fpath = st.text_input("Throughput file path", key="filter_path") if fsource == "Local throughput file" else None
    if fsource=='Idealized top-hat (method check)':
        lo,hi=report['wavelength_range_um'];a,b=(3.5,4.) if lo<=3.5<4.<=hi else (lo+(hi-lo)/3,lo+2*(hi-lo)/3)
        top_lower=st.number_input('Idealized band start [µm]',min_value=.000001,value=float(a),key='filter_top_lower')
        top_upper=st.number_input('Idealized band end [µm]',min_value=.000002,value=float(b),key='filter_top_upper')
        st.caption('Constant unit throughput inside these bounds. This is an explicitly idealized response for method checks, not a measured instrument filter.')
    left, right = st.columns(2)
    with left:
        fname = st.text_input("Filter label", "Custom filter", key="filter_label",disabled=fsource=='Idealized top-hat (method check)')
        delimiter = st.selectbox("Filter separator", ["Comma", "Whitespace"], key="filter_separator",disabled=fsource=='Idealized top-hat (method check)')
        skip = int(st.number_input("Filter header lines", 0, 100, 0, key="filter_skip",disabled=fsource=='Idealized top-hat (method check)'))
    with right:
        detector = st.selectbox("Response convention", ["photon", "energy"], key="detector")
        frame = st.selectbox("Filter wavelength convention", ["vacuum", "air"], key="filter_frame")
        use_limit = st.checkbox("Compare with a supplied contrast limit", key="use_limit")
        limit = st.number_input("Contrast limit (planet/star)", min_value=1e-15, value=1e-5, format="%.3e", key="contrast_limit")
    if st.button("Calculate filter contrast", key="calculate_filter"):
        try:
            if fsource == "Local throughput file":
                path = Path(fpath).expanduser()
                if not fpath.strip() or not path.is_file():
                    raise ValueError("Choose a valid local throughput file.")
                content = path.read_bytes()
                source_name = path.name
            elif fsource=='Upload throughput':
                if fupload is None:
                    raise ValueError("Upload a throughput curve first.")
                content = fupload.getvalue()
                source_name = fupload.name
            if fsource=='Idealized top-hat (method check)':
                if top_lower>=top_upper:raise ValueError('Idealized band end must exceed its start.')
                wave=np.asarray([top_lower,(top_lower+top_upper)/2,top_upper]);throughput=np.ones(3)
                source={'kind':'idealized_top_hat','definition':'Unit throughput within the declared interval; no instrument calibration or observation.','limits_um':[top_lower,top_upper]}
                band_name='Idealized %.3f–%.3f µm top-hat'% (top_lower,top_upper)
            else:
                data = np.loadtxt(io.BytesIO(content), delimiter="," if delimiter == "Comma" else None, skiprows=skip, ndmin=2)
                wave, throughput = data[:, 0], data[:, 1]
                source = {"kind":"supplied_throughput","name": source_name, "sha256": hashlib.sha256(content).hexdigest(),
                          "bytes": len(content), "delimiter": delimiter, "skiprows": skip}
                band_name=fname
            star, planet = st.session_state.native
            if frame != star.metadata.get("wavelength_frame", "vacuum"):
                raise ValueError("Filter and spectrum air/vacuum conventions differ.")
            band = band_contrast(star, planet, np.sqrt(report["area_ratio"]), wave, throughput, detector)
            band.update(name=band_name, source=source, wavelength_frame=frame, mode=mode)
            band["response"] = {"wavelength_um": wave.tolist(), "throughput": throughput.tolist()}
            if use_limit:
                band.update(supplied_contrast_limit=limit, model_at_or_above_limit=band["planet_star_contrast"] >= limit)
            report["filter_results"].append(band)
        except (OSError, ValueError, IndexError, TypeError) as exc:
            st.error("Filter calculation could not run: " + str(exc))
    if report["filter_results"]:
        bands = report["filter_results"]
        st.dataframe(pd.DataFrame([{k: v for k, v in b.items() if k not in ("source","response")} for b in bands]), width="stretch")
        b = bands[-1]
        st.metric("Latest filter-integrated contrast", "%.6g" % b["planet_star_contrast"])
        if "supplied_contrast_limit" in b:
            st.write("Prediction is **%s** the supplied contrast threshold." % ("at or above" if b["model_at_or_above_limit"] else "below"))
    st.caption("A threshold comparison is not a detection probability or an exclusion of a companion. Supply the raw response and choose its convention to avoid applying photon weighting twice.")
with parameter_tab:
    render_parameters(item,st.session_state.native)
with separation_tab:
    render_separation(item, st.session_state.native)
with physical_tab:
    render_physical(item,st.session_state.native)
with example_tab:
    render_example(ROOT,item,st.session_state.native)
with export_tab:
    extra_results={}
    for session_key,export_key in (('separation_result','separation'),('physical_result','physical_properties'),('inference_result','inference'),('example_result','worked_example'),('parameter_result','parameter_exploration')):
        extra=st.session_state.get(session_key)
        if extra and extra.get('source_comparison_id')==item['id']:extra_results[export_key]=extra
    if st.session_state.get('evolution_planet'):extra_results['evolution_planet']=st.session_state.evolution_planet
    def export_signature():
        return hashlib.sha256(json.dumps([report['filter_results'],[s['id'] for s in st.session_state.saved],extra_results],sort_keys=True).encode()).hexdigest()
    st.markdown("**Download the last successful calculation**")
    st.download_button("Spectral contrast CSV", csv_bytes(r), "spectral_contrast.csv", "text/csv", key="download_csv")
    if st.button("Prepare figures and export bundle", key="prepare_export"):
        with st.spinner("Preparing figures and reproducibility records…"):
            st.session_state.export = {"id": item["id"], "png": figure_bytes(item), "pdf": figure_bytes(item, "pdf"),
                                       "zip": export_bundle(item, st.session_state.saved,extra_results),
                                       "signature": export_signature()}
    bundle = st.session_state.get("export")
    if bundle and bundle["id"] == item["id"]:
        signature = export_signature()
        if bundle["signature"] != signature:
            st.info("Results or saved comparisons changed. Prepare the bundle again to include the latest calculations.")
        else:
            cols = st.columns(3)
            cols[0].download_button("PNG figure", bundle["png"], "spectrum_comparison.png", "image/png", key="download_png")
            cols[1].download_button("PDF figure", bundle["pdf"], "spectrum_comparison.pdf", "application/pdf", key="download_pdf")
            cols[2].download_button("Complete results ZIP", bundle["zip"], "atmosphere_results.zip", "application/zip", key="download_zip")
    with st.expander("Model provenance and assumptions"):
        st.json(report)
    st.caption("Complete ZIP includes current spectral, filter, parameter-exploration, separation, physical-property, measurement-likelihood and evolutionary results when calculated. Native atmosphere files are not bundled; retain them.")
st.divider()
st.caption("Atmosphere Lab v"+__version__+" · Designed & developed by Amanpreet Singh")

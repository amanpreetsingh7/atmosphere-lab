"""Band-integrated model overlays and read-only import of existing sensitivity curves.

Heating is an explicit energy-budget approximation, not an irradiated-atmosphere solver.
No observational sensitivity is generated from a model spectrum.
"""
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import io
import json
import tempfile
import zipfile
import numpy as np
from .compare import band_signal, band_contrast, RADII_M
from .spectra import load_spectrum

AU_M = 149597870700.0


@dataclass
class Sensitivity:
    separation_arcsec: np.ndarray
    curves: np.ndarray  # position angle / curve first, separation second
    metadata: dict

    def __post_init__(self):
        x, y = np.asarray(self.separation_arcsec, float), np.asarray(self.curves, float)
        if x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)) or np.any(x <= 0):
            raise ValueError("Sensitivity needs at least two positive, finite separations in arcsec.")
        if y.ndim == 1:
            y = y[None, :]
        if y.ndim != 2 or y.shape[1] != len(x) or y.shape[0] == 0:
            raise ValueError("Sensitivity dimensions do not match its separations.")
        if np.any(np.isinf(y)) or np.any(y[np.isfinite(y)] <= 0) or not np.any(np.isfinite(y)):
            raise ValueError("Contrasts must be positive; missing values may be NaN.")
        order = np.argsort(x)
        x, y = x[order], y[:, order]
        if np.any(np.diff(x) <= 0):
            raise ValueError("Duplicate separations are ambiguous.")
        self.separation_arcsec, self.curves = x, y

    def summary(self):
        count = np.sum(np.isfinite(self.curves), axis=0)
        average = np.full(len(count), np.nan)
        np.divide(np.nansum(self.curves, axis=0), count, out=average, where=count > 0)
        return average, count


def sensitivity_from_idl(data, metadata=None):
    """IDL [separation, PA] is reversed by readsav to NumPy [PA, separation].

    Reject square arrays whose orientation cannot be independently identified.
    Some pipelines save the full planned PA list during a partial run.
    """
    d = {str(k).lower(): v for k, v in data.items()}
    if not {"curves", "separations"}.issubset(d):
        raise ValueError("IDL file must contain curves and separations.")
    x = np.asarray(d["separations"], float).reshape(-1)
    y = np.asarray(d["curves"], float)
    if y.ndim == 1:
        if len(y) != len(x):
            raise ValueError("IDL curve length does not match separations.")
        y = y[None, :]
    elif y.ndim == 2:
        if y.shape[0] == len(x) and y.shape[1] == len(x):
            raise ValueError("Square IDL curve array: orientation is ambiguous; export a long-format CSV.")
        if y.shape[1] != len(x):
            if y.shape[0] != len(x):
                raise ValueError("Neither IDL curve axis matches separations.")
            y = y.T
    else:
        raise ValueError("IDL curves must be a one- or two-dimensional numeric array.")
    meta = dict(metadata or {}, kind="idl_save", aggregation="arithmetic mean of finite PA contrasts")
    if "pas" in d:
        pas = np.asarray(d["pas"], float).reshape(-1)
        if not np.all(np.isfinite(pas)) or len(pas) < len(y):
            raise ValueError("IDL planned position angles are invalid or shorter than saved curves.")
        meta.update(planned_pa_count=len(pas), saved_pa_count=len(y),
                    saved_position_angles_deg=pas[:len(y)].tolist(), partial_pa_run=len(y) < len(pas))
    else:
        meta.update(saved_pa_count=len(y), partial_pa_run=None)
    if "threshold" in d:
        threshold = float(np.asarray(d["threshold"]).squeeze())
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("IDL SNR threshold is invalid.")
        meta["injection_snr_threshold"] = threshold
    meta["completion_note"] = "A matching PA count does not independently verify convergence of every injection."
    return Sensitivity(x, y, meta)


def load_sensitivity(content, filename):
    meta = {"uploaded_name": Path(filename).name, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    if Path(filename).suffix.lower() == ".sav":
        from scipy.io import readsav
        # Compressed readsav files also remain inside this disposable directory.
        with tempfile.TemporaryDirectory(prefix="atmosphere_sensitivity_") as folder:
            path = Path(folder)/"curve.sav"
            path.write_bytes(content)
            data = readsav(str(path), python_dict=True, uncompressed_file_name=str(Path(folder)/"uncompressed.sav"))
        return sensitivity_from_idl(data, meta)
    records = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    if not records or not {"separation_arcsec", "contrast"}.issubset(records[0]):
        raise ValueError("CSV needs separation_arcsec,contrast and optional pa_deg columns.")
    x = sorted({float(r["separation_arcsec"]) for r in records})
    labels = sorted({r.get("pa_deg", "curve") for r in records})
    y = np.full((len(labels), len(x)), np.nan)
    used = set()
    for row in records:
        key = (row.get("pa_deg", "curve"), float(row["separation_arcsec"]))
        if key in used:
            raise ValueError("CSV contains duplicate separation/PA entries.")
        used.add(key)
        y[labels.index(key[0]), x.index(key[1])] = float(row["contrast"]) if row["contrast"].strip() else np.nan
    meta.update(kind="csv", curve_labels=labels, aggregation="arithmetic mean of finite curves", partial_pa_run=None)
    return Sensitivity(np.asarray(x), y, meta)


def heating(theta, distance_pc, rstar_m, tstar_K, tint_K, albedo=.3, true_to_projected=1.0):
    theta = np.asarray(theta, float)
    if theta.ndim != 1 or len(theta) < 2 or np.any(~np.isfinite(theta)) or np.any(theta <= 0) or np.any(np.diff(theta) <= 0):
        raise ValueError("Model separations must be positive, finite and increasing.")
    values = (distance_pc, rstar_m, tstar_K, tint_K, true_to_projected)
    if not all(np.isfinite(v) and v > 0 for v in values) or not 0 <= albedo <= 1 or true_to_projected < 1:
        raise ValueError("Heating inputs must be positive; albedo in [0,1]; true/projected distance ratio >= 1.")
    projected_au = theta*distance_pc
    true_au = projected_au*true_to_projected
    if np.any(true_au*AU_M <= rstar_m):
        raise ValueError("An assumed orbital distance falls inside the star.")
    teq = tstar_K*np.sqrt(rstar_m/(2*true_au*AU_M))*(1-albedo)**.25
    effective = (tint_K**4 + teq**4)**.25
    return projected_au, true_au, teq, effective


def interpolate_band_signal(temperatures, signals, query):
    """Approximate log(signal) vs log(Teff); missing/outside grid values stay NaN."""
    t, f, q = map(lambda x: np.asarray(x, float), (temperatures, signals, query))
    if t.ndim != 1 or f.shape != t.shape or len(t) < 2 or np.any(~np.isfinite(t)) or np.any(~np.isfinite(f)) or np.any(t <= 0) or np.any(f <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError("Temperature sequence needs at least two unique increasing positive temperatures and band signals.")
    result = np.full(q.shape, np.nan)
    good = np.isfinite(q) & (q >= t[0]) & (q <= t[-1])
    result[good] = np.exp(np.interp(np.log(q[good]), np.log(t), np.log(f)))
    return result


def catalog_band_sequence(planet_metadata, wavelength, throughput, detector):
    path = Path(planet_metadata.get("catalog_path", ""))
    if not path.is_file():
        raise ValueError("Temperature-sequence heating requires a local catalog selection.")
    selected = planet_metadata.get("parameters_declared", {})
    held = {k: v for k, v in selected.items() if k != "teff_K"}
    rows = json.loads(path.read_text())
    rows = [row for row in rows if row.get("role", "planet") == "planet" and row["family"] == planet_metadata["family"]
            and {k: v for k, v in row["parameters"].items() if k != "teff_K"} == held
            and (path.parent/row["spectrum"]["file"]).is_file()]
    rows.sort(key=lambda row: row["parameters"]["teff_K"])
    temperatures, signals, provenance = [], [], []
    for row in rows:
        spectrum = load_spectrum(dict(row["spectrum"], parameters=row["parameters"]), path.parent)
        if spectrum.metadata.get("wavelength_frame", "vacuum") != planet_metadata.get("wavelength_frame", "vacuum"):
            raise ValueError("Temperature sequence mixes air and vacuum wavelengths.")
        temperatures.append(float(row["parameters"]["teff_K"]))
        signals.append(band_signal(spectrum, wavelength, throughput, detector))
        provenance.append(spectrum.metadata)
    t, f = np.asarray(temperatures), np.asarray(signals)
    if not len(t) or np.any(~np.isfinite(t)) or np.any(~np.isfinite(f)) or np.any(t <= 0) or np.any(f <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError("Installed temperature nodes must be unique and positive with positive band signals.")
    return np.asarray(temperatures), np.asarray(signals), {"family": planet_metadata["family"], "held_parameters": held,
        "temperature_nodes_K": temperatures, "band_signals": signals, "spectra": provenance,
        "catalog_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "interpolation": "Linear log(band signal) vs log(Teff), with no extrapolation. Held-out accuracy is band/grid dependent; inspect matching installed-model diagnostics. This result alone is not an accuracy certificate."}


def plot_bytes(result, format="png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 7))
    settings = result["settings"]
    observed = result.get("observation")
    if observed:
        x = observed["separation_arcsec"]
        for line in observed["curves"]:
            ax.semilogy(x, line, color=".72", alpha=.55, lw=.8)
        label='SIMULATED sensitivity mean - layout test only' if observed.get('metadata',{}).get('kind')=='simulated_layout_test' else 'Imported sensitivity: finite-value mean'
        ax.semilogy(x, observed["mean_contrast"], color="black", lw=2.5, label=label)
    for line in result["models"]:
        ax.semilogy(result["separation_arcsec"], line["contrast"], lw=1.5, color="dodgerblue", label=line["label"])
        if len(result["models"]) > 3:
            values = np.asarray([np.nan if v is None else v for v in line["contrast"]], float)
            supported = np.flatnonzero(np.isfinite(values) & (values > 0))
            if len(supported):
                index = supported[np.argmin(np.abs(supported-int(.64*len(values))))]
                ax.text(result["separation_arcsec"][index], values[index]*1.08,
                        line["label"].replace("Tintrinsic=", "T_int="), color="dodgerblue", fontsize=7, clip_on=True)
    ax.set_xlabel("Angular separation [arcsec]")
    ax.set_ylabel("Band-integrated planet / star contrast")
    ax.set_xlim(settings["theta_min_arcsec"], settings["theta_max_arcsec"])
    if "contrast_min" in settings:
        ax.set_ylim(settings["contrast_min"], settings["contrast_max"])
    distance = settings["distance_pc"]
    top = ax.secondary_xaxis("top", functions=(lambda x: x*distance, lambda x: x/distance))
    top.set_xlabel("Projected separation [au]")
    jy = settings.get("star_band_flux_Jy")
    if jy:
        right = ax.secondary_yaxis("right", functions=(lambda c: c*jy*1000, lambda f: f/(jy*1000)))
        right.set_ylabel("Band-equivalent flux density [mJy]")
    ax.grid(alpha=.2, which="both")
    if len(result["models"]) <= 3:
        ax.legend(fontsize=7, loc="best")
    elif observed:
        ax.legend(handles=[ax.lines[len(observed["curves"])]], fontsize=7, loc="upper right")
    ax.set_title(settings["target_label"] + " — " + result["model_scope"], fontsize=11, pad=55)
    fig.text(.1, .015, result["bandpass"]["label"] + " / " + result["bandpass"]["detector"] + " response", fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, .96))
    out = io.BytesIO()
    fig.savefig(out, format=format, dpi=180)
    plt.close(fig)
    return out.getvalue()


def export_separation(result):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("separation_report.json", json.dumps(result, indent=2, allow_nan=False))
        z.writestr("contrast_vs_separation.png", plot_bytes(result))
        z.writestr("contrast_vs_separation.pdf", plot_bytes(result, "pdf"))
        text = io.StringIO()
        writer = csv.writer(text)
        writer.writerow(["model", "separation_arcsec", "projected_separation_au", "contrast", "effective_temperature_K"])
        for model in result["models"]:
            temp = model.get("effective_temperature_K", [None]*len(result["separation_arcsec"]))
            writer.writerows(zip([model["label"]]*len(temp), result["separation_arcsec"], result["projected_separation_au"], model["contrast"], temp))
        z.writestr("model_curves.csv", text.getvalue())
        if result.get("observation"):
            obs = result["observation"]
            text = io.StringIO()
            writer = csv.writer(text)
            writer.writerow(["separation_arcsec", "mean_contrast", "finite_curve_count"])
            writer.writerows(zip(obs["separation_arcsec"], obs["mean_contrast"], obs["finite_curve_count"]))
            z.writestr("imported_sensitivity_mean.csv", text.getvalue())
        z.writestr("README.txt", "Only uploaded sensitivity is observational. Simulated layout curves are explicitly labeled and are not measured limits. Model curves are atmosphere predictions.\n"
                   "Heating with non-irradiated model grids is approximate. An adopted evolution mass/age may supply intrinsic temperature, radius and gravity; see evolution_planet. No mass/age posterior, reflected light or self-consistent irradiation solver.\n"
                   "Projected separation differs from true star-planet distance; see the adopted geometry in separation_report.json.\n"
                   "A curve crossing is not an injection-recovery completeness or statistical exclusion.\n")
    return out.getvalue()


def finite_list(array):
    a = np.asarray(array, float)
    return np.where(np.isfinite(a), a, None).tolist()


def simulated_sensitivity(lower,upper,seed=1256):
    """Plot-layout data only; never a measured sensitivity or detection experiment."""
    if not 0<lower<upper:raise ValueError('Invalid separation range for layout test.')
    x=np.geomspace(lower,upper,24)
    rng=np.random.default_rng(seed)
    baseline=1e-5+3e-4*(x/.2)**-2
    values=baseline[None,:]*np.exp(rng.normal(0,.18,(16,1))+rng.normal(0,.13,(16,len(x))))
    return Sensitivity(x,values,{'kind':'simulated_layout_test','is_observation':False,'seed':seed,
        'aggregation':'arithmetic mean','scientific_use':'Layout test only. No observations, completeness or physical constraints are generated.'})

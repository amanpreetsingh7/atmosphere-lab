from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import platform
import warnings
import numpy as np
from . import __version__
from .spectra import load_spectrum, select_from_catalog, file_record
from .compare import compare, radius_m, band_contrast


def run(config_path):
    config_path = Path(config_path).resolve()
    base = config_path.parent
    config = json.loads(config_path.read_text())
    bodies = {}
    for name in ("star", "planet"):
        body = config[name]
        if "catalog" in body:
            catpath = (base / body["catalog"]).resolve()
            spec = select_from_catalog(json.loads(catpath.read_text()), body["family"], body["parameters"])
            bodies[name] = load_spectrum(spec, catpath.parent)
            bodies[name].metadata["catalog"] = file_record(catpath)
        else:
            bodies[name] = load_spectrum(body["spectrum"], base)
    star, planet = bodies["star"], bodies["planet"]
    rs, rp = radius_m(config["star"]["radius"]), radius_m(config["planet"]["radius"])
    result = compare(star, planet, rs, rp, config.get("bin_resolving_power", 100),
                     config.get("wavelength_range_um"), config.get("distance_pc"))
    root = (base / config.get("output_directory", "../results")).resolve()
    out = root / datetime.now().strftime("comparison_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    with (out / "spectral_contrast.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(result.keys())
        writer.writerows(zip(*result.values()))
    notes = []
    for name in ("star", "planet"):
        bad = int(np.sum(result[name+"_native_samples_in_bin"] < 10))
        if bad:
            notes.append("%s: %d bins have fewer than 10 native samples; use coarser bins or inspect sampling." % (name, bad))
    bands = []
    for f in config.get("filters", []):
        path = (base / f["file"]).resolve()
        data = np.loadtxt(path, delimiter=f.get("delimiter"), skiprows=f.get("skiprows", 0), ndmin=2)
        # Filter files use microns and raw dimensionless throughput, not a named-filter approximation.
        w, t = data[:, 0], data[:, 1]
        band = dict(name=f["name"], wavelength_frame=f["wavelength_frame"], response_file=file_record(path))
        if f["wavelength_frame"] != star.metadata["wavelength_frame"]:
            raise ValueError("Filter and spectra must use the same air/vacuum wavelength convention.")
        band.update(band_contrast(star, planet, rp/rs, w, t, f["detector"]))
        if "contrast_limit" in f:
            limit = float(f["contrast_limit"])
            if not np.isfinite(limit) or limit <= 0:
                raise ValueError("Contrast limit must be positive and finite.")
            band.update(supplied_contrast_limit=limit, model_at_or_above_limit=band["planet_star_contrast"] >= limit)
        bands.append(band)
    report = {"software_version": __version__, "python": platform.python_version(), "numpy": np.__version__,
              "created_utc": datetime.now(timezone.utc).isoformat(), "configuration_file": file_record(config_path),
              "config": config, "star": star.metadata, "planet": planet.metadata,
              "rstar_m": rs, "rplanet_m": rp, "area_ratio": (rp/rs)**2,
              "spectrum_bins": len(result["wavelength_um"]), "filter_results": bands, "sampling_notes": notes,
              "scope": "Radius-scaled emergent thermal emission at a common distance. No reflected light, irradiation solver, mass/age inference, extinction, LSF convolution, or likelihood fit.",
              "numerics": "Independent bin integrals of piecewise-linear native F_lambda. Band contrast is ratio of integrals, not mean of wavelength contrasts."}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out / "filter_contrasts.csv").open("w", newline="") as stream:
        if bands:
            keys = [k for k in bands[0] if k != "response_file"]
            keys = list(dict.fromkeys(k for b in bands for k in b if k != "response_file"))
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows({k: b.get(k) for k in keys} for b in bands)
        else:
            stream.write("No filters supplied. See README.md for adding a measured throughput curve.\n")
    plot(result, config, out)
    for note in notes:
        warnings.warn(note)
    print("Results:", out)
    for b in bands:
        print(b["name"], "contrast = %.6g" % b["planet_star_contrast"])
    return out


def plot(result, config, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = result["wavelength_um"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for name in ("star", "planet"):
        y = result[name+"_surface_flambda_W_m2_um"]
        positive = y > 0
        label = config[name].get("label", name)
        axes[0].loglog(x[positive], y[positive], label=label)
    axes[0].set_ylabel(r"Surface $F_\lambda$ [W m$^{-2}$ $\mu$m$^{-1}$]")
    axes[0].legend()
    positive = result["planet_star_contrast"] > 0
    axes[1].loglog(x[positive], result["planet_star_contrast"][positive], color="#a34524")
    axes[1].set_ylabel("Planet / star contrast\nincluding radius scaling")
    axes[1].set_xlabel(r"Wavelength [$\mu$m]")
    for ax in axes:
        ax.grid(alpha=.18, which="both")
    title = config.get("title", "Atmosphere spectrum comparison")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / "spectrum_comparison.png", dpi=180)
    fig.savefig(out / "spectrum_comparison.pdf")
    plt.close(fig)

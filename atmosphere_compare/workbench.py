"""UI-independent snapshots, plots, and exports for the interactive workbench."""
from datetime import datetime, timezone
import csv
import io
import json
import zipfile
import numpy as np
from . import __version__
from .compare import compare
from .spectra import Spectrum


def blackbody_demo(temperature, role):
    from run_demo import surface_blackbody
    wave = np.geomspace(.3, 30, 30000)
    return Spectrum(wave, surface_blackbody(wave, temperature),
                    {"label": "%s %.0f K blackbody" % (role, temperature), "kind": "blackbody_demonstration",
                     "wavelength_frame": "vacuum", "parameters_declared": {"teff_K": temperature},
                     "parameters_verified_from_header": {}, "files": []})


def snapshot(star, planet, rs, rp, distance, bin_r, wave_range, mode, settings):
    result = compare(star, planet, rs, rp, bin_r, wave_range, distance)
    report = {"software_version": __version__, "created_utc": datetime.now(timezone.utc).isoformat(),
              "mode": mode, "star": star.metadata, "planet": planet.metadata,
              "rstar_m": rs, "rplanet_m": rp, "area_ratio": (rp/rs)**2,
              "distance_pc": distance, "bin_resolving_power": bin_r,
              "wavelength_range_um": [float(result["bin_lower_um"][0]), float(result["bin_upper_um"][-1])],
              "settings": settings, "filter_results": [],
              "scope": "Emergent thermal emission; no irradiation solver, reflected light, instrument LSF, or inference.",
              "sampling_notes": []}
    for name in ("star", "planet"):
        bad = int(np.sum(result[name+"_native_samples_in_bin"] < 10))
        if bad:
            report["sampling_notes"].append("%s: %d bins contain fewer than 10 native samples." % (name, bad))
    label = "%s / %s" % (planet.metadata["label"], star.metadata["label"])
    return {"result": result, "report": report, "label": label, "mode": mode}


def csv_bytes(result):
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(result.keys())
    writer.writerows(zip(*result.values()))
    return out.getvalue().encode()


def figure_bytes(item, format="png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = item["result"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for name in ("star", "planet"):
        y = r[name+"_surface_flambda_W_m2_um"]
        m = y > 0
        axes[0].loglog(r["wavelength_um"][m], y[m], label=item["report"][name]["label"])
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel(r"Surface $F_\lambda$ [W m$^{-2}$ $\mu$m$^{-1}$]")
    m = r["planet_star_contrast"] > 0
    axes[1].loglog(r["wavelength_um"][m], r["planet_star_contrast"][m], color="#c66d35")
    axes[1].set_ylabel("Planet / star contrast")
    axes[1].set_xlabel(r"Wavelength [$\mu$m]")
    for ax in axes:
        ax.grid(alpha=.2, which="both")
    fig.suptitle("BLACKBODY DEMONSTRATION" if item["mode"] == "Demonstration" else "ATMOSPHERE SPECTRA COMPARISON")
    fig.tight_layout()
    out = io.BytesIO()
    fig.savefig(out, format=format, dpi=180)
    plt.close(fig)
    return out.getvalue()


def comparison_figure_bytes(items, format="png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    for item in items:
        r = item["result"]
        m = r["planet_star_contrast"] > 0
        ax.loglog(r["wavelength_um"][m], r["planet_star_contrast"][m], label=item["label"])
    ax.set_xlabel(r"Wavelength [$\mu$m]")
    ax.set_ylabel("Planet / star contrast")
    ax.set_title("Saved comparisons — %s" % items[0]["mode"])
    ax.grid(alpha=.2, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = io.BytesIO()
    fig.savefig(out, format=format, dpi=180)
    plt.close(fig)
    return out.getvalue()


def physical_figure_bytes(result,format='png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=result['rows']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for radius in sorted({r['radius_Rjup'] for r in rows}):
        group=sorted([r for r in rows if r['radius_Rjup']==radius],key=lambda r:r['temperature_K'])
        axes[0].semilogy([r['temperature_K'] for r in group],[r['planet_star_contrast'] for r in group],'-o',markersize=3,label='%g Rjup'%radius)
    for temperature in sorted({r['temperature_K'] for r in rows}):
        group=sorted([r for r in rows if r['temperature_K']==temperature],key=lambda r:r['radius_Rjup'])
        axes[1].semilogy([r['radius_Rjup'] for r in group],[r['planet_star_contrast'] for r in group],'-o',markersize=3,label='%g K'%temperature)
    for ax in axes:
        if result.get('input_contrast'):ax.axhline(result['input_contrast'],color='black',ls='--',label=result['kind'])
        ax.grid(alpha=.2,which='both');ax.set_ylabel('Band planet/star contrast');ax.legend(fontsize=6,ncol=2)
    axes[0].set_xlabel('Planet temperature [K]');axes[1].set_xlabel('Planet radius [Rjup]')
    fig.suptitle('Atmosphere predictions through '+result['band']['name'])
    fig.tight_layout()
    output=io.BytesIO();fig.savefig(output,format=format,dpi=180);plt.close(fig)
    return output.getvalue()


def export_bundle(item, saved,extras=None):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("spectral_contrast.csv", csv_bytes(item["result"]))
        z.writestr("report.json", json.dumps(item["report"], indent=2))
        z.writestr("settings.json", json.dumps(item["report"]["settings"], indent=2))
        z.writestr("spectrum_comparison.png", figure_bytes(item))
        z.writestr("spectrum_comparison.pdf", figure_bytes(item, "pdf"))
        z.writestr("filter_results.json", json.dumps(item["report"]["filter_results"], indent=2))
        for i, other in enumerate(saved):
            z.writestr("saved_comparisons/%02d.csv" % i, csv_bytes(other["result"]))
            z.writestr("saved_comparisons/%02d.json" % i, json.dumps(other["report"], indent=2))
        compatible = [s for s in saved if s["mode"] == item["mode"]]
        if compatible:
            z.writestr("saved_comparisons.png", comparison_figure_bytes(compatible))
            z.writestr("saved_comparisons.pdf", comparison_figure_bytes(compatible, "pdf"))
        extras=extras or {}
        if extras.get('parameter_exploration'):
            from .parameter_exploration import write_exports
            write_exports(z,extras['parameter_exploration'])
        if extras.get('separation'):
            from .separation import plot_bytes
            other=extras['separation']
            z.writestr('separation/report.json',json.dumps(other,indent=2,allow_nan=False))
            z.writestr('separation/contrast_vs_separation.png',plot_bytes(other))
            z.writestr('separation/contrast_vs_separation.pdf',plot_bytes(other,'pdf'))
        if extras.get('physical_properties'):
            other=extras['physical_properties']
            z.writestr('physical_properties/report.json',json.dumps(other,indent=2,allow_nan=False))
            stream=io.StringIO();rows=other['rows']
            if rows:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            z.writestr('physical_properties/grid.csv',stream.getvalue())
            z.writestr('physical_properties/temperature_and_radius.png',physical_figure_bytes(other))
            z.writestr('physical_properties/temperature_and_radius.pdf',physical_figure_bytes(other,'pdf'))
        if extras.get('evolution_planet'):
            z.writestr('evolution/planet.json',json.dumps(extras['evolution_planet'],indent=2,allow_nan=False))
        if extras.get('inference'):
            other=extras['inference']
            z.writestr('inference/report.json',json.dumps(other,indent=2,allow_nan=False))
            stream=io.StringIO();rows=other['rows']
            if rows:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            z.writestr('inference/measurement_likelihoods.csv',stream.getvalue())
            z.writestr('inference/measurement_likelihoods.png',inference_figure_bytes(other))
            z.writestr('inference/measurement_likelihoods.pdf',inference_figure_bytes(other,'pdf'))
        if extras.get('worked_example'):
            other=extras['worked_example']
            z.writestr('worked_example/example.json',json.dumps(other,indent=2,allow_nan=False))
            stream=io.StringIO();rows=other['checks']
            if rows:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            z.writestr('worked_example/checks.csv',stream.getvalue())
        z.writestr("README.txt", "These are model predictions, not observations. The mode is recorded in every report.\n"
                   "Native atmosphere/filter files are not included; retain their originals and source versions.\n"
                   "Re-running uploaded-file calculations requires reselecting the original files with matching hashes.\n"
                   "Output-bin width is not an instrument line-spread function.\n")
    return out.getvalue()


def inference_figure_bytes(result,format='png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=result['rows'];mass='mass_Mjup' in rows[0]
    x,y=('mass_Mjup','age_Myr') if mass else ('temperature_K','radius_Rjup')
    fig,ax=plt.subplots(figsize=(8,6))
    points=ax.scatter([r[x] for r in rows],[r[y] for r in rows],c=[r['relative_likelihood'] for r in rows],vmin=0,vmax=1,cmap='viridis',s=40)
    fig.colorbar(points,ax=ax,label='Relative likelihood at tested points')
    if mass:ax.set_xscale('log');ax.set_yscale('log')
    ax.set_xlabel('Mass [M_jup]' if mass else 'Temperature [K]');ax.set_ylabel('Age [Myr]' if mass else 'Radius [R_jup]')
    ax.set_title('Conditional measurement likelihoods')
    fig.text(.1,.025,'Fixed atmosphere family, stellar inputs and response; no implicit grid prior.',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));out=io.BytesIO();fig.savefig(out,format=format,dpi=180);plt.close(fig)
    return out.getvalue()

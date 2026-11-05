"""Run in PyCharm: install an official PHOENIX/Bobcat model pair on your laptop.

Downloads approximately 1.22 GB on first run; keeps a reusable archive cache.
Selects 1000 K and 1500 K Bobcat spectra at g=100 m/s^2 (log g=4 cgs).
Writes catalog entries only after header and reader checks pass.
"""
from pathlib import Path
from datetime import datetime, timezone
import copy
import json
import platform
import shutil
import sys
import zipfile
import numpy as np
from atmosphere_compare.downloads import (PHOENIX_ROOT, PHOENIX_WAVE, PHOENIX_STAR, BOBCAT_URL,
                                          BOBCAT_BYTES, BOBCAT_MD5, download, extract_selected, digest_file)
from atmosphere_compare.spectra import load_spectrum
from atmosphere_compare.workbench import snapshot, export_bundle

ROOT = Path(__file__).resolve().parent
TEMPERATURES = (1000, 1500)
STAR_PARAMS = {"teff_K": 5000, "logg_cgs": 4.5, "metallicity_dex": 0.0, "alpha_dex": 0.0}


def backup_and_write(path, content):
    path = Path(path)
    if path.exists():
        backup = path.with_name(path.stem + ".before_model_setup_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + path.suffix)
        shutil.copy2(path, backup)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(content, indent=2), encoding="utf-8")
    part.replace(path)


def energy_diagnostic(spectrum, temperature):
    # A diagnostic only: native coverage need not be bolometrically complete.
    x, f = spectrum.wavelength_um, spectrum.surface_flux
    integrated = float(np.sum(np.diff(x)*(f[1:]+f[:-1])/2))
    expected = 5.670374419e-8*temperature**4
    return {"integrated_flux_over_native_range_W_m2": integrated, "sigma_teff4_W_m2": expected,
            "native_range_flux_over_sigma_teff4": integrated/expected,
            "interpretation": "Coverage-dependent unit/normalization diagnostic; spectra are not renormalized."}


def main():
    import astropy
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    cache = models / "archives"
    cache.mkdir(exist_ok=True)
    print("Installing real atmosphere spectra (no blackbody substitution).", flush=True)
    print("First run transfers about 1.22 GB and retains the archive for later grid additions.", flush=True)
    print("Partial downloads resume where supported. Re-run this file after an interruption.", flush=True)
    archive_path = cache / "spectra_m+0.0.tar.gz"
    missing_bytes = BOBCAT_BYTES
    if archive_path.exists():
        missing_bytes = 0
    elif archive_path.with_name(archive_path.name+".part").exists():
        missing_bytes -= archive_path.with_name(archive_path.name+".part").stat().st_size
    if shutil.disk_usage(models).free < max(missing_bytes, 0)+150_000_000:
        raise OSError("Not enough free space for the cached archive and selected spectra. Free about 1.4 GB before running.")
    star_directory = models / "phoenix"
    planet_directory = models / "bobcat"
    wave_path = download(PHOENIX_ROOT+PHOENIX_WAVE, star_directory/PHOENIX_WAVE, 12556800)
    star_path = download(PHOENIX_ROOT+"PHOENIX-ACES-AGSS-COND-2011/Z-0.0/"+PHOENIX_STAR,
                         star_directory/PHOENIX_STAR, 6287040)
    archive = download(BOBCAT_URL, archive_path, BOBCAT_BYTES, BOBCAT_MD5)
    destinations = {"spectra/sp_t%dg100nc_m0.0.gz" % t: planet_directory/("sp_t%dg100nc_m0.0.gz" % t) for t in TEMPERATURES}
    selected = extract_selected(archive, destinations)
    star_spec = {"format": "phoenix_hires", "file": str(star_path), "wavelength_file": str(wave_path),
                 "label": "PHOENIX: 5000 K, log g 4.5, solar metallicity", "wavelength_frame": "vacuum",
                 "parameters": STAR_PARAMS, "source_url": PHOENIX_ROOT+"PHOENIX-ACES-AGSS-COND-2011/Z-0.0/"+PHOENIX_STAR}
    star = load_spectrum(star_spec, ROOT)
    if not set(STAR_PARAMS).issubset(star.metadata["parameters_verified_from_header"]):
        raise ValueError("The stellar file does not supply all required parameter headers.")
    from astropy.io import fits
    h = fits.getheader(star_path, 0)
    wh = fits.getheader(wave_path, 0)
    if str(h.get("BUNIT", "")).replace(" ", "") != "erg/s/cm^2/cm" or wh.get("UNIT") != "Angstrom":
        raise ValueError("PHOENIX file-unit headers differ from the adapter's verified convention.")
    star.metadata["unit_headers_checked"] = {"BUNIT": h["BUNIT"], "wavelength_UNIT": wh["UNIT"]}
    star.metadata["family"] = "phoenix_hires"
    cases, rows, planetary = [], [], []
    star_entry = {"role": "star", "family": "phoenix_hires", "parameters": STAR_PARAMS,
                  "spectrum": dict(star_spec, file="../models/phoenix/"+PHOENIX_STAR,
                                   wavelength_file="../models/phoenix/"+PHOENIX_WAVE)}
    star_entry["spectrum"].pop("parameters")
    rows.append(star_entry)
    for temperature in TEMPERATURES:
        params = {"teff_K": temperature, "logg_cgs": 4.0, "metallicity_dex": 0.0, "co_relative_solar": 1.0}
        file = planet_directory / ("sp_t%dg100nc_m0.0.gz" % temperature)
        spec = {"format": "sonora_bobcat", "file": str(file), "label": "Bobcat: %d K, log g 4.0, solar metallicity" % temperature,
                "wavelength_frame": "vacuum", "parameters": params, "source_url": "https://zenodo.org/records/5063476"}
        planet = load_spectrum(spec, ROOT)
        if not set(params).issubset(planet.metadata["parameters_verified_from_header"]):
            raise ValueError("Bobcat parameter header was not fully parsed.")
        header_text = " ".join(planet.metadata["original_header"]).lower()
        if "microns" not in header_text or "erg/cm^2/s/hz" not in header_text:
            raise ValueError("Bobcat unit line does not match the verified native convention.")
        planet.metadata.update(family="sonora_bobcat", archive_md5_verified=BOBCAT_MD5,
                               archive_member="spectra/"+file.name, unit_header_checked=True)
        relative_spec = dict(spec, file="../models/bobcat/"+file.name)
        relative_spec.pop("parameters")
        rows.append({"role": "planet", "family": "sonora_bobcat", "parameters": params, "spectrum": relative_spec})
        diagnostic = energy_diagnostic(planet, temperature)
        print("Bobcat %d K: %d samples; native range %.3f–%.3f µm; integrated flux / sigma*T^4 = %.5f" %
              (temperature, len(planet.wavelength_um), planet.wavelength_um[0], planet.wavelength_um[-1],
               diagnostic["native_range_flux_over_sigma_teff4"]), flush=True)
        planetary.append({"metadata": planet.metadata, "energy_diagnostic": diagnostic})
        settings = {"mode": "Atmosphere models", "star": star_spec, "planet": spec,
                    "radii": {"star": {"value": 1, "unit": "R_sun"}, "planet": {"value": 1, "unit": "R_jup"}},
                    "distance_pc": 10, "bin_resolving_power": 100, "wavelength_range_um": [.8, 5.0]}
        item = snapshot(star, planet, 6.957e8, 7.1492e7, 10, 100, [.8, 5.0], "Atmosphere models", settings)
        item["label"] = "Bobcat %d K / PHOENIX 5000 K" % temperature
        cases.append(item)
    catalog_path = ROOT/"configs/app_catalog.json"
    from atmosphere_compare.model_library import upsert_catalog
    merged=upsert_catalog(ROOT,rows,replace=True)
    config_path = ROOT/"configs/real_models.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    config.update(title="PHOENIX star / Sonora Bobcat planet — real spectra", bin_resolving_power=100,
                  wavelength_range_um=[.8,5.0], distance_pc=10, output_directory="../results", filters=[])
    for role, row in (("star", rows[0]), ("planet", rows[1])):
        config[role] = {"label": row["spectrum"]["label"], "radius": {"value": 1, "unit": "R_sun" if role=="star" else "R_jup"},
                        "spectrum": dict(row["spectrum"], parameters=row["parameters"])}
    backup_and_write(config_path, config)
    report = {"status": "model_pair_installed_and_native_readers_checked_on_laptop", "created_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "numpy": np.__version__, "astropy": astropy.__version__,
              "official_archive": {"source_url": BOBCAT_URL, "bytes": BOBCAT_BYTES, "md5_verified": BOBCAT_MD5},
              "selected_archive_members": selected, "star": star.metadata,
              "star_energy_diagnostic": energy_diagnostic(star, 5000), "planets": planetary,
              "code_sha256": {p: digest_file(ROOT/p) for p in ("setup_real_models.py", "atmosphere_compare/downloads.py",
                                                               "atmosphere_compare/spectra.py", "atmosphere_compare/compare.py")},
              "scope": "Two illustrative non-irradiated planet spectra and one stellar spectrum; radii are supplied assumptions. No calibrated-filter or published-photometry validation yet."}
    (models/"installation_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    destination = Path.home()/"Desktop"
    if not destination.is_dir():
        destination = Path.home()
    result_zip = destination/("atmosphere_real_models_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")+".zip")
    with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("installation_manifest.json", json.dumps(report, indent=2))
        z.writestr("model_comparison_results.zip", export_bundle(cases[0], cases))
        z.writestr("app_catalog.json", json.dumps(merged, indent=2))
    print("\nREADY: restart launch_app.py, select Atmosphere models and Local catalog.", flush=True)
    print("Star grid: phoenix_hires; planet grid: sonora_bobcat. Planet temperatures available: 1000 and 1500 K.", flush=True)
    print("Upload results:", result_zip, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nSETUP STOPPED:", exc, flush=True)
        print("The app was not given an unvalidated replacement model. Re-run after fixing the reported issue.", flush=True)
        raise SystemExit(1)

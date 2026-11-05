# New in v0.8.0: cloud and composition subsets

Use **Model library > Download models**. Choose a temperature and gravity,
then add **Diamondback clouds + metallicity** (8 files) or **Flame Skimmer
metallicity + C/O + mixing** (11 files). Default 1000 K / g=100 m/s² subsets
require about 44 MB and 7 MB transferred respectively, including indexes.
Disk storage is about 100 MB and 3.24 MB. Other coordinates have other sizes.
Repeat at other listed native nodes to extend the library. The index is
checked for every exact required member; missing nodes stop the installer.

Sources are pinned to Diamondback record 12735103 and Flame Skimmer v1
record 20030439. Complete archive sizes and published MD5s are in the
installer. Range mode does **not** verify the MD5 of a complete archive;
receipts explicitly record this. ZIP CRCs check selected members and SHA256
records their installed bytes. A server that ignores byte ranges is refused;
there is no automatic multi-gigabyte fallback. Differing existing files are
not overwritten. The catalog is committed only after every import validates.

Flame Skimmer uses the R3000 quickstart format, with wavelength in microns
and emission flux in erg/s/cm²/cm. Install the updated requirements first.
Use a separate family/adapter for another resolution or release. Elf Owl v2
is a manual native NetCDF import, not a large automatic archive download.
Consult PARAMETER_GUIDE.txt for the in-app workflow and source links.

The following older instructions remain relevant to the existing installers.

# Initial model pair and primary sources

Model files are not bundled. Download them on your laptop, retaining the source archive/version and original filename. You can keep the original names and edit JSON paths; example renamed paths are only conveniences.

## Automatic starter-model setup in PyCharm

Run **setup_real_models.py** using the same interpreter as launch_app.py. It downloads the 5000 K, log g=4.5, solar-metallicity PHOENIX file and wavelength array, plus the official solar-metallicity Bobcat archive. About 1.22 GB is transferred on the first run; the archive stays under models/archives/ for reuse. Download progress appears in the Run panel, partial transfers resume where the server supports Range requests, and Bobcat's official MD5 is checked before extraction. About 1.4 GB of free disk space is needed.

The installer selects **1000 K and 1500 K**, g=100 m/s² (log g=4.0 cgs), solar-metallicity, solar-relative C/O Bobcat spectra. It checks model parameters and unit headers, performs the first comparison on your laptop, and updates configs/app_catalog.json only after reader checks pass. It backs up affected configurations and preserves unrelated custom catalog entries. A coverage-dependent integrated-flux diagnostic is recorded without renormalizing the spectra; that diagnostic is not a validation of calibrated filter photometry.

When it prints READY, run launch_app.py, select Atmosphere models → Local catalog, and choose phoenix_hires for the star and sonora_bobcat for the planet. Both planetary temperatures appear as exact available values. The script writes atmosphere_real_models_*.zip to the Desktop containing installation metadata and comparison outputs; upload that ZIP for review. Original spectra and the large archive are not included in the results ZIP.

The starter objects/radii are illustrative, not a measured star–planet system. The installer uses the original filenames and stores source URLs, hashes, header checks, and archive-member identities. No model grids are silently substituted. More grids and temperatures can be installed and validated later.

## Stellar spectrum: PHOENIX / Göttingen high-resolution library

Official page: https://phoenix.astro.physik.uni-goettingen.de/spectra/

The official site points to https://ftp.gwdg.de/pub/misc/phoenix/HiResFITS/.
Download the common wavelength array `WAVE_PHOENIX-ACES-AGSS-COND-2011.fits` and one stellar flux file from the metallicity subfolder. The site's naming rule gives a 5000 K, log(g)=4.5, solar-metallicity/alpha example as:

`PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte05000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits`

Place the wavelength file in `models/`. Point the star's `file` setting at your downloaded flux file (the example config uses `phoenix_star.fits`). High-resolution wavelengths are Angstroms and surface fluxes are F_lambda in erg/s/cm²/cm, not per Angstrom. The adapter converts that convention explicitly. Check header `PHXTEFF` and `PHXLOGG`; the loader checks them against the declared parameters when present. The grid ends at approximately 5.5 microns; longer-wave planet spectra do not expand the stellar coverage.

Paper: Husser et al. (2013), https://doi.org/10.1051/0004-6361/201219058 and https://arxiv.org/abs/1303.5632.

## Planet spectrum: Sonora Bobcat

Official data record: https://zenodo.org/records/5063476.

For solar metallicity and solar-relative C/O, download `spectra_m+0.0.tar.gz`, then extract the actual spectrum with Teff=1000 K and g=100 m/s² (log(g)=4.0 in cgs) for the example configuration. Select a spectrum, not a structure or evolution table. The archive is about 1.2 GB; the tool does not initiate this download automatically. Keep the original archive/name for provenance. Place your chosen spectrum in `models/` and update `planet.spectrum.file` (the example uses `bobcat_planet.dat`).

The record documents model parameters, wavelength in microns, and emergent radiation flux F_nu in erg/cm²/s/Hz. It corrected an earlier Eddington-flux description in February 2024: these spectra already contain radiation flux, so **do not multiply them by another 4*pi**. Inspection of the native archive shows individually gzipped files with two header lines: eight numeric parameters followed by labels, then a units line. The reader now accepts native .gz spectra, detects the start of two-column data automatically, and parses the first eight header tokens. g in m/s² is multiplied by 100 before taking log10 for cgs. C/O is relative to solar, not an absolute C/O number.

Paper: Marley et al. (2021), https://doi.org/10.3847/1538-4357/ac141d.

The record cautions that monochromatic spectral samples do not determine unresolved structure between points. Broad output-bin integration is numerical bookkeeping; native sampling and instrument convolution still require scientific validation. Sampling counts below 10 points/bin are reported for review rather than hidden.

## Other grid families

- Sonora Diamondback: https://zenodo.org/records/12735103. It documents different spectral units (W/m²/m), plus cloudy-model parameters; do not reuse the Bobcat F_nu adapter. Its description labels the flux F_nu despite per-wavelength dimensions. The release-specific native emission reader checks this per-wavelength header and parses cloud/composition filenames. It has synthetic fixture tests; genuine grid validation remains local.
- ATMO 2020: Phillips et al. (2020), https://arxiv.org/abs/2003.13717. Original emission ASCII and STScI converted FITS are separate adapters. In-app STScI equilibrium downloads cover its log g 5 release. Native reader science must be validated on the laptop.
- BT-Settl: use the exact release documentation and verify wavelength units, linear versus logarithmic flux, and radiation versus Eddington convention. The generic reader takes linear flux; it will not interpret a logarithmic native export automatically.

Native formats vary across releases. Adding a grid name to a configuration is not sufficient validation. Shared parameters across grids may also use different abundance references.

Sources checked October 12, 2026 UTC. No claims are made about models not actually loaded. These are model-selection instructions, not a claim that the default star/planet form a measured system.


## Version 0.7 additions

- Bobcat native evolution: https://zenodo.org/records/5063476 — `evolution_and_photometery.tar.gz`, 950605 bytes, MD5 `a7c844bdcc4093e5b667ecf944ca2128`. The included README defines Msun=1.989e30 kg, Rsun=6.9599e8 m, ages in Gyr, counted mass blocks and hot-start assumptions. In-app installation preserves those conventions before unit normalization.
- Cholla: https://zenodo.org/records/4450269 — `spectra.tar.gz`, 443284127 bytes, MD5 `84013e450b762153cc5724521a2eb2ac`. Native emission headers specify micron and W/m2/m, with T/g/logKzz encoded in model filenames. This is not the Bobcat Fnu format.
- ATMO STScI conversion: https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/atmo-2020-atmosphere-models. Its official directory is https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/atmo2020/. Real FITS headers use ANGSTROMS and the synphot FLAM alias (erg/s/cm2/Angstrom). The original ATMO ASCII instead uses micron and W/m2/micron. Only emission spectra are appropriate here.
- BT-Settl CIFIST through SVO: https://svo2.cab.inta-csic.es/theory/newov2/index.php. This adapter takes emission VOTables with column units, explicitly selected air wavelengths, and atmospheric parameter declarations. Raw logarithmic `.spec.7` files require a separate verified conversion. Air/vacuum mixing is rejected.

Official release metadata and small native headers were inspected here. Full real-grid installation and interpolation/energy tests execute on the user's laptop. Do not interpret an adapter's existence as validation of every model release.


## 0.7.2 overlap and reader checks
The Bobcat temperature/gravity installer now requests available g=31,56,100,178,316,562,1000 m/s2 members (approximately log g=3.5–5.0). The final catalog uses verified native header values. This provides a requested overlap with STScI ATMO log g=5.0, without implying every temperature/gravity combination is installed. New native files and band interpolation must be checked on the laptop. PHOENIX BUNIT/wavelength UNIT and explicitly present Bobcat/ATMO unit headers are checked at load time as well as during setup.

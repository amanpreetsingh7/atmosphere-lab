"""Install a requested native stellar spectrum from the STScI CK04 grid."""
from pathlib import Path
import math
import urllib.request
from atmosphere_compare.downloads import download, tls_context
from atmosphere_compare.model_library import register_model
from atmosphere_compare.spectra import load_spectrum

def install_ck04(root, teff_K, logg_cgs, metallicity_dex, expected_bytes=None):
    """Install an explicitly requested native CK04 cell; never snap to a node."""
    root = Path(root)
    temperature, gravity, metal = map(float, (teff_K, logg_cgs, metallicity_dex))
    if not all(math.isfinite(v) for v in (temperature, gravity, metal)) or temperature <= 0 or temperature != round(temperature):
        raise ValueError('Choose a positive native integer CK04 temperature.')
    if not 0 <= gravity <= 5 or not math.isclose(gravity*2, round(gravity*2), abs_tol=1e-10):
        raise ValueError('CK04 gravity columns use native log g = 0, 0.5, ..., 5.')
    if abs(metal*10-round(metal*10)) > 1e-9:raise ValueError('Choose a native CK04 metallicity value.')
    prefix = 'ck'+('m' if metal < 0 else 'p')+('%02d' % round(abs(metal)*10))
    filename = prefix+'_'+str(int(temperature))+'.fits'
    url = 'https://ssb.stsci.edu/trds/grid/ck04models/'+prefix+'/'+filename
    destination = root/'models/ck04'/filename
    if expected_bytes is None:
        if destination.is_file():expected_bytes = destination.stat().st_size
        else:
            request = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Atmosphere-Lab'})
            try:
                with urllib.request.urlopen(request, timeout=30, context=tls_context()) as reply:
                    expected_bytes = int(reply.headers['Content-Length'])
            except Exception as exc:
                raise ValueError('Native CK04 node unavailable or its size cannot be verified. Install actual bracketing native temperatures for interpolation; no nearest model was substituted. URL: '+url) from exc
    if not 0 < expected_bytes < 2_000_000:raise ValueError('Unexpected CK04 file size.')
    path = download(url, destination, expected_bytes)
    parameters = {'teff_K': temperature, 'logg_cgs': gravity, 'metallicity_dex': metal}
    mapping = {'format': 'ck04_stsci_fits', 'file': str(path), 'gravity_column': 'g%02d' % round(gravity*10),
        'wavelength_frame': 'vacuum', 'source_url': url,
        'label': 'CK04 ATLAS9: %g K, log g %g, metallicity %g' % (temperature, gravity, metal)}
    # Validate native identity/units and unavailable gravity columns before catalog mutation.
    load_spectrum(dict(mapping, parameters=parameters), root)
    row, metadata = register_model(root, 'star', 'ck04_atlas9_stsci', mapping, parameters, replace=True)
    print('READY:', mapping['label'], 'coverage', metadata['wavelength_range_um'], flush=True)
    return row

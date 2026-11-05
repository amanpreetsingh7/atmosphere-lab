"""Local prediction exports and model-library locking."""
from contextlib import contextmanager
from pathlib import Path
import csv
import io
import json
import zipfile

import numpy as np
from atmosphere_compare.compare import RADII_M


def _csv(columns):
    out = io.StringIO();writer = csv.writer(out);writer.writerow(list(columns))
    writer.writerows(zip(*columns.values()));return out.getvalue()


def export_result(result, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = Path(destination);folder.mkdir(parents=True, exist_ok=True)
    (folder/'prediction.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    weights = result['spectral_weights'];response = result['response_samples']
    columns = {'wavelength_um': weights['wavelength_um'], 'star_photon_density_relative': weights['star'],
        'planet_photon_density_relative': weights['planet'], 'filter_transmission': response['filter'],
        'reference_sky_transmission': response['sky'], 'filter_times_sky': response['product']}
    (folder/'relative_photon_weights.csv').write_text(_csv(columns))
    (folder/'spectral_contrast.csv').write_text(_csv({
        'wavelength_um': result['spectral_prediction']['wavelength_um'],
        'planet_star_contrast': result['spectral_prediction']['planet_star_contrast']}))
    lo, hi = result['band']['limits_um']
    (folder/'band_contrast.csv').write_text(_csv({'band': [result['band']['name']],
        'minimum_um': [lo], 'maximum_um': [hi], 'planet_star_contrast': [result['planet_star_contrast']],
        'delta_magnitude': [result['delta_magnitude']], 'detector': [result['detector']]}))
    w = np.asarray(weights['wavelength_um'])
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    for key, label in (('filter', 'Filter response'), ('sky', 'Sky transmission'), ('product', 'Filter × sky')):
        axes[0].plot(w, response[key], label=label)
    axes[0].set(ylabel='Transmission', ylim=(0, 1.05));axes[0].legend(fontsize=9)
    axes[1].plot(w, weights['star'], label='Star, linear photon weight')
    axes[1].plot(w, weights['planet'], label='Companion, linear photon weight')
    axes[1].set(xlabel='Wavelength [µm]', ylabel='Photon spectral density / maximum')
    axes[1].legend(fontsize=9)
    labels = [result['catalog_selections'][body]['spectrum'].get('label', result['catalog_selections'][body]['family']) for body in ('star', 'planet')]
    fig.suptitle(result['band']['name'])
    axes[0].set_title(labels[0] + ' / ' + labels[1], fontsize=9, wrap=True)
    fig.savefig(folder/'response_and_weights.png', dpi=180);plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    spectral = np.array([np.nan if v is None else v for v in result['spectral_prediction']['planet_star_contrast']])
    ax.semilogy(w, spectral, label='Sampled spectral contrast')
    ax.axhline(result['planet_star_contrast'], color='tab:orange', linestyle='--', label=result['detector'].title()+'-weighted band contrast')
    ax.set(xlabel='Wavelength [µm]', ylabel='Companion / star contrast', title='Theoretical spectral and band contrast')
    ax.text(.02, .03, 'Star %.3g R_sun · companion %.3g R_Jup' %
            (result['rstar_m']/RADII_M['R_sun'], result['rplanet_m']/RADII_M['R_jup']), transform=ax.transAxes, fontsize=8)
    ax.legend(fontsize=9);fig.savefig(folder/'model_contrast.png', dpi=180);plt.close(fig)
    (folder/'README.txt').write_text('Atmosphere brightness prediction through the response declared in prediction.json.\n'
        'Band contrast is a model prediction; no observation-derived sensitivity curve is generated.\n'
        'Weights are linear photon spectral-density weights, maximum normalized on a uniform wavelength grid.\n'
        'Filter/sky sources, cleanup (if any) and wavelength assumptions are in prediction.json.\n'
        'Telescope area, optics, QE, sky emission and observation-specific conditions are not modeled.\n')
    output = folder.with_suffix('.zip')
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.iterdir()):
            if path.is_file():archive.write(path, path.name)
    return output


@contextmanager
def model_access(root):
    import fcntl
    lock = Path(root)/'models/install_logs/installation.lock';lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as stream:
        try:fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Finish or stop the existing model job first.')
        yield

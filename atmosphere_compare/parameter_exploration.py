"""Controlled native-grid comparisons: change exactly one atmosphere coordinate."""
import csv
import hashlib
import io
import json
import zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from . import __version__
from .compare import compare,band_contrast
from .spectra import load_spectrum

PARAMETER_NAMES={'teff_K':'Temperature [K]','logg_cgs':'log g [cgs]',
 'metallicity_dex':'Metallicity [M/H; dex]','co_relative_solar':'C/O relative to solar',
 'fsed':'Cloud sedimentation f_sed / cloud-free','logKzz_cgs':'log Kzz [cm² s⁻¹]',
 'chemistry':'Chemistry','alpha_dex':'Alpha abundance [dex]'}
ORDER=('teff_K','logg_cgs','metallicity_dex','co_relative_solar','chemistry','fsed','logKzz_cgs')


def value_order(value):
    return (0,float(value)) if isinstance(value,(int,float)) else (1,str(value))


def value_label(value):
    return 'cloud-free' if value=='cloud_free' else ('%g'%value if isinstance(value,(int,float)) else str(value))


def candidates(rows,family,axis,held):
    return [r for r in rows if r.get('role','planet')=='planet' and r['family']==family
            and axis in r['parameters'] and {k:v for k,v in r['parameters'].items() if k!=axis}==held]


def explore(star,rows,base,axis,rstar_m,rplanet_m,resolving_power=100,wavelength_range=None,bands=()):
    if not 2<=len(rows)<=12:raise ValueError('Choose between two and twelve native models.')
    first=rows[0]
    if axis not in first['parameters']:raise ValueError('The chosen parameter is absent.')
    held={k:v for k,v in first['parameters'].items() if k!=axis}
    formats={r['spectrum']['format'] for r in rows}
    if len(formats)!=1:raise ValueError('Keep native formats and releases separate in controlled comparisons.')
    if any(r['family']!=first['family'] or r.get('role','planet')!='planet' or
           {k:v for k,v in r['parameters'].items() if k!=axis}!=held for r in rows):
        raise ValueError('A controlled comparison must hold every other atmosphere parameter and family fixed.')
    values=[r['parameters'][axis] for r in rows]
    if len({json.dumps(v,sort_keys=True) for v in values})!=len(values):raise ValueError('Duplicate native values for the varied parameter.')
    ordered=sorted(rows,key=lambda r:value_order(r['parameters'][axis]))
    spectra=[load_spectrum(dict(r['spectrum'],parameters=r['parameters']),base) for r in ordered]
    lo=max([star.wavelength_um[0]]+[s.wavelength_um[0] for s in spectra])
    hi=min([star.wavelength_um[-1]]+[s.wavelength_um[-1] for s in spectra])
    if wavelength_range is not None:
        a,b=map(float,wavelength_range)
        if not np.isfinite([a,b]).all() or a>=b:raise ValueError('Requested wavelength interval is invalid.')
        lo,hi=max(lo,a),min(hi,b)
    if lo>=hi:raise ValueError('Selected spectra have no shared wavelength interval.')
    curves=[];band_rows=[]
    for row,spectrum in zip(ordered,spectra):
        value=row['parameters'][axis]
        result=compare(star,spectrum,rstar_m,rplanet_m,resolving_power,[lo,hi])
        curves.append({'value':value,'label':value_label(value),'parameters':row['parameters'],
                       'spectrum':spectrum.metadata,'result':{k:v.tolist() for k,v in result.items()}})
        for band in bands:
            if 'response' not in band:continue
            if band.get('wavelength_frame','vacuum')!=star.metadata.get('wavelength_frame','vacuum'):
                raise ValueError('A response uses a different wavelength convention.')
            response=band['response']
            signal=band_contrast(star,spectrum,rplanet_m/rstar_m,response['wavelength_um'],response['throughput'],band['detector'])
            band_rows.append(dict(parameter_value=value,band=band['name'],**signal))
    return {'software_version':__version__,'created_utc':datetime.now(timezone.utc).isoformat(),
            'family':first['family'],'varied_parameter':axis,'parameter_label':PARAMETER_NAMES.get(axis,axis),
            'held_parameters':held,'rstar_m':float(rstar_m),'rplanet_m':float(rplanet_m),
            'bin_resolving_power':float(resolving_power),'requested_wavelength_range_um':wavelength_range,
            'shared_wavelength_range_um':[float(lo),float(hi)],'star':star.metadata,'curves':curves,
            'responses':list(bands),'band_rows':band_rows,
            'method':'Exact installed native models only; same star, radii and wavelength bins for every curve. No interpolation of cloud, metallicity, C/O or mixing coordinates. Bands are integrated on each native spectrum, not the display bins.',
            'scope':'Emergent thermal spectra at fixed imposed radii; no reflected light or self-consistent irradiation. Cloud-free is a discrete model case. Differences across families include other model physics and abundance conventions.'}


def figure_bytes(result,fmt='png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True,constrained_layout=True)
    for curve in result['curves']:
        data=curve['result'];label=curve['label']
        axes[0].loglog(data['wavelength_um'],data['planet_surface_flambda_W_m2_um'],label=label)
        axes[1].loglog(data['wavelength_um'],data['planet_star_contrast'],label=label)
    axes[0].set_ylabel('Companion surface Fλ [W m⁻² µm⁻¹]')
    axes[1].set_ylabel('Companion / star contrast');axes[1].set_xlabel('Wavelength [µm]')
    axes[0].legend(title=result['parameter_label'],fontsize=8,ncol=2)
    import textwrap
    from .compare import RADII_M
    short={'teff_K':'T [K]','logg_cgs':'log g [cgs]','metallicity_dex':'[M/H; dex]',
           'co_relative_solar':'C/O [solar]','chemistry':'chemistry','fsed':'f_sed','logKzz_cgs':'log Kzz [cm²/s]'}
    fixed=', '.join('%s=%s'%(short.get(k,k),value_label(v)) for k,v in result['held_parameters'].items())
    radii='Rstar=%.4g Rsun; Rp=%.4g Rjup; fixed stellar spectrum; native thermal models'%(result['rstar_m']/RADII_M['R_sun'],result['rplanet_m']/RADII_M['R_jup'])
    fig.suptitle(result['family']+' — '+result['parameter_label'],fontsize=11)
    axes[0].set_title(textwrap.fill(fixed+'\n'+radii,width=110),fontsize=8)
    for ax in axes:ax.grid(True,which='both',alpha=.2)
    out=io.BytesIO();fig.savefig(out,format=fmt,dpi=180);plt.close(fig);return out.getvalue()


def write_exports(archive,result,prefix='parameter_exploration/'):
    from .workbench import csv_bytes
    archive.writestr(prefix+'report.json',json.dumps(result,indent=2,allow_nan=False))
    archive.writestr(prefix+'parameter_comparison.png',figure_bytes(result))
    archive.writestr(prefix+'parameter_comparison.pdf',figure_bytes(result,'pdf'))
    stream=io.StringIO()
    if result['band_rows']:
        writer=csv.DictWriter(stream,fieldnames=list(result['band_rows'][0]));writer.writeheader();writer.writerows(result['band_rows'])
    archive.writestr(prefix+'band_contrasts.csv',stream.getvalue())
    for i,curve in enumerate(result['curves']):archive.writestr(prefix+'spectra/%02d.csv'%i,csv_bytes(curve['result']))


def export_bundle(result):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:write_exports(archive,result,prefix='')
    return buffer.getvalue()

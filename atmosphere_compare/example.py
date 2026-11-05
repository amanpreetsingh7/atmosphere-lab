"""Worked-example checks on the user's selected spectra, without substituted data."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from . import __version__
from .compare import band_signal,band_contrast,compare,equivalent_flux_density_Jy,H,C


def independent_band_signal(spec,wavelength,throughput,detector):
    """Simpson integration on every native/response segment; independent of Gauss rule."""
    w,t=np.asarray(wavelength,float),np.asarray(throughput,float)
    x,f=spec.wavelength_um,spec.surface_flux
    if detector not in ('energy','photon') or w.ndim!=1 or t.shape!=w.shape or len(w)<3 or np.any(np.diff(w)<=0) or np.any(~np.isfinite(w)) or np.any(~np.isfinite(t)) or np.any(t<0) or np.any(t>1) or not np.any(t>0):raise ValueError('Invalid observing response.')
    if w[0]<x[0] or w[-1]>x[-1]:raise ValueError('Independent integration requires complete response coverage.')
    knots=np.unique(np.concatenate((w,x[(x>w[0])&(x<w[-1])])))
    def integrand(q):
        y=np.interp(q,x,f)*np.interp(q,w,t)
        return y*q*1e-6/(H*C) if detector=='photon' else y
    left,right=knots[:-1],knots[1:]
    return float(np.sum((right-left)*(integrand(left)+4*integrand((left+right)/2)+integrand(right))/6))


def relative_error(actual,expected):
    actual,expected=np.asarray(actual,float),np.asarray(expected,float)
    positive=expected>0
    if actual.shape!=expected.shape or not np.any(positive) or np.any(~np.isfinite(actual)) or np.any(~np.isfinite(expected)):raise ValueError('A consistency check lacks comparable finite positive values.')
    error=float(np.max(np.abs(actual[positive]/expected[positive]-1)))
    if np.any(actual[~positive]!=expected[~positive]):return None
    return error


def run_example(item,native,band):
    star,planet=native;report=item['report'];response=band['response'];w,t=response['wavelength_um'],response['throughput'];detector=band['detector']
    if any(s.metadata.get('wavelength_frame','vacuum')!=band['wavelength_frame'] for s in native):raise ValueError('The response and spectra must use the same wavelength convention.')
    rs,rp,d=report['rstar_m'],report['rplanet_m'],report['distance_pc']
    baseline=band_contrast(star,planet,rp/rs,w,t,detector)
    if baseline['planet_star_contrast']<=0:raise ValueError('The selected planet needs positive signal in this response.')
    doubled=band_contrast(star,planet,2*rp/rs,w,t,detector)
    distant=compare(star,planet,rs,rp,report['bin_resolving_power'],report['wavelength_range_um'],2*d)
    checks=[]
    def add(name,actual,expected):
        error=relative_error(actual,expected)
        checks.append({'check':name,'max_relative_difference':error,'tolerance':1e-10,'passed':error is not None and error<=1e-10})
    add('Double planet radius: band contrast increases by four',doubled['planet_star_contrast'],4*baseline['planet_star_contrast'])
    add('Double shared distance: spectral contrast is unchanged',distant['planet_star_contrast'],item['result']['planet_star_contrast'])
    for body in ('star','planet'):
        add('Double shared distance: '+body+' flux is one quarter',distant[body+'_observed_flambda_W_m2_um'],item['result'][body+'_observed_flambda_W_m2_um']/4)
        spec=star if body=='star' else planet
        independent=independent_band_signal(spec,w,t,detector)
        add(body.title()+' band: independent Simpson and Gauss integration agree',independent,baseline[body+'_surface_band_signal'])
    flux=equivalent_flux_density_Jy(star,rs,d,w,t,detector)
    return {'software_version':__version__,'source_comparison_id':item['id'],'selected_band':band,'inputs':{'rstar_m':rs,'rplanet_m':rp,'distance_pc':d},
        'source_spectra':{'star':star.metadata,'planet':planet.metadata},'checks':checks,
        'band_prediction':dict(baseline,star_band_equivalent_Jy=flux,planet_band_equivalent_mJy=baseline['planet_star_contrast']*flux*1000),
        'is_observation':False,'source_input_type':'User-selected spectrum files. Their physical model origin must be verified from provenance.',
        'interpretation':'Worked atmosphere prediction with numerical consistency checks. No brightness measurement, noise or observed sensitivity is invented. Agreement checks establish the implemented scaling/integration of these files, not the physical accuracy of an atmosphere grid.'}


def load_validation(root):
    directory=Path(root)/'results';marker=directory/'latest_validation.json'
    if not marker.is_file():return None
    record=json.loads(marker.read_text());path=Path(record['path']).resolve()
    try:path.relative_to(directory.resolve())
    except ValueError:raise ValueError('Validation report path is outside this project results folder.') from None
    with zipfile.ZipFile(path) as archive:
        if archive.getinfo('validation.json').file_size>25_000_000:raise ValueError('Validation report is unexpectedly large.')
        raw=archive.read('validation.json')
    return {'report':json.loads(raw),'zip_path':str(path),'report_sha256':hashlib.sha256(raw).hexdigest()}


def validation_coverage(validation,native,band):
    reports=validation['report'].get('reports',[]);records=[r for report in reports for r in report.get('records',[])]
    covered={}
    for body,spectrum in zip(('star','planet'),native):
        hashes={r['sha256'] for r in spectrum.metadata.get('files',[])}
        params=spectrum.metadata.get('parameters_declared',{})
        covered[body]=bool(hashes) and any(hashes=={f['sha256'] for f in r['metadata'].get('files',[])} and r.get('parameters')==params for r in records)
    def signature(b):return {'response':b['response'],'detector':b['detector'],'wavelength_frame':b.get('wavelength_frame','vacuum')}
    covered['response']=any(signature(b)==signature(band) for b in validation['report'].get('responses',[]))
    recorded_version=validation['report'].get('software_version')
    covered['diagnostic_software_version']=recorded_version
    covered['software_version_matches_current']=None if recorded_version is None else recorded_version==__version__
    covered['meaning']='Matching source hashes, parameter declarations and response arrays. This is coverage of the diagnostics, not a scientific pass certificate.'
    return covered

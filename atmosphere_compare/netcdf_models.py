"""Release-specific Sonora NetCDF emission adapters, with explicit unit checks."""
import json
import math
import re
import warnings
import numpy as np

NUMBER=r'([+-]?\d+(?:\.\d+)?)'


def filename_parameters(name, fmt):
    if fmt == 'sonora_flame_skimmer_v1_r3k':
        match=re.fullmatch(r'spectra_(?:logkzz_'+NUMBER+r'_)?teff_'+NUMBER+r'_grav_'+NUMBER+r'_mh_'+NUMBER+r'_co_'+NUMBER+r'_R3000\.nc',name)
    else:
        match=re.fullmatch(r'spectra_logzz_'+NUMBER+r'_teff_'+NUMBER+r'_grav_'+NUMBER+r'_mh_'+NUMBER+r'_co_'+NUMBER+r'\.nc',name)
    if not match:raise ValueError('Retain the published filename for the selected Sonora release and resolution.')
    k,t,g,m,co=match.groups()
    t,g,m,co=map(float,(t,g,m,co))
    if t<=0 or g<=0 or co<=0:raise ValueError('Invalid native atmosphere coordinates.')
    params={'teff_K':t,'logg_cgs':math.log10(g*100),'metallicity_dex':m,'co_relative_solar':co,
            'chemistry':'equilibrium' if k is None else 'disequilibrium'}
    if k is not None:params['logKzz_cgs']=float(k)
    return params


def load_netcdf(path, fmt):
    try:from netCDF4 import Dataset
    except ImportError as exc:raise ImportError('Sonora NetCDF models need netCDF4. Install this release\'s requirements.txt in the interpreter running the app.') from exc
    from astropy import units as u
    filename=filename_parameters(path.name,fmt)
    flame=fmt=='sonora_flame_skimmer_v1_r3k'
    flux_name='flux_emission' if flame else 'flux'
    with Dataset(path,'r') as ds:
        if not {'wavelength',flux_name}.issubset(ds.variables):raise ValueError('Missing native wavelength or emission flux variable.')
        wvar,fvar=ds.variables['wavelength'],ds.variables[flux_name]
        if wvar.dimensions!=fvar.dimensions or len(wvar.dimensions)!=1:raise ValueError('Native spectrum must use matching 1D wavelength coordinates.')
        wu=getattr(wvar,'units','');fu=getattr(fvar,'units','')
        try:
            wfactor=u.Unit(wu).to(u.um)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='.*multiple slashes.*',category=u.UnitsWarning)
                ffactor=u.Unit(fu).to(u.erg/u.s/u.cm**2/u.cm)
        except (ValueError,TypeError) as exc:raise ValueError('NetCDF wavelength and flux units are absent or incompatible with this release.') from exc
        if not math.isclose(wfactor,1,rel_tol=1e-12) or not math.isclose(ffactor,1,rel_tol=1e-12):
            raise ValueError('Native NetCDF adapter requires microns and erg/s/cm²/cm. Use an explicit ASCII adapter for converted units.')
        w,f=wvar[:],fvar[:]
        if np.ma.is_masked(w) or np.ma.is_masked(f):raise ValueError('Masked native spectral samples cannot be silently discarded.')
        raw=getattr(ds,'planet_params',None)
        if not isinstance(raw,str):raise ValueError('Native planet_params JSON is required.')
        params=json.loads(raw)
        if not isinstance(params,dict):raise ValueError('planet_params must be a JSON object.')
        headers={'wavelength_units':wu,'flux_units':fu,'planet_params':params}
        if flame:
            resolution=getattr(ds,'rebinned_resolution','')
            if resolution!='R=3000':raise ValueError('Select the R=3000 quickstart release; do not mix native resolutions in one family.')
            headers['rebinned_resolution']=resolution
        measured={}
        def quantity(key,target_unit):
            value=params[key]
            if not isinstance(value,dict) or not {'value','unit'}.issubset(value):raise ValueError('Missing native quantity/unit for '+key)
            return float((float(value['value'])*u.Unit(value['unit'])).to_value(target_unit))
        if flame:
            # R3k quickstart has unit-bearing input Teff. The original R30k
            # notebook has a different scalar effective_temp convention;
            # that layout is deliberately not accepted by this adapter.
            measured.update(teff_K=quantity('effective_temp',u.K),logg_cgs=math.log10(quantity('gravity',u.cm/u.s**2)),metallicity_dex=float(params['logmh']))
            co=params.get('cto',{})
            if not isinstance(co,dict) or 'relative' not in str(co.get('unit','')).lower() or '0.458' not in str(co['unit']):
                raise ValueError('Flame Skimmer quickstart C/O must be explicitly relative to solar (=0.458).')
            measured['co_relative_solar']=float(co['value'])
        else:
            measured.update(teff_K=quantity('teff',u.K),logg_cgs=math.log10(quantity('logg',u.cm/u.s**2)),
                            metallicity_dex=float(params['mh']),co_relative_solar=float(params['cto']),logKzz_cgs=float(params['logkzz']))
        if 'logKzz_cgs' in filename and flame:
            # Different release writers use logkzz or logKzz. Verify when a
            # scalar is present; otherwise the unchanged filename is authority.
            for key in ('logkzz','logKzz'):
                if key in params and isinstance(params[key],(float,int)):measured['logKzz_cgs']=float(params[key])
    extra={'parameters_verified_from_filename':filename,'parameters_verified_from_header':measured,
           'native_headers':headers,'native_units_verified_from_header':True,
           'co_convention':'C/O divided by solar C/O (0.458); not the absolute atomic ratio.',
           'source_release':'Flame Skimmer v1 R3000 quickstart' if flame else 'Elf Owl v2 (user-selected release; the file itself does not independently establish version)',
           'source_record_url':'https://zenodo.org/records/20030439' if flame else 'https://zenodo.org/records/15150874',
           'native_resolving_power':3000 if flame else None}
    return np.asarray(w,float),np.asarray(f,float),'um','erg_cm2_s_cm',extra

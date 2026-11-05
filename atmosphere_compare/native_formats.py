"""Additional published emission formats; no transmission-depth spectra accepted."""
from pathlib import Path
import io
import re
import gzip
import math
import numpy as np


def two_column_text(path):
    text=gzip.open(path,'rt').read() if path.suffix=='.gz' else path.read_text()
    text=text.replace('D+','E+').replace('D-','E-');lines=text.splitlines()
    for i,line in enumerate(lines[:40]):
        tokens=line.split()
        if len(tokens)==2:
            try:a,b=map(float,tokens)
            except ValueError:continue
            if a>0 and b>=0:return np.loadtxt(io.StringIO(text),skiprows=i,ndmin=2),lines[:i]
    raise ValueError('No two-column spectrum found below the native header.')


def load_extra(path,fmt,spec):
    filename={};header_parameters={};extra={}
    if fmt in ('sonora_elf_owl_v2','sonora_flame_skimmer_v1_r3k'):
        from .netcdf_models import load_netcdf
        return load_netcdf(path,fmt)
    if fmt in ('sonora_cholla','sonora_diamondback','atmo2020_ascii'):
        data,header=two_column_text(path);wave,flux=data[:,0],data[:,1]
        compact=''.join(' '.join(header).lower().split()).replace('^','')
        if fmt in ('sonora_cholla','sonora_diamondback'):
            # Match the complete spectral-density unit, not its prefix:
            # W/m2/micron and W/m2/m are different by a factor of 10^6.
            unit_text=re.sub(r'\s*/\s*','/',' '.join(header).lower().replace('^',''))
            flux_units=re.findall(r'(?<![a-z0-9])w/(?:m2|m²)/([a-z0-9µμ/]+)',unit_text)
            if not ('micron' in compact or 'wavelength(um)' in compact) or not flux_units or any(unit!='m' for unit in flux_units):
                raise ValueError('Expected native micron and W/m2/m emission header. Supply an explicit ASCII adapter for a different release.')
            wave_unit,flux_unit='um','W_m2_m'
            name=path.name.removesuffix('.gz') if hasattr(str,'removesuffix') else path.name[:-3] if path.name.endswith('.gz') else path.name
            if fmt=='sonora_cholla':
                match=re.fullmatch(r'(\d+)K_(\d+(?:\.\d+)?)g_logkzz(\d+)\.spec',name)
                if not match:raise ValueError('Unrecognized Cholla filename; retain its published model filename.')
                t,g,k=map(float,match.groups())
                filename={'teff_K':t,'logg_cgs':math.log10(g*100),'logKzz_cgs':k,'metallicity_dex':0.}
            else:
                match=re.fullmatch(r't(\d+)g(\d+(?:\.\d+)?)(f\d+(?:\.\d+)?|nc)_m([+-]?\d+(?:\.\d+)?)_co(\d+(?:\.\d+)?)(?:\.[^.]+)?',name)
                if not match:raise ValueError('Unrecognized Diamondback filename; retain the published t/g/fsed/metal/C-O filename.')
                t,g,cloud,m,co=match.groups()
                filename={'teff_K':float(t),'logg_cgs':math.log10(float(g)*100),'fsed':'cloud_free' if cloud=='nc' else float(cloud[1:]),'metallicity_dex':float(m),'co_relative_solar':float(co)}
                extra.update(co_convention='C/O relative to solar (solar C/O=0.458).',source_release='Sonora Diamondback',source_record_url='https://zenodo.org/records/12735103')
        else:
            # Original Phillips 2020 emission output, NOT the STScI conversion.
            wave_unit,flux_unit='um','W_m2_um'
            if header and not (("micron" in compact or "wavelength[um]" in compact) and ("w/m2/micron" in compact or "w/m2/um" in compact)):
                raise ValueError('Original ATMO emission header must specify micron and W/m2/micron. Use explicit ASCII units for a converted release.')
            match=re.fullmatch(r'spec_t(\d+)_lg([\d.]+)_(ceq|neq_weak|neq_strong)\.txt',path.name,re.IGNORECASE)
            if match:filename={'teff_K':float(match[1]),'logg_cgs':float(match[2]),'chemistry':match[3].lower()}
            extra['native_units_verified_from_header']=bool(header)
            extra['unit_authority']='Original ATMO 2020 emission layout documented by STScI; user must select this original release explicitly.'
        extra['original_header']=header
    elif fmt=='atmo2020_stsci_fits':
        from astropy.io import fits
        from astropy import units as u
        with fits.open(path,memmap=False) as hdus:
            tables=[h for h in hdus if isinstance(h,(fits.BinTableHDU,fits.TableHDU)) and h.data is not None]
            if len(tables)!=1:raise ValueError('Expected one ATMO STScI spectral table.')
            table=tables[0];names={n.lower():n for n in table.columns.names}
            if not {'wavelength','flux'}.issubset(names):raise ValueError('ATMO STScI table needs WAVELENGTH and FLUX.')
            wname,fname=names['wavelength'],names['flux']
            wu=table.columns[wname].unit;fu=table.columns[fname].unit
            if not wu or not fu:raise ValueError('FITS spectral column units are required.')
            # STScI's published files use the synphot aliases ANGSTROMS and FLAM.
            wunit=u.AA if wu.strip().upper()=='ANGSTROMS' else u.Unit(wu)
            funit=u.erg/u.s/u.cm**2/u.AA if fu.strip().upper()=='FLAM' else u.Unit(fu)
            wave=(np.asarray(table.data[wname],float)*wunit).to_value(u.um)
            flux=(np.asarray(table.data[fname],float)*funit).to_value(u.W/u.m**2/u.um)
            extra['fits_column_units']={'wavelength':wu,'flux':fu}
            for key,candidates in {'teff_K':('TEFF','PHXTEFF'),'logg_cgs':('LOGG','PHXLOGG')}.items():
                for h in hdus:
                    for candidate in candidates:
                        if candidate in h.header:header_parameters[key]=float(h.header[candidate])
        wave_unit,flux_unit='um','W_m2_um'
        match=re.fullmatch(r'spec_t(\d+)_lg([\d.]+)_(ceq|neq_weak|neq_strong)\.fits',path.name)
        if match:filename={'teff_K':float(match[1]),'logg_cgs':float(match[2]),'metallicity_dex':0.,'chemistry':match[3]}
    elif fmt=='bt_settl_svo':
        from astropy.table import Table
        from astropy import units as u
        if spec.get('wavelength_frame')!='air':raise ValueError('SVO BT-Settl CIFIST uses air wavelengths. Choose air explicitly; this adapter does not convert frames.')
        table=Table.read(path,format='votable');names={n.lower():n for n in table.colnames}
        if not {'wavelength','flux'}.issubset(names):raise ValueError('Expected SVO emission VOTable Wavelength and Flux columns.')
        wcol,fcol=table[names['wavelength']],table[names['flux']]
        if wcol.unit is None or fcol.unit is None:raise ValueError('SVO wavelength and surface-flux units are required.')
        wave=u.Quantity(wcol).to_value(u.um);flux=u.Quantity(fcol).to_value(u.W/u.m**2/u.um)
        wave_unit,flux_unit='um','W_m2_um'
        extra['unit_authority']='SVO VOTable column units; BT-Settl CIFIST air convention. Atmospheric parameters are user declarations unless separately verified.'
    else:raise ValueError('Unsupported additional adapter.')
    extra.update(parameters_verified_from_filename=filename,parameters_verified_from_header=header_parameters)
    return wave,flux,wave_unit,flux_unit,extra

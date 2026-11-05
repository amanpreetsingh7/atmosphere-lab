"""STScI-resampled BT-Settl CIFIST2011_2015 single-gravity FITS tables.

These derived spectra differ from the high-resolution PHOENIX starter and
from SVO VOTables. Preserve the source filename, units and distribution label.
"""
import math
import re
import numpy as np


def read_bt_settl_stsci(path,spec):
    from astropy.io import fits
    match=re.fullmatch(r'phoenixm0\.0_(\d+)_5\.0_2011\.fits',path.name)
    if not match:raise ValueError('Retain the published solar STScI BT-Settl filename: phoenixm0.0_T_5.0_2011.fits.')
    named={'teff_K':float(match[1]),'logg_cgs':5.,'metallicity_dex':0.}
    if named['teff_K'] not in range(1200,3501,100):raise ValueError('Temperature is outside this STScI-resampled release.')
    if spec.get('wavelength_frame','vacuum')!='vacuum':raise ValueError('This integration adopts vacuum coordinates for this distribution; converted spectra require a separate explicit adapter.')
    with fits.open(path,memmap=False) as hdus:
        if 'TEFF' not in hdus[0].header or not math.isclose(float(hdus[0].header['TEFF']),named['teff_K'],abs_tol=1e-6):raise ValueError('BT-Settl filename/header temperature mismatch.')
        tables=[h for h in hdus if isinstance(h,fits.BinTableHDU)]
        if len(tables)!=1:raise ValueError('Expected one BT-Settl binary table.')
        table=tables[0];names={n.lower():n for n in table.columns.names}
        if set(names)!={'wavelength','flux'}:raise ValueError('Expected native WAVELENGTH and flux columns.')
        wn,fn=names['wavelength'],names['flux'];wu,fu=table.columns[wn].unit,table.columns[fn].unit
        if not wu or wu.upper().strip() not in ('ANGSTROM','ANGSTROMS') or not fu or fu.upper().strip()!='FLAM':raise ValueError('BT-Settl native units must be Angstrom and FLAM.')
        wave=np.asarray(table.data[wn],float);flux=np.asarray(table.data[fn],float)
    if len(wave)<3 or not np.any(flux>0):raise ValueError('BT-Settl spectrum has no usable emission.')
    return wave,flux,{'parameters_verified_from_filename':named,
        'parameters_verified_from_header':{'teff_K':named['teff_K']},
        'native_units_verified_from_header':True,'fits_column_units':{'wavelength':wu,'flux':fu},
        'source_release':'STScI-resampled BT-Settl CIFIST2011_2015 (2011 filename suffix)',
        'source_record_url':'https://ssb.stsci.edu/trds/grid/phoenixBTS11_15/AA_README_PHOENIXBTS11_15',
        'sampling_note':'STScI-derived resampling, not the original PHOENIX sampling. Published distribution: 300000 samples from approximately 10 Angstrom to 35 micron.',
        'surface_flux_authority':'PHOENIX BT-Settl atmosphere emission distributed by STScI as FLAM; no magnitude normalization or additional pi factor is applied. Check coverage-limited bolometric energy on installation.',
        'wavelength_frame_authority':'Vacuum convention adopted for this reader; the STScI FITS headers/README do not independently state air/vacuum. This is not the SVO air-coordinate adapter.'}

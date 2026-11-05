"""STScI's Castelli-Kurucz ATLAS9 surface-flux tables; exact native columns."""
import math
import re
import numpy as np


def read_ck04(path, spec):
    from astropy.io import fits
    from astropy import units as u
    match = re.fullmatch(r'ck([mp])(\d{2})_(\d+)\.fits', path.name)
    if not match:
        raise ValueError('Keep the published CK04 cksmh_temperature.fits filename.')
    sign, metal, temperature = match.groups()
    named = {'teff_K': float(temperature),
             'metallicity_dex': (1 if sign == 'p' else -1) * float(metal) / 10}
    column = spec.get('gravity_column', 'g45')
    if not re.fullmatch(r'g\d{2}', column):
        raise ValueError('Choose an exact CK04 gravity column, for example g45.')
    with fits.open(path, memmap=False) as hdus:
        h = hdus[0].header
        if 'TEFF' not in h or 'LOG_Z' not in h:
            raise ValueError('CK04 native TEFF and LOG_Z headers are required.')
        measured = {'teff_K': float(h['TEFF']), 'metallicity_dex': float(h['LOG_Z']),
                    'logg_cgs': float(column[1:]) / 10}
        for key, value in named.items():
            if not math.isclose(value, measured[key], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError('CK04 filename/header parameter mismatch for ' + key)
        tables = [x for x in hdus if isinstance(x, fits.BinTableHDU)]
        if len(tables) != 1:
            raise ValueError('Expected one CK04 native binary table.')
        table = tables[0]
        names = {n.lower(): n for n in table.columns.names}
        if 'wavelength' not in names or column not in names:
            raise ValueError('Missing wavelength or requested native gravity column.')
        wname, fname = names['wavelength'], names[column]
        wu, fu = table.columns[wname].unit, table.columns[fname].unit
        if not wu or not fu:
            raise ValueError('CK04 wavelength and flux units are required.')
        wunit = u.AA if wu.upper() in ('ANGSTROM', 'ANGSTROMS') else u.Unit(wu)
        funit = u.erg/u.s/u.cm**2/u.AA if fu.upper() == 'FLAM' else u.Unit(fu)
        if not math.isclose(wunit.to(u.AA), 1., rel_tol=1e-12):
            raise ValueError('Native CK04 wavelengths must be Angstroms.')
        if not math.isclose(funit.to(u.erg/u.s/u.cm**2/u.AA), 1., rel_tol=1e-12):
            raise ValueError('Native CK04 flux must be FLAM surface flux.')
        wave = np.asarray(table.data[wname], float)
        flux = np.asarray(table.data[fname], float)
        if not np.any(flux > 0):
            raise ValueError('This CK04 gravity column is unavailable (all zero).')
    return wave, flux, {'parameters_verified_from_filename': named,
        'parameters_verified_from_header': measured,
        'native_units_verified_from_header': True,
        'selected_gravity_column': column, 'source_release': 'STScI CK04 ATLAS9',
        'fits_column_units': {'wavelength': wu, 'flux': fu},
        'surface_flux_authority': 'STScI native HISTORY and CK04 atlas documentation; scale observed flux by (R/D)^2.',
        'wavelength_frame_authority': 'Vacuum convention adopted for this integration; the FITS table does not independently specify air/vacuum.',
        'source_record_url': 'https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/castelli-and-kurucz-atlas'}

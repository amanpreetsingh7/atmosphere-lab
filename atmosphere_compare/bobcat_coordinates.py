"""Native Bobcat filename indices and reported physical coordinates.

The October 16 laptop inventory contains five +1 K filename/header offsets.
Filename temperatures identify mesh connectivity only; reported temperatures
and gravities remain the physical coordinates. No spectrum is renormalized.
"""
import math
import re


MAX_TEMPERATURE_OFFSET_K = 1.0
MAX_GRAVITY_LABEL_OFFSET_M_S2 = 1.0


def grid_coordinates(filename, reported):
    match = re.fullmatch(
        r'sp_t(\d+)g(\d+(?:\.\d+)?)nc_m([+-]?\d+(?:\.\d+)?)(?:_co(\d+(?:\.\d+)?))?(?:\.gz)?',
        filename)
    if not match:
        return None
    nominal_temperature, filename_gravity, metallicity = map(float, match.groups()[:3])
    relative_co = float(match[4]) if match[4] is not None else 1.0
    actual_temperature = float(reported['teff_K'])
    actual_gravity = 10.**float(reported['logg_cgs'])/100.
    if abs(actual_temperature-nominal_temperature) > MAX_TEMPERATURE_OFFSET_K+1e-9:
        raise ValueError('Bobcat '+filename+': filename temperature %g K, reported header %g K; allowed identity offset is at most %g K.'%
                         (nominal_temperature, actual_temperature, MAX_TEMPERATURE_OFFSET_K))
    if abs(actual_gravity-filename_gravity) > MAX_GRAVITY_LABEL_OFFSET_M_S2+1e-9:
        raise ValueError('Bobcat '+filename+': filename gravity %g m/s2, reported header %g m/s2; the label/header difference exceeds %g m/s2.'%
                         (filename_gravity, actual_gravity, MAX_GRAVITY_LABEL_OFFSET_M_S2))
    for key, value in (('metallicity_dex',metallicity),('co_relative_solar',relative_co)):
        if not math.isclose(float(reported[key]),value,rel_tol=1e-8,abs_tol=1e-8):
            raise ValueError('Bobcat '+filename+': filename '+key+'=%g disagrees with reported header %g.'%(value,reported[key]))
    return {'nominal_teff_K':nominal_temperature, 'filename_gravity_m_s2':filename_gravity,
            'reported_teff_K':actual_temperature, 'reported_gravity_m_s2':actual_gravity,
            'temperature_header_minus_filename_K':actual_temperature-nominal_temperature,
            'meaning':'Filename temperature is a connectivity index only. Interpolation and displayed physical parameters retain reported header temperature and gravity.'}

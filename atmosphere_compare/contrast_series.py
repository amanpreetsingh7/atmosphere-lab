"""Exact native temperature series and portable brightness-only exports."""
from pathlib import Path
import copy
import csv
import hashlib
import io
import json
import math
import zipfile
import numpy as np
from .properties import ModelLibrary,predict_from_properties,response_from_request


def available_nodes(library,companion):
    held={k:v for k,v in companion['parameters'].items() if k!='teff_K'}
    rows=[r for r in library.rows if r['role']=='planet' and r['family']==companion['family']
          and set(r['parameters'])==set(held)|{'teff_K'}
          and all(r['parameters'][k]==v for k,v in held.items())]
    nodes={}
    for row in rows:
        t=float(row['parameters']['teff_K'])
        if t in nodes:raise ValueError('Duplicate native temperature at the held parameters: %g K.'%t)
        nodes[t]=row
    return nodes


def source_signature(library,response,base):
    tokens=[library.sha256]
    for row in library.rows:
        for key in ('file','wavelength_file'):
            if key in row['spectrum']:
                path=(library.path.parent/Path(row['spectrum'][key]).expanduser()).resolve()
                stat=path.stat() if path.is_file() else None
                tokens.append([str(path),stat.st_size if stat else None,stat.st_mtime_ns if stat else None])
    if isinstance(response,dict):
        paths=[Path(base)/Path(response[k]['file']).expanduser() for k in ('filter','sky')]
        for path in paths:
            stat=path.stat();tokens.append([str(path.resolve()),stat.st_size,stat.st_mtime_ns])
    else:tokens.append(response.metadata)
    return hashlib.sha256(json.dumps(tokens,sort_keys=True).encode()).hexdigest()


def calculate_series(catalog,star,companion,response,temperatures,radii,*,radius_unit='R_jup',base=None,progress=None):
    library=ModelLibrary(catalog);base=Path(base) if base is not None else library.path.parent
    temperatures=list(map(float,temperatures));radii=list(map(float,radii))
    if not temperatures or not radii or len(radii)>4 or len(temperatures)*len(radii)>600:raise ValueError('Choose 1-600 temperature/radius combinations.')
    if len(set(temperatures))!=len(temperatures) or len(set(radii))!=len(radii):raise ValueError('Temperatures and radii must be unique.')
    if not np.isfinite(temperatures+radii).all() or min(temperatures+radii)<=0:raise ValueError('Temperatures and radii must be finite and positive.')
    nodes=available_nodes(library,companion)
    if any(t not in nodes for t in temperatures):raise ValueError('Every selected temperature must be an exact native node at the held companion parameters.')
    # The star may explicitly use the existing bounded interpolation API.
    before=source_signature(library,response,base);band=response_from_request(response,base)
    rows=[];sources=[];star_source=None
    for i,t in enumerate(sorted(temperatures)):
        body=copy.deepcopy(companion);body['selection']='exact';body['parameters']['teff_K']=t
        body['radius']={'value':1.,'unit':radius_unit}
        result=predict_from_properties(catalog,star,body,band,base=base,samples=31,library=library)
        if result['sampling_counts']['planet']['construction']!='native_exact':raise ValueError('The series requires exact native companion spectra.')
        star_source=result['source_spectra']['star']
        sources.append({'temperature_K':t,'source':result['source_spectra']['planet'],'contrast_at_unit_radius':result['planet_star_contrast']})
        for radius in radii:
            rows.append({'temperature_K':t,'radius_value':radius,'radius_unit':radius_unit,'contrast':result['planet_star_contrast']*radius**2})
        if progress:progress(i+1,len(temperatures))
    if ModelLibrary(catalog).sha256!=library.sha256 or source_signature(library,response,base)!=before:raise ValueError('Model/catalog/response files changed during calculation; calculate again.')
    for row in rows:
        if not math.isfinite(row['contrast']) or row['contrast']<=0:raise ValueError('Series contains an invalid contrast.')
    provenance={'star':copy.deepcopy(star),'planet_family':companion['family'],
        'held_planet_parameters':{k:v for k,v in companion['parameters'].items() if k!='teff_K'},
        'band':dict(result['band'],limits_um=list(band.limits_um)),'detector':'photon','catalog_sha256':library.sha256,
        'source_signature':before,'star_source':star_source,'temperatures_and_sources':sources,
        'companion_selection':'exact','radius_unit':radius_unit,
        'scope':'Calculated surface-atmosphere brightness at imposed radii. No irradiation, reflected light, measured sensitivity, fixed-mass evolution sequence or detection probability.'}
    return {'rows':rows,'provenance':provenance}


def calculate_heated_series(catalog,star,companion,response,temperatures,radii,*,distance_pc,
                            bond_albedo=.3,true_to_projected=1.,separations=(.1,1.),samples=301,
                            radius_unit='R_jup',base=None,progress=None):
    from .heated_series import attach_heating
    library=ModelLibrary(catalog)
    # Every installed matching node is needed as potential heated-temperature
    # support, even when the requested Tmin curves use only a few temperatures.
    nodes=sorted(available_nodes(library,companion))
    data=calculate_series(catalog,star,companion,response,nodes,radii,radius_unit=radius_unit,base=base,progress=progress)
    return attach_heating(data,temperatures,distance_pc=distance_pc,bond_albedo=bond_albedo,
        true_to_projected=true_to_projected,separations=separations,samples=samples)


def _scrub(value):
    if isinstance(value,dict):return {k:_scrub(v) for k,v in value.items()}
    if isinstance(value,list):return [_scrub(v) for v in value]
    if isinstance(value,str) and Path(value).is_absolute():return Path(value).name
    return value


def export_series(root,series,separations=(.1,1.)):
    from datetime import datetime
    from contrast_series_plot import draw_series
    folder=Path(root)/'results'/('contrast_series_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));folder.mkdir(parents=True)
    data=_scrub(series);data['separation_display_arcsec']=list(separations)
    (folder/'series.json').write_text(json.dumps(data,indent=2,allow_nan=False))
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=['temperature_K','radius_value','radius_unit','contrast']);writer.writeheader();writer.writerows(data['rows'])
    (folder/'contrasts.csv').write_text(stream.getvalue())
    if 'heated_separation' in data:
        from .heated_series import heated_csv_rows
        stream=io.StringIO();rows=list(heated_csv_rows(data))
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (folder/'heated_contrasts.csv').write_text(stream.getvalue())
    draw_series(data,folder)
    heating_note=('Heated curves and heated_contrasts.csv use Tmin for temperature without heating. '+data['heated_separation']['settings']['scope']+'\n') if 'heated_separation' in data else ''
    (folder/'README.txt').write_text('Figures and tabulated results from the declared atmosphere and response inputs.\nNo native spectrum files or observations are bundled.\nNative unheated table: '+data['provenance']['scope']+'\n'+heating_note)
    archive=folder.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(folder.iterdir()):z.write(path,path.name)
    return archive

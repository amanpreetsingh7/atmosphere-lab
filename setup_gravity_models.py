"""Index the official archive, then install a bounded temperature/gravity slice."""
from pathlib import Path
import json
import re
import tarfile
from atmosphere_compare.downloads import download,extract_selected,BOBCAT_URL,BOBCAT_BYTES,BOBCAT_MD5
from atmosphere_compare.spectra import load_spectrum
from setup_real_models import backup_and_write

ROOT=Path(__file__).resolve().parent


def main(gravities=(31,56,100,178,316,562,1000)):
    archive=download(BOBCAT_URL,ROOT/'models/archives/spectra_m+0.0.tar.gz',BOBCAT_BYTES,BOBCAT_MD5)
    destinations={};selections={}
    print('Indexing actual archive members; selecting 200-2400 K and g =',gravities,'m/s2.',flush=True)
    with tarfile.open(archive,'r|gz') as source:
        for member in source:
            match=re.fullmatch(r'spectra/sp_t(\d+)g(\d+)nc_m0\.0\.gz',member.name)
            if not match or not member.isfile():continue
            t,g=map(int,match.groups())
            if 200<=t<=2400 and g in gravities:
                destinations[member.name]=ROOT/'models/bobcat'/Path(member.name).name
                selections[member.name]=(t,g)
    if not destinations:raise ValueError('No matching models in this archive release.')
    print('Found',len(destinations),'real grid members.',flush=True)
    extract_selected(archive,destinations)
    rows=[];header_offsets=[]
    for index,(member,path) in enumerate(destinations.items()):
        spec={'format':'sonora_bobcat','file':str(path),'wavelength_frame':'vacuum','source_url':'https://zenodo.org/records/5063476'}
        model=load_spectrum(spec,ROOT);p=model.metadata['parameters_verified_from_header']
        header=' '.join(model.metadata['original_header']).lower()
        if not all(k in p for k in ('teff_K','logg_cgs','metallicity_dex','co_relative_solar')) or 'erg/cm^2/s/hz' not in header:
            raise ValueError('Native units or parameters could not be verified: '+member)
        coordinates=model.metadata.get('grid_coordinates')
        if not coordinates or coordinates['nominal_teff_K']!=selections[member][0] or coordinates['filename_gravity_m_s2']!=selections[member][1]:
            raise ValueError('Archive member and parsed grid identity disagree: '+member)
        if p['metallicity_dex']!=0 or p['co_relative_solar']!=1:
            raise ValueError('Unexpected model composition: '+member+'; header='+str(p))
        if coordinates['temperature_header_minus_filename_K']:
            header_offsets.append({'member':member,'grid_coordinates':coordinates})
        spec.update(file='../models/bobcat/'+path.name,label='Bobcat: header %g K, log g %.6f'%(p['teff_K'],p['logg_cgs']))
        rows.append({'role':'planet','family':'sonora_bobcat','parameters':p,'spectrum':spec,'grid_coordinates':coordinates})
        if (index+1)%10==0:print('Verified',index+1,'/',len(destinations),flush=True)
    from atmosphere_compare.model_library import upsert_catalog
    upsert_catalog(ROOT,rows,replace=True,replace_source_files=True)
    (ROOT/'models/gravity_installation.json').write_text(json.dumps({'models':len(rows),'archive_md5':BOBCAT_MD5,
        'header_temperature_offsets':header_offsets,
        'coordinates':'Physical temperatures and gravity from headers; nominal filename temperature connects native mesh nodes only.',
        'scope':'Native parameter/unit/identity checks. Band interpolation must still be assessed against held-out physical spectra.'},indent=2))
    print('Retained reported temperatures for',len(header_offsets),'filename/header offsets; no spectrum relabeled or renormalized.',flush=True)
    print('READY: temperature/gravity atmospheres installed. Refresh selectors.',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:print('GRAVITY INSTALL STOPPED:',exc,flush=True);raise SystemExit(1)

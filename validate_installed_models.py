"""Background validation job started from the app, running on the laptop."""
from pathlib import Path
from datetime import datetime
import json
import hashlib
import platform
import zipfile
from importlib.metadata import version,PackageNotFoundError
from atmosphere_compare import __version__
from atmosphere_compare.validation import audit_catalog

ROOT=Path(__file__).resolve().parent


def software_record():
    code=Path(__file__).resolve().parent
    files=('spectra.py','ck04.py','bandpass.py','properties.py','bt_settl_stsci.py','contrast_series.py','heated_series.py','native_formats.py','netcdf_models.py','parameter_exploration.py','bobcat_coordinates.py','compare.py','grid.py','separation.py','validation.py','evolution.py','inference.py')
    def installed_version(name):
        try:return version(name)
        except PackageNotFoundError:return 'not installed'
    return {'software_version':__version__,'python_version':platform.python_version(),
        'package_versions':{name:installed_version(name) for name in ('numpy','scipy','astropy','netCDF4')},
        'source_code_sha256':{'atmosphere_compare/'+name:hashlib.sha256((code/'atmosphere_compare'/name).read_bytes()).hexdigest() for name in files}}


def main():
    request=ROOT/'configs/validation_request.json'
    settings=json.loads(request.read_text()) if request.exists() else {}
    bands=settings.get('bands',[])
    catalogs=settings.get('catalogs',[str(ROOT/'configs/app_catalog.json')])
    reports=[audit_catalog(p,bands) for p in sorted(set(catalogs))]
    result=dict(software_record(),created_local=datetime.now().isoformat(),reports=reports,responses=bands,
        status='completed_with_load_errors' if any(r['load_errors'] for r in reports) else 'diagnostics_completed',
        note='Review the actual interpolation errors and physical assumptions before using results scientifically. No universal accuracy cutoff is imposed.')
    folder=ROOT/'results';folder.mkdir(exist_ok=True)
    output=folder/('installed_model_validation_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip')
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('validation.json',json.dumps(result,indent=2,allow_nan=False))
    (folder/'latest_validation.json').write_text(json.dumps({'path':str(output),'status':result['status'],'catalogs':catalogs,'bands':[b['name'] for b in bands]},indent=2))
    print('VALIDATION REPORT:',output,flush=True)
    print(result['status'],flush=True)
    if result['status']=='completed_with_load_errors':raise SystemExit(1)


if __name__=='__main__':
    try:main()
    except Exception as exc:print('VALIDATION STOPPED:',exc,flush=True);raise SystemExit(1)

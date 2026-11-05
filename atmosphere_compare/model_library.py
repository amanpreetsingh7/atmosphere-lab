"""Local model registration and one serialized, resumable model-install job."""
from pathlib import Path
from datetime import datetime
import json
import shutil
import subprocess
import sys
import threading
from .spectra import load_spectrum


def canonical_parameters(value):
    if isinstance(value,dict):return {k:canonical_parameters(v) for k,v in value.items()}
    if isinstance(value,list):return [canonical_parameters(v) for v in value]
    if isinstance(value,(int,float)) and not isinstance(value,bool):return float(value)
    return value


def register_model(root, role, family, spectrum, parameters, replace=False):
    root=Path(root)
    if role not in ('star','planet') or not family.strip() or not isinstance(parameters,dict):
        raise ValueError('Choose a role, model-family name and parameter object.')
    spec=dict(spectrum,parameters=parameters)
    loaded=load_spectrum(spec,root)
    params=dict(loaded.metadata.get('parameters_verified_from_filename',{}))
    params.update(loaded.metadata.get('parameters_verified_from_header',{}))
    params.update(parameters)
    mapping=dict(spectrum)
    mapping.pop('parameters',None)
    for key in ('file','wavelength_file'):
        if key in mapping:
            path=(root/mapping[key]).expanduser().resolve()
            try:
                mapping[key]='../'+str(path.relative_to(root))
            except ValueError:
                mapping[key]=str(path)
    row={'role':role,'family':family.strip(),'parameters':params,'spectrum':mapping}
    if loaded.metadata.get('grid_coordinates'):
        row['grid_coordinates']=loaded.metadata['grid_coordinates']
    upsert_catalog(root,[row],replace)
    return row,loaded.metadata


def register_folder(root,role,family,folder,pattern,adapter,parameters,settings,replace=False):
    """Validate every proposed row before one catalog commit; native identity required."""
    root=Path(root);folder=Path(folder).expanduser().resolve()
    if not folder.is_dir() or role not in ('star','planet') or not family.strip():raise ValueError('Choose an existing grid folder, role and family.')
    paths=sorted(p for p in folder.glob(pattern) if p.is_file() and not p.name.startswith('._'))
    if not paths or len(paths)>500:raise ValueError('Select between one and 500 spectrum files per import.')
    new=[]
    for path in paths:
        spec=dict(settings,format=adapter,file=str(path),parameters=parameters)
        loaded=load_spectrum(spec,root)
        params=dict(loaded.metadata.get('parameters_verified_from_filename',{}));params.update(loaded.metadata['parameters_verified_from_header']);params.update(parameters)
        if 'teff_K' not in params:raise ValueError('No verified/declared temperature for '+path.name+'. Import this file individually with its parameters.')
        spec.pop('parameters');spec['label']=family+': '+path.name
        row={'role':role,'family':family.strip(),'parameters':params,'spectrum':spec}
        if loaded.metadata.get('grid_coordinates'):
            row['grid_coordinates']=loaded.metadata['grid_coordinates']
        new.append(row)
    keys=[json.dumps(canonical_parameters(r['parameters']),sort_keys=True) for r in new]
    if len(set(keys))!=len(keys):raise ValueError('Imported files have duplicate parameters. Declare all distinguishing grid parameters before import.')
    upsert_catalog(root,new,replace)
    return len(new)


def upsert_catalog(root,new,replace=False,replace_source_files=False):
    """Serialize the read/merge/backup/write so concurrent registrations are retained."""
    if replace_source_files and not replace:
        raise ValueError('Replacing rows for the same source files requires explicit replace=True.')
    import fcntl
    catalog=Path(root)/'configs/app_catalog.json';catalog.parent.mkdir(parents=True,exist_ok=True)
    identity=lambda r:json.dumps([r.get('role'),r['family'],canonical_parameters(r['parameters'])],sort_keys=True)
    keys=[identity(r) for r in new]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate new catalog selections.')
    with catalog.with_suffix('.lock').open('a') as lockfile:
        fcntl.flock(lockfile.fileno(),fcntl.LOCK_EX)
        old=json.loads(catalog.read_text()) if catalog.exists() else []
        if not replace and any(identity(r) in keys for r in old):raise ValueError('This exact role/family/parameter selection already exists. Choose Replace exact selection to update it.')
        def source_identity(row):
            path=row.get('spectrum',{}).get('file')
            return (row.get('role'),row['family'],str((catalog.parent/Path(path).expanduser()).resolve())) if path else None
        replaced_sources={source_identity(row) for row in new} if replace_source_files else set()
        merged=[r for r in old if identity(r) not in keys and
                (not replace_source_files or source_identity(r) not in replaced_sources)]+new
        if catalog.exists():shutil.copy2(catalog,catalog.with_name('app_catalog.before_import_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json'))
        part=catalog.with_suffix('.json.part');part.write_text(json.dumps(merged,indent=2));part.replace(catalog)
    return merged


class ModelJob:
    ALLOWED={'ck04_star':'setup_ck04_star.py','starter':'setup_real_models.py','temperature_sequence':'setup_separation_models.py',
             'evolution':'setup_evolution_models.py','gravity_grid':'setup_gravity_models.py','cholla':'setup_cholla_models.py',
             'atmo':'setup_atmo_models.py','diamondback_parameters':('setup_parameter_models.py','diamondback'),
             'flame_parameters':('setup_parameter_models.py','flame'),'validation':'validate_installed_models.py','software_checks':'run_laptop_validation.py'}
    def __init__(self,root):
        self.root=Path(root)
        self.process=None
        self.log=None
        self.name=None
        self.cancelled=False
        self.lock=threading.Lock()

    def start(self,name,parameters=None):
        with self.lock:
            if name not in self.ALLOWED:
                raise ValueError('Unknown model installation job.')
            if self.process is not None and self.process.poll() is None:
                raise ValueError('A model download is already running. Finish or stop it before starting another.')
            entry=self.ALLOWED[name]
            arguments=list(entry[1:]) if isinstance(entry,tuple) else []
            if parameters is not None:
                import math
                if name == 'ck04_star':
                    if not isinstance(parameters, dict) or set(parameters) != {'teff_K', 'logg_cgs', 'metallicity_dex'}:
                        raise ValueError('CK04 job needs teff_K, logg_cgs and metallicity_dex.')
                    for key in ('teff_K', 'logg_cgs', 'metallicity_dex'):
                        value = float(parameters[key])
                        if not math.isfinite(value):raise ValueError('CK04 inputs must be finite.')
                        arguments += ['--'+key, str(value)]
                elif name not in ('diamondback_parameters','flame_parameters') or not isinstance(parameters,dict) or set(parameters)!={'temperature','gravity'}:
                    raise ValueError('Only parameter-subset jobs accept temperature and gravity arguments.')
                for key in (() if name == 'ck04_star' else ('temperature','gravity')):
                    value=float(parameters[key])
                    if not math.isfinite(value) or value<=0:raise ValueError('Subset coordinates must be positive and finite.')
                    arguments+=['--'+key,str(value)]
            script=self.root/(entry[0] if isinstance(entry,tuple) else entry)
            if not script.is_file():
                raise FileNotFoundError('Model installer is missing: '+str(script))
            directory=self.root/'models/install_logs'
            directory.mkdir(parents=True,exist_ok=True)
            import fcntl
            lockfile=(directory/'installation.lock').open('a')
            try:fcntl.flock(lockfile.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                lockfile.close()
                raise ValueError('Another installation still owns the model cache. Wait for it to finish before starting another.')
            self.log=directory/(name+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.log')
            try:
                with self.log.open('w') as output:
                    self.process=subprocess.Popen([sys.executable,'-u',str(script)]+arguments,cwd=str(self.root),
                        stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,pass_fds=(lockfile.fileno(),))
            finally:lockfile.close()
            self.name,self.cancelled=name,False

    def status(self):
        with self.lock:
            code=self.process.poll() if self.process is not None else None
            running=self.process is not None and code is None
            text=''
            if self.log and self.log.exists():
                with self.log.open('rb') as stream:
                    stream.seek(0,2)
                    stream.seek(max(0,stream.tell()-12000))
                    text=stream.read().decode('utf-8',errors='replace')
            return {'started':self.process is not None,'running':running,'exit_code':code,
                'name':self.name,'log':text,'cancelled':self.cancelled}

    def stop(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
                self.cancelled=True

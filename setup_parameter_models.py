"""Install small, pinned native subsets for cloud/composition comparisons."""
from pathlib import Path
import argparse
import json
import math
import zipfile
from atmosphere_compare.remote_zip import RangeFile,selected_members
from atmosphere_compare.model_library import register_folder
from atmosphere_compare.netcdf_models import filename_parameters

ROOT=Path(__file__).resolve().parent
SOURCES={
 'diamondback':dict(record='12735103',archive='spectra.zip',bytes=8560369066,
                   official_archive_md5='629ae5f8e7163169d6f07bfea05d6fc2',
                   adapter='sonora_diamondback',family='sonora_diamondback',count=8),
 'flame':dict(record='20030439',archive='spectra_R3k_quickstart.zip',bytes=8676930102,
             official_archive_md5='0fcdcc0370df324e5cc9b11f8d01ffb4',
             adapter='sonora_flame_skimmer_v1_r3k',family='sonora_flame_skimmer_v1_r3k',count=11)}


def wanted_members(names,kind,temperature=1000,gravity=100):
    """Exact native nodes; no nearest match or substitution for missing models."""
    if kind=='diamondback':
        stem='spectra/t%gg%g'%(temperature,gravity)
        wanted={stem+'%s_m0.0_co1.0.spec'%c for c in ('f1','f2','f3','f4','f8','nc')}
        wanted.update(stem+'f2_m%s_co1.0.spec'%m for m in ('-0.5','+0.5'))
        result=sorted(n for n in names if n in wanted)
    else:
        result=[]
        for n in names:
            if not n.startswith(('eq/','deq/')):continue
            try:p=filename_parameters(Path(n).name,'sonora_flame_skimmer_v1_r3k')
            except ValueError:continue
            if p['teff_K']!=temperature or not math.isclose(p['logg_cgs'],math.log10(gravity*100),abs_tol=1e-12):continue
            m,co=p['metallicity_dex'],p['co_relative_solar']
            if p['chemistry']=='equilibrium':
                take=(co==1 and m in (-.5,0,.5)) or (m==0 and co in (.5,1,1.5,2.5))
            else:take=m==0 and co==1 and p['logKzz_cgs'] in (2,4,7,8,9)
            if take:result.append(n)
        result.sort()
    if len(result)!=SOURCES[kind]['count'] or len(set(result))!=len(result):
        raise ValueError('Official archive does not contain the complete expected subset. Catalog unchanged.')
    return result


def main(kind,temperature=1000,gravity=100):
    source=SOURCES[kind]
    temperatures=range(900,2401,100) if kind=='diamondback' else (50,100,150,200,250,300,350,400,450,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2200,2400)
    gravities=(31,100,316,1000,3160) if kind=='diamondback' else (1,3.1,10,31,100,316,1000,3160)
    if temperature not in temperatures or gravity not in gravities:raise ValueError('Choose a published subset temperature and gravity; nearest-node substitution is disabled.')
    if kind=='flame':
        try:import netCDF4
        except ImportError as exc:raise ImportError('Install this update\'s requirements.txt before adding Flame Skimmer models.') from exc
    url='https://zenodo.org/records/%s/files/%s?download=1'%(source['record'],source['archive'])
    folder=ROOT/'models'/source['family']
    print('Reading official ZIP index; only selected models will be transferred.',flush=True)
    remote=RangeFile(url,source['bytes'])
    with zipfile.ZipFile(remote) as archive:
        names=wanted_members(archive.namelist(),kind,temperature,gravity)
        infos=[archive.getinfo(n) for n in names]
        print('Selected %d models: %.1f MB transfer plus index, %.1f MB model storage.'%
              (len(names),sum(i.compress_size for i in infos)/1e6,sum(i.file_size for i in infos)/1e6),flush=True)
        selected_members(archive,names,folder,dict(source,url=url))
    print('Checking native units and model identity before registering any selections.',flush=True)
    count=register_folder(ROOT,'planet',source['family'],folder,'*.spec' if kind=='diamondback' else '*.nc',
        source['adapter'],{},dict(wavelength_frame='vacuum',source_url=url),replace=True)
    print('READY: %d native models registered. Refresh model selectors.'%count,flush=True)
    print('Use Parameter exploration for controlled cloud, metallicity, C/O and mixing comparisons.',flush=True)
    print('Transfer: %.1f MB. Native diagnostics and plots run on this laptop.'%(remote.transferred/1e6),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('family',choices=tuple(SOURCES))
    parser.add_argument('--temperature',type=float,default=1000);parser.add_argument('--gravity',type=float,default=100);args=parser.parse_args()
    try:main(args.family,args.temperature,args.gravity)
    except Exception as exc:print('PARAMETER MODELS STOPPED:',exc,flush=True);raise SystemExit(1)

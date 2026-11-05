"""Download and import the official chemical-disequilibrium emission grid."""
from pathlib import Path
import re
import tarfile
from atmosphere_compare.downloads import download,extract_selected
from atmosphere_compare.model_library import register_folder

ROOT=Path(__file__).resolve().parent


def main():
    archive=download('https://zenodo.org/api/records/4450269/files/spectra.tar.gz/content',ROOT/'models/archives/cholla_spectra.tar.gz',443284127,'84013e450b762153cc5724521a2eb2ac')
    destinations={}
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            if member.isfile() and re.fullmatch(r'\d+K_\d+(?:\.\d+)?g_logkzz\d+\.spec',Path(member.name).name):
                name=Path(member.name).name
                if any(p.name==name for p in destinations.values()):raise ValueError('Duplicate Cholla spectrum basenames in archive.')
                destinations[member.name]=ROOT/'models/cholla'/name
    if not destinations:raise ValueError('No recognized Cholla emission members in archive.')
    extract_selected(archive,destinations)
    count=register_folder(ROOT,'planet','sonora_cholla',ROOT/'models/cholla','*.spec','sonora_cholla',{},
        {'wavelength_frame':'vacuum','source_url':'https://zenodo.org/records/4450269'},replace=True)
    print('READY:',count,'Cholla spectra registered; refresh selectors.',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:print('CHOLLA INSTALL STOPPED:',exc,flush=True);raise SystemExit(1)

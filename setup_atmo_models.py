"""Install the official STScI ATMO 2020 equilibrium emission release."""
from pathlib import Path
import re
import urllib.request
from atmosphere_compare.downloads import download,tls_context
from atmosphere_compare.model_library import register_folder

ROOT=Path(__file__).resolve().parent
URL='https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/atmo2020/'


def main():
    with urllib.request.urlopen(URL,timeout=30,context=tls_context()) as response:listing=response.read(2_000_000).decode()
    names=sorted(set(re.findall(r'href=["\'](spec_t\d+_lg5\.0_ceq\.fits)["\']',listing)))
    if not names:raise ValueError('No documented ATMO equilibrium FITS entries in official directory.')
    for index,name in enumerate(names):
        url=URL+name
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30,context=tls_context()) as response:
            size=int(response.headers.get('Content-Length','0'))
        if not 5760<=size<=10_000_000:raise ValueError('Unexpected official FITS size for '+name)
        print('ATMO download',index+1,'/',len(names),flush=True)
        download(url,ROOT/'models/atmo2020_stsci'/name,size)
    count=register_folder(ROOT,'planet','atmo2020_stsci_equilibrium',ROOT/'models/atmo2020_stsci','spec_t*_lg5.0_ceq.fits',
        'atmo2020_stsci_fits',{}, {'wavelength_frame':'vacuum','source_url':URL},replace=True)
    print('READY:',count,'ATMO equilibrium spectra at log g=5; refresh selectors.',flush=True)
    print('FITS units/native filenames checked; atmosphere science still needs local diagnostics.',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:print('ATMO INSTALL STOPPED:',exc,flush=True);raise SystemExit(1)

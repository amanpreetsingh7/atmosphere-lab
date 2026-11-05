"""Small analytic spectra for UI checks; never installed as science models."""
from pathlib import Path
import json
import numpy as np
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def write_fixture_catalog(path, temperatures=(1000,1500)):
    path=Path(path)
    w=np.linspace(.5,6,2000)
    rows=[]
    for role,t in [('star',5000)]+[('planet',t) for t in temperatures]:
        name='%s_%s.dat'%(role,t)
        signal=(w+1)*100 if role=='star' else np.ones(len(w))*(t/1000)**4
        np.savetxt(path/name,np.column_stack((w,signal)))
        rows.append({'role':role,'family':'analytic_fixture','parameters':{'teff_K':t,'logg_cgs':4.0,'metallicity_dex':0},
            'spectrum':{'format':'ascii','file':name,'flux_unit':'W_m2_um','wavelength_unit':'um','flux_convention':'surface_flux',
                        'label':'Synthetic test fixture %s %s'%(role,t)}})
    catalog=path/'catalog.json'
    catalog.write_text(json.dumps(rows))
    return catalog


def load_app(catalog):
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=45).run()
    at.text_input(key='star_catalog_path').set_value(str(catalog)).run()
    at.text_input(key='planet_catalog_path').set_value(str(catalog)).run()
    at.button(key='run').click().run()
    return at

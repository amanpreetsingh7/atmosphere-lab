"""Real-file diagnostics executed only on the user's computer via the app job."""
from pathlib import Path
from itertools import groupby
import json
import numpy as np
from .spectra import load_spectrum
from .compare import band_signal
from .separation import interpolate_band_signal
from .grid import BandGrid


def audit_catalog(path,bands=()):
    path=Path(path);rows=json.loads(path.read_text());records=[];errors=[]
    for index,row in enumerate(rows):
        print('Checking installed model',index+1,'/',len(rows),flush=True)
        try:
            spec=load_spectrum(dict(row['spectrum'],parameters=row['parameters']),path.parent)
            record={'role':row.get('role'),'family':row['family'],'parameters':row['parameters'],'metadata':spec.metadata,'bands':{}}
            if spec.metadata.get('grid_coordinates'):
                record['grid_coordinates']=spec.metadata['grid_coordinates']
            # Coverage-limited integral; not renormalized or a pass/fail energy criterion.
            area=float(np.sum(np.diff(spec.wavelength_um)*(spec.surface_flux[:-1]+spec.surface_flux[1:])/2))
            record['native_coverage_flux_W_m2']=area
            if 'teff_K' in row['parameters']:
                record['native_coverage_flux_over_sigma_Teff4']=area/(5.670374419e-8*float(row['parameters']['teff_K'])**4)
            for band in bands:
                if band.get('wavelength_frame','vacuum')!=spec.metadata['wavelength_frame']:
                    record['bands'][band['name']]={'unsupported':'Wavelength convention mismatch'};continue
                response=band['response']
                try:
                    signal=band_signal(spec,response['wavelength_um'],response['throughput'],band['detector'])
                    record['bands'][band['name']]={'signal':signal}
                except ValueError as exc:record['bands'][band['name']]={'unsupported':str(exc)}
            records.append(record)
        except (OSError,ValueError,TypeError,KeyError,ImportError) as exc:
            errors.append({'index':index,'model':row,'error':str(exc)})
    heldout=[];gravity_tests=[];unsupported_gravity_tests=[]
    # The omitted middle temperature is predicted only from its two neighbors.
    for band in bands:
        groups={}
        for record in records:
            result=record['bands'].get(band['name'],{})
            p=record['parameters']
            if 'signal' not in result or 'teff_K' not in p or record['role']!='planet':continue
            key=(record['family'],json.dumps({k:v for k,v in p.items() if k!='teff_K'},sort_keys=True))
            groups.setdefault(key,[]).append(record)
        for key,group in groups.items():
            group.sort(key=lambda r:r['parameters']['teff_K'])
            for i in range(1,len(group)-1):
                a,b,c=group[i-1:i+2]
                t=[a['parameters']['teff_K'],c['parameters']['teff_K']]
                if t[0]>=t[1] or not all(r['bands'][band['name']]['signal']>0 for r in (a,b,c)):continue
                predicted=float(interpolate_band_signal(t,[a['bands'][band['name']]['signal'],c['bands'][band['name']]['signal']],[b['parameters']['teff_K']])[0])
                truth=b['bands'][band['name']]['signal']
                heldout.append({'family':key[0],'held_parameters':json.loads(key[1]),'band':band['name'],'omitted_temperature_K':b['parameters']['teff_K'],
                    'native_signal':truth,'interpolated_signal':predicted,'fractional_error':predicted/truth-1})
        gravity_groups={}
        mesh_groups={}
        for record in records:
            p=record['parameters'];signal=record['bands'].get(band['name'],{}).get('signal')
            if record['role']!='planet' or 'logg_cgs' not in p or signal is None or signal<=0:continue
            if record.get('grid_coordinates'):
                key=(record['family'],json.dumps({k:v for k,v in p.items() if k not in ('teff_K','logg_cgs')},sort_keys=True))
                mesh_groups.setdefault(key,[]).append(record)
                continue
            key=(record['family'],json.dumps({k:v for k,v in p.items() if k!='logg_cgs'},sort_keys=True))
            gravity_groups.setdefault(key,[]).append(record)
        for key,group in gravity_groups.items():
            group.sort(key=lambda r:r['parameters']['logg_cgs'])
            for i in range(1,len(group)-1):
                a,b,c=group[i-1:i+2];ga,gb,gc=[r['parameters']['logg_cgs'] for r in (a,b,c)]
                if not ga<gb<gc:continue
                fraction=(gb-ga)/(gc-ga)
                pred=float(np.exp((1-fraction)*np.log(a['bands'][band['name']]['signal'])+fraction*np.log(c['bands'][band['name']]['signal'])))
                truth=b['bands'][band['name']]['signal']
                gravity_tests.append({'family':key[0],'held_parameters':json.loads(key[1]),'band':band['name'],'omitted_logg_cgs':gb,
                    'native_signal':truth,'interpolated_signal':pred,'fractional_error':pred/truth-1})
        for key,group in mesh_groups.items():
            gravities=sorted({r['parameters']['logg_cgs'] for r in group})
            for omitted in gravities[1:-1]:
                training=[r for r in group if r['parameters']['logg_cgs']!=omitted]
                grid=BandGrid(training,[r['bands'][band['name']]['signal'] for r in training])
                for record in (r for r in group if r['parameters']['logg_cgs']==omitted):
                    query=record['parameters']
                    try:pred,corners=grid.evaluate(query)
                    except ValueError as exc:
                        unsupported_gravity_tests.append({'family':key[0],'band':band['name'],'parameters':query,'reason':str(exc)});continue
                    truth=record['bands'][band['name']]['signal']
                    gravity_tests.append({'family':key[0],'held_parameters':json.loads(key[1]),'band':band['name'],
                        'omitted_logg_cgs':omitted,'temperature_K':query['teff_K'],
                        'native_signal':truth,'interpolated_signal':pred,'fractional_error':pred/truth-1,
                        'method':'Entire gravity track withheld; reported-temperature native-cell interpolation.',
                        'interpolation_corners':corners})
    return {'catalog':str(path.resolve()),'records':records,'load_errors':errors,'heldout_temperature_band_tests':heldout,'heldout_gravity_band_tests':gravity_tests,
        'unsupported_gravity_holdouts':unsupported_gravity_tests,
        'scope':'Installed-file unit/parameter and band-coverage checks; coverage-limited bolometric diagnostics; withheld temperature and gravity-track band tests. Not a blanket scientific approval, full spectrum interpolation test, atmosphere solver validation or observational completeness test.'}

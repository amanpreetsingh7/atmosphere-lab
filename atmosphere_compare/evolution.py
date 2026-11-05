"""Explicit-unit mass/age table adapter with bounded log-bilinear interpolation."""
import csv
import hashlib
import io
import numpy as np
from .compare import RADII_M

G=6.67430e-11
MJUP_KG=1.89813e27
BOBCAT_MSUN_KG=1.989e30
BOBCAT_RSUN_M=6.9599e8
COLUMNS=('mass_Mjup','age_Myr','intrinsic_temperature_K','radius_Rjup')


def read_table(content,name):
    reader=csv.DictReader(io.StringIO(content.decode('utf-8-sig')))
    if not set(COLUMNS).issubset(reader.fieldnames or []):
        raise ValueError('Evolution CSV needs '+','.join(COLUMNS)+'.')
    rows=[{k:float(row[k]) for k in COLUMNS} for row in reader]
    if not rows or any(not np.isfinite(v) or v<=0 for row in rows for v in row.values()):
        raise ValueError('Evolution values must be positive and finite.')
    keys=[(row['mass_Mjup'],row['age_Myr']) for row in rows]
    if len(set(keys))!=len(keys):raise ValueError('Evolution table has duplicate mass/age rows.')
    return rows,{'name':name,'sha256':hashlib.sha256(content).hexdigest(),'columns':list(COLUMNS),
                 'source_validation':'User-supplied evolutionary table; provenance/physical applicability must be verified.'}


def bracket(nodes,value):
    nodes=np.unique(nodes)
    if not np.isfinite(value) or value<nodes[0] or value>nodes[-1]:raise ValueError('Mass or age is outside the supplied evolution table; extrapolation is disabled.')
    exact=nodes[np.isclose(nodes,value,rtol=1e-12,atol=0)]
    if len(exact):return [(float(exact[0]),1.0)]
    hi=np.searchsorted(nodes,value);lo=hi-1
    fraction=float((np.log(value)-np.log(nodes[lo]))/(np.log(nodes[hi])-np.log(nodes[lo])))
    return [(float(nodes[lo]),1-fraction),(float(nodes[hi]),fraction)]


def evaluate(rows,mass,age,metadata):
    masses=bracket([r['mass_Mjup'] for r in rows],mass)
    lookup={(r['mass_Mjup'],r['age_Myr']):r for r in rows}
    corners=[]
    output={'mass_Mjup':float(mass),'age_Myr':float(age)}
    for m,wm in masses:
        if metadata.get('track_age_interpolation'):
            ages=bracket([r['age_Myr'] for r in rows if r['mass_Mjup']==m],age)
        else:
            ages=bracket([r['age_Myr'] for r in rows],age)
        for a,wa in ages:
            if (m,a) not in lookup:raise ValueError('The required mass/age interpolation cell is incomplete. Choose an exact row or provide its missing corners.')
            corners.append((lookup[(m,a)],wm*wa))
    for key in ('intrinsic_temperature_K','radius_Rjup'):
        output[key]=float(np.exp(sum(weight*np.log(row[key]) for row,weight in corners)))
    gravity=G*output['mass_Mjup']*MJUP_KG/(output['radius_Rjup']*RADII_M['R_jup'])**2
    output.update(logg_cgs=float(np.log10(gravity*100)),source=metadata,
        interpolation=('Interpolate each bracketing mass track in log(age), then in log(mass); log(T) and log(radius). No extrapolation on either track.' if metadata.get('track_age_interpolation') else 'Exact row or bilinear in log(mass), log(age), log(temperature), log(radius); complete cells only, no extrapolation.'),
        source_corners=[dict(row,weight=weight) for row,weight in corners])
    if any('source_logg_cgs' in r for r,w in corners):
        source_g=sum(w*r['source_logg_cgs'] for r,w in corners)
        output['interpolated_source_logg_cgs']=float(source_g)
        output['computed_minus_source_logg_dex']=float(output['logg_cgs']-source_g)
    return output


def read_bobcat_tracks(content,name,source_url=None):
    """Official *_mass format: counted blocks, Msun/Gyr/Rsun conventions from README."""
    lines=[line.strip() for line in content.decode('utf-8-sig').splitlines() if line.strip()]
    header=lines.pop(0)
    if not all(unit in header for unit in ('M/Msun','age(Gyr)','Teff(K)','R/Rsun')):
        raise ValueError('Expected official Bobcat mass-track header, not a mass_age or photometry table.')
    rows=[];i=0
    while i<len(lines):
        tokens=lines[i].split();i+=1
        if len(tokens)!=1 or not tokens[0].isdigit():raise ValueError('Missing Bobcat track block count.')
        count=int(tokens[0]);block=[]
        if count<1 or i+count>len(lines):raise ValueError('Incomplete Bobcat evolution track block.')
        for line in lines[i:i+count]:
            values=[float(v.replace('D','E')) for v in line.split()]
            if len(values)!=7 or not np.all(np.isfinite(values)):raise ValueError('Expected seven finite columns in Bobcat track.')
            m,a,lum,t,g,r,moment=values
            if min(m,a,t,r)<=0:raise ValueError('Nonpositive mass, age, temperature or radius in track.')
            block.append({'mass_Mjup':m*BOBCAT_MSUN_KG/MJUP_KG,'age_Myr':a*1000,
                'intrinsic_temperature_K':t,'radius_Rjup':r*BOBCAT_RSUN_M/RADII_M['R_jup'],
                'source_logg_cgs':g,'source_logL_Lsun':lum})
        if len({r['mass_Mjup'] for r in block})!=1:raise ValueError('Bobcat mass block mixes masses.')
        rows.extend(block);i+=count
    if not rows or len({(r['mass_Mjup'],r['age_Myr']) for r in rows})!=len(rows):raise ValueError('Empty or duplicate evolution tracks.')
    return rows,{'name':name,'sha256':hashlib.sha256(content).hexdigest(),'native_format':'sonora_bobcat_mass_tracks',
        'source_url':source_url,'track_age_interpolation':True,'columns':list(COLUMNS),
        'source_units':{'mass':'Msun = 1.989e30 kg','age':'Gyr','radius':'Rsun = 6.9599e8 m'},
        'normalized_units':{'mass':'Mjup = 1.89813e27 kg','age':'Myr','radius':'Rjup = 71492000 m'},
        'source_validation':'Published Bobcat 2021 hot-start, non-irradiated cloudless tracks. Native format parsed; scientific applicability and interpolation accuracy require validation.',
        'model_assumptions':'Hot-start; ages below about 10 Myr depend strongly on initial conditions. Evolution composition must match atmosphere composition.'}

"""Approximate heating of exported native band contrasts; no extrapolation.

Pure NumPy functions are also embedded in the one-file figure export. Native
spectra are integrated by calculate_series before this transformation is used.
"""
import copy
import numpy as np


def attach_heating(series, temperatures, *, distance_pc, bond_albedo=.3,
                   true_to_projected=1., separations=(.1, 1.), samples=301):
    """Return a copy with heated curves; preserve the native unheated table.

    Interpolate log band contrast vs log reported native Teff, at fixed star,
    gravity/composition and reference radius. This matches the established
    separation-panel approximation, not the surface-flux interpolation API.
    """
    p=series['provenance'];star=p['star'];native=p['temperatures_and_sources']
    nodes=sorted((float(r['temperature_K']),float(r['contrast_at_unit_radius'])) for r in native)
    t,c=map(np.asarray,zip(*nodes)) if nodes else (np.array([]),np.array([]))
    if (len(t)<2 or np.any(~np.isfinite(t)) or np.any(~np.isfinite(c))
            or np.any(t<=0) or np.any(c<=0) or np.any(np.diff(t)<=0)):
        raise ValueError('Heating needs at least two unique positive native temperature nodes and contrasts.')
    intrinsic=list(map(float,temperatures))
    if not intrinsic or len(intrinsic)>8 or len(set(intrinsic))!=len(intrinsic):
        raise ValueError('Choose one to eight unique Tmin values for the heated plot.')
    if not np.isfinite(intrinsic).all() or min(intrinsic)<t[0] or max(intrinsic)>t[-1]:
        raise ValueError('Tmin must lie inside the available native temperature range.')
    lo,hi=map(float,separations)
    if not np.isfinite([lo,hi]).all() or not 0<lo<hi:
        raise ValueError('Minimum separation must be positive and below maximum.')
    if isinstance(samples,bool) or not isinstance(samples,(int,np.integer)) or not 2<=samples<=2001:
        raise ValueError('Choose 2-2001 separation samples.')
    distance_pc,bond_albedo,true_to_projected=map(float,(distance_pc,bond_albedo,true_to_projected))
    if not np.isfinite([distance_pc,bond_albedo,true_to_projected]).all() or distance_pc<=0 or not 0<=bond_albedo<=1 or true_to_projected<1:
        raise ValueError('Distance must be positive, Bond albedo in [0,1], and true/projected separation ratio at least one.')
    units={'R_sun':6.957e8,'R_jup':7.1492e7,'R_earth':6.3781e6,'m':1.}
    radius=star['radius'];rs=float(radius['value'])*units[radius['unit']]
    ts=float(star['parameters']['teff_K'])
    if not np.isfinite([rs,ts]).all() or min(rs,ts)<=0:raise ValueError('Stellar temperature and radius must be positive and finite.')
    radii=sorted({float(row['radius_value']) for row in series['rows']})
    if not radii or len(radii)>4 or not np.isfinite(radii).all() or min(radii)<=0:
        raise ValueError('Choose one to four positive companion radii.')
    if any(row['radius_unit']!=p['radius_unit'] for row in series['rows']):
        raise ValueError('The native series mixes radius units.')
    # Check the table against its underlying native unit-radius contrasts.
    lookup=dict(nodes)
    if len(series['rows'])!=len(nodes)*len(radii):raise ValueError('Incomplete native series table.')
    seen=set()
    for row in series['rows']:
        key=(float(row['temperature_K']),float(row['radius_value']))
        if key in seen or key[0] not in lookup or not np.isclose(float(row['contrast']),lookup[key[0]]*key[1]**2,rtol=1e-12,atol=0):
            raise ValueError('Native series table differs from its source-node contrasts.')
        seen.add(key)
    theta=np.linspace(lo,hi,int(samples));projected=theta*distance_pc;true_au=projected*true_to_projected
    if np.any(true_au*149597870700.<=rs):raise ValueError('An assumed separation lies inside the star.')
    teq=ts*np.sqrt(rs/(2*true_au*149597870700.))*(1-bond_albedo)**.25
    curves=[]
    for ti in sorted(intrinsic):
        effective=np.sqrt(np.hypot(ti**2,teq**2))
        supported=(effective>=t[0])&(effective<=t[-1]);unit=np.full(theta.shape,np.nan)
        unit[supported]=np.exp(np.interp(np.log(effective[supported]),np.log(t),np.log(c)))
        unheated=float(np.exp(np.interp(np.log(ti),np.log(t),np.log(c))))
        for r in radii:
            contrast=unit*r**2
            curves.append({'Tmin_K':ti,'radius_value':r,'radius_unit':p['radius_unit'],
                'effective_temperature_K':effective.tolist(),
                'contrast':[float(x) if np.isfinite(x) else None for x in contrast],
                'unheated_contrast':unheated*r**2,'omitted_out_of_grid_samples':int(np.sum(~supported))})
    if not any(any(x is not None for x in curve['contrast']) for curve in curves):
        raise ValueError('All heated temperatures are outside the native grid. Choose lower Tmin values or larger separations.')
    result=copy.deepcopy(series)
    result['heated_separation']={'separation_arcsec':theta.tolist(),'projected_separation_au':projected.tolist(),
        'assumed_true_separation_au':true_au.tolist(),'equilibrium_temperature_K':teq.tolist(),'curves':curves,
        'settings':{'distance_pc':distance_pc,'bond_albedo':bond_albedo,'true_to_projected_ratio':true_to_projected,
            'stellar_teff_K':ts,'stellar_radius_m':rs,'temperatures_min_K':sorted(intrinsic),
            'redistribution':'Uniform redistribution over the full planet surface',
            'interpolation':'Linear log band contrast vs log reported native temperature at fixed star, gravity, composition and reference radius; no extrapolation.',
            'temperature_identity':'Teff^4 = Tmin^4 + Teq^4',
            'equilibrium_identity':'Teq = Tstar sqrt(Rstar/(2a)) (1-A_B)^(1/4)',
            'geometry':'a = angular separation times distance times true/projected ratio; instantaneous true distance, not a measured orbit.',
            'scope':'Approximate energy-budget heating using isolated atmospheres. Not a self-consistent irradiated spectrum, reflected-light model, mass evolution track, measured sensitivity or detection probability. Interpolation accuracy is grid/band dependent.'}}
    return result


def heated_csv_rows(series):
    h=series['heated_separation']
    for curve in h['curves']:
        for i,theta in enumerate(h['separation_arcsec']):
            yield {'separation_arcsec':theta,'projected_separation_au':h['projected_separation_au'][i],
                'assumed_true_separation_au':h['assumed_true_separation_au'][i],
                'Tmin_K':curve['Tmin_K'],'radius_value':curve['radius_value'],'radius_unit':curve['radius_unit'],
                'equilibrium_temperature_K':h['equilibrium_temperature_K'][i],
                'effective_temperature_K':curve['effective_temperature_K'][i],
                'contrast':curve['contrast'][i],'unheated_contrast':curve['unheated_contrast']}


def draw_heated_series(series,folder):
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    h=series['heated_separation'];settings=h['settings'];p=series['provenance'];star=p['star']
    temps=sorted({c['Tmin_K'] for c in h['curves']});radii=sorted({c['radius_value'] for c in h['curves']})
    colors=plt.get_cmap('tab10').colors;styles=[':', '-', '--', '-.']
    unit={'R_jup':r'$R_{\rm J}$','R_earth':r'$R_\oplus$','R_sun':r'$R_\odot$','m':'m'}[p['radius_unit']]
    with plt.rc_context({'font.size':10,'pdf.fonttype':42}):
        fig,ax=plt.subplots(figsize=(12,8.4));fig.subplots_adjust(left=.105,right=.775,bottom=.28,top=.75)
        for curve in h['curves']:
            i,j=temps.index(curve['Tmin_K']),radii.index(curve['radius_value'])
            ax.semilogy(h['separation_arcsec'],[np.nan if x is None else x for x in curve['contrast']],color=colors[i],ls=styles[j],lw=1.8)
        ax.set(xlim=(h['separation_arcsec'][0],h['separation_arcsec'][-1]),xlabel='Angular separation [arcsec]',ylabel='Band-integrated companion / star contrast')
        ax.grid(which='major',alpha=.25);ax.grid(which='minor',axis='y',alpha=.12)
        distance=settings['distance_pc']
        sec=ax.secondary_xaxis('top',functions=(lambda x:x*distance,lambda x:x/distance));sec.set_xlabel('Projected separation [au]',labelpad=8)
        handles=[Line2D([],[],color=colors[i],lw=2,label='%g K'%t) for i,t in enumerate(temps)]
        first=ax.legend(handles=handles,title=r'Intrinsic temperature $T_{\min}$',loc='upper left',bbox_to_anchor=(1.015,1.02),fontsize=9,title_fontsize=9,frameon=False);ax.add_artist(first)
        handles=[Line2D([],[],color='.25',lw=1.8,ls=styles[i],label='%g %s'%(r,unit)) for i,r in enumerate(radii)]
        ax.legend(handles=handles,title='Assumed companion radius',loc='upper left',bbox_to_anchor=(1.015,.32),fontsize=9,title_fontsize=9,frameon=False)
        fig.suptitle('Theoretical contrast with approximate stellar heating',fontsize=17,y=.97)
        star_text='%s: %g K, radius %g %s'%(star['family'].replace('_',' '),star['parameters']['teff_K'],star['radius']['value'],star['radius']['unit'])
        fig.text(.105,.90,star_text,fontsize=10)
        held=', '.join('%s=%s'%(k,v) for k,v in sorted(p['held_planet_parameters'].items()))
        import textwrap
        for i,line in enumerate(textwrap.wrap(p['planet_family'].replace('_',' ')+': '+held,width=115)):
            fig.text(.105,.87-i*.025,line,fontsize=9)
        fig.text(.105,.815,'Distance %g pc; Bond albedo %g; true/projected separation %g'%(distance,settings['bond_albedo'],settings['true_to_projected_ratio']),fontsize=9)
        lines=[r'$T_{\min}$: temperature without stellar heating; $T_{\rm eff}^{4}=T_{\min}^{4}+T_{\rm eq}^{4}$.',
            r'Full redistribution: $T_{\rm eq}=T_\star\sqrt{R_\star/(2a)}(1-A_{\rm B})^{1/4}$.',
            '%s / photon response (%g-%g um)'%(p['band'].get('name','Declared response'),*p['band']['limits_um']),
            'Isolated-model band contrasts interpolated in log T and log contrast; fixed gravity/composition; no extrapolation.',
            'Not a self-consistent irradiated atmosphere. No reflected light, measured sensitivity or detection probability.',
            'Unsupported heated temperatures are omitted. Native-node accuracy and interpolation errors remain model dependent.']
        if series.get('target_example'):
            lines.append('AB combined-light effective radius; cloud-free b proxy, not an atmosphere fit. Both radii use the same native T and gravity.')
        for i,line in enumerate(lines):fig.text(.105,.20-i*.027,line,fontsize=8.5)
        for ext in ('png','pdf'):fig.savefig(folder/('03_contrast_vs_separation_heated.'+ext),dpi=200)
        plt.close(fig)

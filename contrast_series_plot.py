"""Shared static figure style, also embedded in the portable figure script."""
def draw_series(series,folder):
    from pathlib import Path
    import math
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rows=series['rows'];p=series['provenance']
    values={(float(r['temperature_K']),float(r['radius_value'])):float(r['contrast']) for r in rows}
    ts=sorted({t for t,r in values});radii=sorted({r for t,r in values})
    if len(values)!=len(rows) or len(values)!=len(ts)*len(radii):raise ValueError('Incomplete or duplicate temperature/radius table.')
    if not all(math.isfinite(c) and c>0 for c in values.values()):raise ValueError('Invalid contrast value.')
    for t in ts:
        if any(not math.isclose(values[t,r]/r**2,values[t,radii[0]]/radii[0]**2,rel_tol=1e-10) for r in radii):raise ValueError('Radius scaling differs from the declared fixed surface spectrum.')
    lo,hi=series.get('separation_display_arcsec',[.1,1.])
    if not np.isfinite([lo,hi]).all() or not 0<lo<hi:raise ValueError('Invalid separation display interval.')
    unit={'R_jup':'R$_J$','R_earth':'R$_{\\oplus}$','R_sun':'R$_{\\odot}$','m':'m'}[p['radius_unit']]
    star=p['star'];sp=star['parameters'];held=p['held_planet_parameters']
    star_text='%s: %g K, radius %g %s'%(star['family'].replace('_',' '),sp['teff_K'],star['radius']['value'],star['radius']['unit'])
    planet_text=p['planet_family'].replace('_',' ')
    if 'logg_cgs' in held:planet_text+=': log g = %g'%held['logg_cgs']
    band=p['band'];name=band.get('name','Declared observing response');limits=band['limits_um']
    footer='%s / photon response (%g-%g um)'%(name,*limits)
    styles=[':', '-', '--', '-.']
    with plt.rc_context({'font.size':10,'axes.titlesize':11,'axes.spines.top':True,'axes.spines.right':True,'pdf.fonttype':42}):
        fig,ax=plt.subplots(figsize=(10,7))
        for i,r in enumerate(radii):ax.semilogy(ts,[values[t,r] for t in ts],lw=1.5,marker='o',ms=3,ls=styles[i%4],label='%g %s'%(r,unit))
        ax.set(xlabel='Companion effective temperature [K]',ylabel='Band-integrated companion / star contrast')
        ax.set_title('Theoretical contrast - native atmosphere temperature sequence',pad=45)
        ax.text(.5,1.025,star_text+'; '+planet_text,transform=ax.transAxes,ha='center',fontsize=8)
        ax.legend(title='Assumed companion radius',fontsize=8,title_fontsize=8,loc='lower right');ax.grid(alpha=.2,which='both')
        fig.text(.1,.045,footer,fontsize=8)
        fig.text(.1,.018,'Markers: exact companion nodes; lines: visual guides. Composition held fixed; no irradiation or measured sensitivity.',fontsize=8)
        fig.tight_layout(rect=(0,.075,1,.98))
        for ext in ('png','pdf'):fig.savefig(folder/('01_contrast_vs_temperature.'+ext),dpi=180)
        plt.close(fig)
        chosen=[ts[i] for i in np.unique(np.linspace(0,len(ts)-1,min(5,len(ts))).round().astype(int))]
        reference=min(radii,key=lambda r:abs(r-1.))
        fig,ax=plt.subplots(figsize=(10,7))
        for t in chosen:
            for i,r in enumerate(radii):ax.semilogy((lo,hi),(values[t,r],)*2,color='dodgerblue',lw=1.5,ls=styles[i%4])
            ax.text(1.01,values[t,reference],'%g K'%t,transform=ax.get_yaxis_transform(),color='dodgerblue',fontsize=8,va='center')
        ax.set(xlim=(lo,hi),xlabel='Angular separation [arcsec]',ylabel='Band-integrated companion / star contrast')
        ax.set_title('Theoretical contrast - fixed atmosphere predictions',pad=45)
        ax.text(.5,1.025,star_text+'; '+planet_text,transform=ax.transAxes,ha='center',fontsize=8)
        handles=[Line2D([],[],color='dodgerblue',ls=styles[i%4],label='%g %s'%(r,unit)) for i,r in enumerate(radii)]
        ax.legend(handles=handles,title='Assumed companion radius',fontsize=8,title_fontsize=8,loc='lower right',bbox_to_anchor=(.99,.18));ax.grid(alpha=.2,which='both')
        fig.text(.1,.045,footer,fontsize=8)
        fig.text(.1,.018,'Temperature labels refer to %g %s curves. Fixed intrinsic atmospheres; no irradiation or measured sensitivity.'%(reference,p['radius_unit']),fontsize=8)
        fig.tight_layout(rect=(0,.075,.97,.98))
        for ext in ('png','pdf'):fig.savefig(folder/('02_contrast_vs_separation.'+ext),dpi=180)
        plt.close(fig)
    if 'heated_separation' in series:
        draw_heated_series(series,folder)


from atmosphere_compare.heated_series import draw_heated_series

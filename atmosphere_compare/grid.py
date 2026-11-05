"""Bounded band interpolation on complete atmosphere cells, with provenance."""
from itertools import product
from pathlib import Path
import hashlib
import json
import numpy as np
from .spectra import load_spectrum
from .compare import band_signal


def nodes_and_weights(nodes, value, logarithmic=False):
    nodes=np.unique(np.asarray(nodes,float)); value=float(value)
    if not len(nodes) or not np.isfinite(value):
        raise ValueError('Query is outside the installed atmosphere grid. Extrapolation is disabled.')
    exact=np.flatnonzero(np.isclose(nodes,value,rtol=1e-10,atol=1e-10))
    if len(exact):return [(float(nodes[exact[0]]),1.0)]
    if value<nodes[0] or value>nodes[-1]:raise ValueError('Query is outside the installed atmosphere grid. Extrapolation is disabled.')
    hi=np.searchsorted(nodes,value);lo=hi-1
    transform=np.log if logarithmic else lambda x:x
    if logarithmic and nodes[0]<=0:raise ValueError('Logarithmic coordinates must be positive.')
    fraction=float((transform(value)-transform(nodes[lo]))/(transform(nodes[hi])-transform(nodes[lo])))
    return [(float(nodes[lo]),1-fraction),(float(nodes[hi]),fraction)]


class BandGrid:
    def __init__(self, rows, signals, axes=('teff_K','logg_cgs'), metadata=None):
        self.axes=tuple(axes);self.rows=rows;self.signals=np.asarray(signals,float)
        if not rows or self.signals.shape!=(len(rows),) or np.any(~np.isfinite(self.signals)) or np.any(self.signals<=0):
            raise ValueError('A band grid needs positive finite signals and matching atmosphere rows.')
        self.lookup={tuple(float(row['parameters'][axis]) for axis in self.axes):(i,float(signal)) for i,(row,signal) in enumerate(zip(rows,self.signals))}
        if len(self.lookup)!=len(rows):raise ValueError('Duplicate atmosphere cells: keep additional composition/cloud parameters fixed.')
        self.nodes={axis:np.unique([key[j] for key in self.lookup]) for j,axis in enumerate(self.axes)}
        self.metadata=metadata or {}
        self.mesh_lookup=None
        tagged=[row.get('grid_coordinates') for row in rows]
        if any(tagged):
            if self.axes!=('teff_K','logg_cgs') or not all(tagged):
                raise ValueError('Reported-coordinate Bobcat mesh requires temperature/gravity axes and verified connectivity for every row. Reinstall or separate incompatible exports.')
            self.mesh_lookup={}
            for i,(row,coordinates) in enumerate(zip(rows,tagged)):
                nominal=float(coordinates['nominal_teff_K']);reported=float(row['parameters']['teff_K'])
                gravity=float(row['parameters']['logg_cgs'])
                if not np.isfinite(nominal) or nominal<=0 or abs(nominal-reported)>1.+1e-9:
                    raise ValueError('Invalid Bobcat nominal/reported temperature coordinates.')
                key=(nominal,gravity)
                if key in self.mesh_lookup:raise ValueError('Duplicate nominal Bobcat temperature/gravity cell.')
                self.mesh_lookup[key]=(i,float(self.signals[i]))
            self.nominal_nodes=np.unique([key[0] for key in self.mesh_lookup])
            for gravity in self.nodes['logg_cgs']:
                ordered=[rows[self.mesh_lookup[(nominal,gravity)][0]]['parameters']['teff_K']
                         for nominal in self.nominal_nodes if (nominal,gravity) in self.mesh_lookup]
                if np.any(np.diff(ordered)<=0):raise ValueError('Reported Bobcat mesh temperatures must increase without folded cells.')
            self.metadata['interpolation']='Log(band signal) on complete native Bobcat quadrilateral cells in reported log(temperature), logg. Nominal filename temperature identifies adjacency only; physical coordinates are not snapped. Nonnegative isoparametric weights; no missing-cell bridging or extrapolation.'

    def _corner(self,index,signal,weight):
        result={'parameters':self.rows[index]['parameters'],'weight':float(weight),'surface_band_signal':signal}
        if self.rows[index].get('grid_coordinates'):
            result['grid_coordinates']=self.rows[index]['grid_coordinates']
        return result

    def _projected_mesh_temperatures(self,gravity_weights):
        out=[]
        for nominal in self.nominal_nodes:
            if any((nominal,g) not in self.mesh_lookup for g,w in gravity_weights):out.append(np.nan);continue
            if len(gravity_weights)==1:
                out.append(float(self.rows[self.mesh_lookup[(nominal,gravity_weights[0][0])][0]]['parameters']['teff_K']))
                continue
            out.append(float(np.exp(sum(w*np.log(self.rows[self.mesh_lookup[(nominal,g)][0]]['parameters']['teff_K'])
                                       for g,w in gravity_weights))))
        return np.asarray(out)

    def temperatures_at_gravity(self,gravity):
        """Native temperatures at an installed gravity; mapped nodes otherwise."""
        weights=nodes_and_weights(self.nodes['logg_cgs'],gravity)
        if self.mesh_lookup is not None:
            values=self._projected_mesh_temperatures(weights)
            return values[np.isfinite(values)]
        if len(weights)==1:
            g=weights[0][0]
            return np.unique([r['parameters']['teff_K'] for r in self.rows if r['parameters']['logg_cgs']==g])
        return self.nodes['teff_K']

    def _evaluate_mesh(self,query):
        temperature=float(query['teff_K'])
        if not np.isfinite(temperature) or temperature<=0:raise ValueError('Query is outside the installed atmosphere grid.')
        gravity_weights=nodes_and_weights(self.nodes['logg_cgs'],query['logg_cgs'])
        mapped=self._projected_mesh_temperatures(gravity_weights)
        exact=np.flatnonzero(np.isclose(mapped,temperature,rtol=1e-10,atol=1e-10))
        if len(exact):temperature_weights=[(self.nominal_nodes[exact[0]],1.)]
        else:
            indices=[i for i in range(len(mapped)-1) if np.isfinite(mapped[i:i+2]).all() and mapped[i]<temperature<mapped[i+1]]
            if not indices:
                finite=mapped[np.isfinite(mapped)]
                if len(finite) and (temperature<finite[0] or temperature>finite[-1]):
                    raise ValueError('Query is outside the reported atmosphere mesh; extrapolation is disabled.')
                raise ValueError('Required atmosphere interpolation cell is incomplete; missing native nodes cannot be bridged.')
            i=indices[0]
            fraction=float(np.log(temperature/mapped[i])/np.log(mapped[i+1]/mapped[i]))
            temperature_weights=[(self.nominal_nodes[i],1-fraction),(self.nominal_nodes[i+1],fraction)]
        corners=[];value=0.
        for nominal,wt in temperature_weights:
            for gravity,wg in gravity_weights:
                index,signal=self.mesh_lookup[(nominal,gravity)];weight=wt*wg
                value+=weight*np.log(signal);corners.append(self._corner(index,signal,weight))
        return float(np.exp(value)),corners

    def evaluate(self, query):
        if self.mesh_lookup is not None:return self._evaluate_mesh(query)
        brackets=[nodes_and_weights(self.nodes[a],query[a],a in ('teff_K','tint_K','teq_K')) for a in self.axes]
        corners=[];value=0.
        for combination in product(*brackets):
            key=tuple(v for v,w in combination);weight=float(np.prod([w for v,w in combination]))
            if key not in self.lookup:raise ValueError('Required atmosphere interpolation cell is incomplete. Install its missing temperature/gravity nodes.')
            i,signal=self.lookup[key];value+=weight*np.log(signal)
            corners.append(self._corner(i,signal,weight))
        return float(np.exp(value)),corners

    def evaluate_many(self, temperatures, gravity):
        # Unsupported points are explicit gaps; errors are recorded individually.
        out=[];errors={}
        for i,temp in enumerate(temperatures):
            try:out.append(self.evaluate({'teff_K':temp,'logg_cgs':gravity})[0])
            except ValueError as exc:out.append(np.nan);errors[str(i)]=str(exc)
        return np.asarray(out),errors


def catalog_band_grid(metadata,wavelength,throughput,detector,axes=('teff_K','logg_cgs')):
    path=Path(metadata.get('catalog_path',''))
    if not path.is_file():raise ValueError('Grid interpolation requires a local catalog selection.')
    selected=metadata.get('parameters_declared',{})
    held={k:v for k,v in selected.items() if k not in axes}
    rows=[r for r in json.loads(path.read_text()) if r.get('role','planet')=='planet' and r['family']==metadata['family']
          and {k:v for k,v in r['parameters'].items() if k not in axes}==held
          and all(a in r['parameters'] for a in axes) and (path.parent/r['spectrum']['file']).is_file()]
    signals=[];sources=[];verified_rows=[]
    for row in rows:
        spec=load_spectrum(dict(row['spectrum'],parameters=row['parameters']),path.parent)
        if spec.metadata['wavelength_frame']!=metadata.get('wavelength_frame','vacuum'):raise ValueError('Atmosphere grid mixes wavelength conventions.')
        verified=dict(row)
        if spec.metadata.get('grid_coordinates'):
            coordinates=spec.metadata['grid_coordinates']
            if row.get('grid_coordinates') and row['grid_coordinates'].get('nominal_teff_K')!=coordinates['nominal_teff_K']:
                raise ValueError('Catalog connectivity disagrees with the native Bobcat filename.')
            verified['grid_coordinates']=coordinates
        elif row.get('grid_coordinates'):
            raise ValueError('Catalog mesh connectivity is not verified by this native reader.')
        verified_rows.append(verified)
        signals.append(band_signal(spec,wavelength,throughput,detector));sources.append(spec.metadata)
    return BandGrid(verified_rows,signals,axes,{'family':metadata['family'],'held_parameters':held,'spectra':sources,
        'catalog_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'interpolation':'Multilinear log(band signal), log(temperature) and linear logg; complete bracketing cells only. No extrapolation. Accuracy must be checked with held-out models.'})

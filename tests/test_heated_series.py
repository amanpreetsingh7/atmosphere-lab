"""Analytic and app/export controls; fixture spectra are manufactured."""
from pathlib import Path
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
import numpy as np
from streamlit.testing.v1 import AppTest
from atmosphere_compare.heated_series import attach_heating,heated_csv_rows
from atmosphere_compare.separation import heating
from atmosphere_compare.contrast_series import calculate_series,calculate_heated_series,export_series
from test_properties import ROOT,write_library,bodies,response_mapping


def analytic_series():
    temps=[200.,400.,800.,1600.,2400.];radii=[.8,1.,1.2]
    return {'rows':[{'temperature_K':t,'radius_value':r,'radius_unit':'R_jup','contrast':t**2*1e-10*r**2} for t in temps for r in radii],
        'provenance':{'star':{'family':'manufactured_star','parameters':{'teff_K':5000.,'logg_cgs':4.5},'radius':{'value':1.,'unit':'R_sun'}},
            'planet_family':'manufactured_planet','held_planet_parameters':{'logg_cgs':4.,'cloud':'clear'},'radius_unit':'R_jup',
            'band':{'name':'Manufactured unity band','limits_um':[9,14]},'scope':'Manufactured native-node fixture',
            'temperatures_and_sources':[{'temperature_K':t,'contrast_at_unit_radius':t**2*1e-10} for t in temps]}}


class HeatedSeriesChecks(unittest.TestCase):
    def heat(self,series=None,temperatures=None,**options):
        return attach_heating(series or analytic_series(),temperatures or [200,400,800],distance_pc=options.pop('distance_pc',10),**options)

    def test_energy_identity_existing_heating_agreement_and_analytic_interpolation(self):
        result=self.heat();h=result['heated_separation'];x=np.array(h['separation_arcsec'])
        _,a,eq,eff=heating(x,10,6.957e8,5000,200,.3,1)
        np.testing.assert_allclose(h['assumed_true_separation_au'],a,rtol=1e-14)
        np.testing.assert_allclose(h['equilibrium_temperature_K'],eq,rtol=1e-14)
        for curve in h['curves']:
            te=np.array(curve['effective_temperature_K'])
            np.testing.assert_allclose(te**4,curve['Tmin_K']**4+eq**4,rtol=2e-15)
            np.testing.assert_allclose(curve['contrast'],te**2*1e-10*curve['radius_value']**2,rtol=3e-14)

    def test_no_absorption_native_node_limit_and_input_preservation(self):
        source=analytic_series();before=copy.deepcopy(source);r=self.heat(source,bond_albedo=1)
        self.assertEqual(source,before)
        for curve in r['heated_separation']['curves']:
            np.testing.assert_allclose(curve['contrast'],curve['unheated_contrast'],rtol=1e-14)

    def test_radius_scaling_and_geometry_distance_effect(self):
        one=self.heat()['heated_separation'];far=self.heat(distance_pc=20)['heated_separation'];geometry=self.heat(true_to_projected=2)['heated_separation']
        for a,b in zip(far['curves'],geometry['curves']):np.testing.assert_allclose(a['contrast'],b['contrast'],rtol=1e-14)
        for a,b in zip(one['curves'],far['curves']):self.assertTrue(np.all(np.array(a['contrast'])>=b['contrast']))
        for i in range(0,len(one['curves']),3):
            np.testing.assert_allclose(np.array(one['curves'][i]['contrast'])/one['curves'][i+1]['contrast'],.64,rtol=1e-14)
            np.testing.assert_allclose(np.array(one['curves'][i+2]['contrast'])/one['curves'][i+1]['contrast'],1.44,rtol=1e-14)

    def test_outside_grid_is_null_without_clamping_or_extrapolation(self):
        r=self.heat(temperatures=[200,2400]);curves=r['heated_separation']['curves']
        for curve in curves[-3:]:self.assertTrue(all(x is None for x in curve['contrast']))
        self.assertTrue(all(c['omitted_out_of_grid_samples']==301 for c in curves[-3:]))
        json.dumps(r,allow_nan=False)
        with self.assertRaisesRegex(ValueError,'All heated'):self.heat(temperatures=[2400])

    def test_partial_grid_gaps_and_near_star_rejection(self):
        r=self.heat(temperatures=[2300],separations=(.001,1))
        c=r['heated_separation']['curves'][0]
        self.assertGreater(c['omitted_out_of_grid_samples'],0);self.assertLess(c['omitted_out_of_grid_samples'],301)
        with self.assertRaisesRegex(ValueError,'inside the star'):self.heat(separations=(.00001,.001))

    def test_invalid_physical_inputs_and_inconsistent_source_table(self):
        for options in ({'distance_pc':0},{'bond_albedo':-1},{'bond_albedo':float('nan')},{'true_to_projected':.5},{'separations':(1,.1)},{'samples':True},{'samples':1}):
            with self.subTest(options=options),self.assertRaises(ValueError):self.heat(**options)
        for ts in ([200,200],[100],[2500],[float('nan')],list(range(200,209))):
            with self.subTest(ts=ts),self.assertRaises(ValueError):self.heat(temperatures=ts)
        for mutation in ('duplicate','flux','missing','unit'):
            s=analytic_series()
            if mutation=='duplicate':s['provenance']['temperatures_and_sources'].append(s['provenance']['temperatures_and_sources'][0])
            if mutation=='flux':s['rows'][0]['contrast']*=2
            if mutation=='missing':s['rows'].pop()
            if mutation=='unit':s['rows'][0]['radius_unit']='m'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.heat(s)

    def test_general_star_parameters_and_radius_unit_conversion(self):
        base=analytic_series();other=copy.deepcopy(base);other['provenance']['star']['parameters']['teff_K']=6000
        first=self.heat(base)['heated_separation'];second=self.heat(other)['heated_separation']
        np.testing.assert_allclose(np.array(second['equilibrium_temperature_K'])/first['equilibrium_temperature_K'],1.2)
        # The actual app recomputes native contrasts when switching stars; this
        # pure transform uses exactly the star bound to each source export.
        other=copy.deepcopy(base);other['provenance']['star']['radius']={'value':6.957e8,'unit':'m'}
        np.testing.assert_allclose(self.heat(other)['heated_separation']['equilibrium_temperature_K'],first['equilibrium_temperature_K'])

    def test_fine_sampling_matches_common_points_and_csv_contains_temperatures(self):
        a=self.heat();b=self.heat(samples=601)
        for c,d in zip(a['heated_separation']['curves'],b['heated_separation']['curves']):np.testing.assert_allclose(c['contrast'],d['contrast'][::2],rtol=1e-14)
        row=next(heated_csv_rows(a));self.assertIn('Tmin_K',row);self.assertIn('effective_temperature_K',row)

    def test_native_support_uses_unselected_upper_nodes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);catalog,_=write_library(root);star,planet=bodies()
            data=calculate_heated_series(catalog,star,planet,response_mapping(),[1000],[1],radius_unit='m',base=root,distance_pc=10)
            self.assertEqual(len(data['rows']),2)
            self.assertEqual(len(data['heated_separation']['curves']),1)
            self.assertEqual([n['temperature_K'] for n in data['provenance']['temperatures_and_sources']],[1000,2000])

    def test_heated_export_contains_results_and_figures_without_external_code(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            data=self.heat()
            archive=export_series(root,data)
            with zipfile.ZipFile(archive) as bundle:
                names=set(bundle.namelist())
                self.assertIn('heated_contrasts.csv',names)
                self.assertIn('03_contrast_vs_separation_heated.png',names)
                self.assertIn('03_contrast_vs_separation_heated.pdf',names)
                self.assertFalse(any(name.endswith('.py') for name in names))
                restored=json.loads(bundle.read('series.json'))
                self.assertEqual(restored['heated_separation'],data['heated_separation'])


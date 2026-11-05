"""Synthetic fixture and receiver checks; no real native-grid calculation."""
from pathlib import Path
import copy,json,shutil,subprocess,sys,tempfile,unittest,zipfile
import numpy as np
from streamlit.testing.v1 import AppTest
from test_properties import write_library,bodies,response_mapping
from atmosphere_compare.properties import ModelLibrary
from atmosphere_compare.contrast_series import available_nodes,calculate_series,export_series

class SeriesChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.catalog,self.rows=write_library(self.root);self.star,self.planet=bodies()
        self.response=response_mapping()
    def tearDown(self):self.temp.cleanup()
    def calculate(self,**kwargs):
        args=dict(radius_unit='m',base=self.root);args.update(kwargs)
        return calculate_series(self.catalog,self.star,self.planet,self.response,[1000.,2000.],[1.,2.],**args)
    def test_exact_grid_and_radius_formula(self):
        calls=[];r=self.calculate(progress=lambda done,total:calls.append((done,total)))
        np.testing.assert_allclose([v['contrast'] for v in r['rows']],[.001,.004,.002,.008],rtol=1e-12)
        self.assertEqual(calls,[(1,2),(2,2)])
        self.assertEqual(r['provenance']['companion_selection'],'exact')
        self.assertEqual(r['provenance']['held_planet_parameters']['cloud'],'clear')
    def test_invalid_inputs_and_no_gravity_snapping(self):
        with self.assertRaises(ValueError):calculate_series(self.catalog,self.star,self.planet,self.response,[1500.],[1.],base=self.root)
        for radii in ([1.,1.],[0.],[float('nan')],[1,2,3,4,5]):
            with self.assertRaises(ValueError):calculate_series(self.catalog,self.star,self.planet,self.response,[1000.],radii,base=self.root)
        p=copy.deepcopy(self.planet);p['parameters']['logg_cgs']=4.2
        self.assertFalse(available_nodes(ModelLibrary(self.catalog),p))
    def test_ambiguous_native_nodes_fail(self):
        self.catalog.write_text(json.dumps(self.rows+[self.rows[2]]))
        with self.assertRaises(ValueError):available_nodes(ModelLibrary(self.catalog),self.planet)
    def test_model_mutation_during_calculation_rejected(self):
        def mutate(done,total):
            if done==1:
                path=self.root/'models'/Path(self.rows[-1]['spectrum']['file']).name
                path.write_text(path.read_text()+'\n')
        with self.assertRaises(ValueError):self.calculate(progress=mutate)
    def test_gui_manual_export_and_stale_inputs(self):
        source='from pathlib import Path\nfrom general_inputs_ui import render\nrender(Path(%r))'%str(self.root)
        app=AppTest.from_string(source,default_timeout=60).run()
        app.selectbox(key='general_planet_family').set_value('planet_grid').run()
        app.button(key='general_predict').click().run()
        self.assertEqual(len(app.exception),0)
        self.assertNotIn('property_series_result',app.session_state)
        app.checkbox(key='property_series_all').check().run()
        app.button(key='property_series_calculate').click().run()
        self.assertEqual(len(app.exception),0);self.assertFalse(app.error)
        self.assertEqual(len(app.session_state['property_series_result']['data']['rows']),6)
        app.text_input(key='property_series_radii_R_jup').set_value('1,2').run()
        self.assertNotIn('property_series_result',app.session_state)


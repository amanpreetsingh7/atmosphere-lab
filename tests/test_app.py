"""Widget checks using explicitly analytic file fixtures in atmosphere mode."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from streamlit.testing.v1 import AppTest
from app_fixtures import ROOT, write_fixture_catalog, load_app


class InterfaceChecks(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.path=Path(self.directory.name)
        self.catalog=write_fixture_catalog(self.path)

    def tearDown(self):
        self.directory.cleanup()

    def start(self):
        at=load_app(self.catalog)
        self.assertEqual(len(at.exception),0)
        self.assertIn('current',at.session_state)
        return at

    def test_radius_run_and_saved_snapshot(self):
        at=self.start()
        original=at.session_state['current']['result']['planet_star_contrast'].copy()
        at.button(key='save').click().run()
        at.number_input(key='planet_radius').set_value(2.0).run()
        np.testing.assert_allclose(at.session_state['current']['result']['planet_star_contrast'],original)
        at.button(key='run').click().run()
        self.assertEqual(len(at.exception),0)
        np.testing.assert_allclose(at.session_state['current']['result']['planet_star_contrast'],original*4)
        np.testing.assert_allclose(at.session_state['saved'][0]['result']['planet_star_contrast'],original)

    def test_filter_and_export_refresh(self):
        at=self.start()
        file=self.path/'response.csv'
        np.savetxt(file,[[2,0],[3,1],[4,0]],delimiter=',')
        at.radio(key='filter_source').set_value('Local throughput file').run()
        at.text_input(key='filter_path').set_value(str(file)).run()
        at.checkbox(key='use_limit').check().run()
        at.button(key='calculate_filter').click().run()
        self.assertEqual(len(at.exception),0)
        bands=at.session_state['current']['report']['filter_results']
        self.assertEqual(len(bands),1)
        self.assertEqual(bands[0]['response']['wavelength_um'],[2,3,4])
        at.button(key='prepare_export').click().run()
        self.assertGreater(len(at.session_state['export']['zip']),1000)
        at.button(key='save').click().run()
        self.assertTrue(any('changed' in e.value for e in at.info))

    def test_invalid_range_clears_old_result(self):
        at=self.start()
        at.checkbox(key='custom_range').check().run()
        at.number_input(key='wave_lower').set_value(50.0).run()
        at.button(key='run').click().run()
        self.assertEqual(len(at.exception),0)
        self.assertTrue(at.error)
        self.assertNotIn('current',at.session_state)

    def test_available_parameters_sorted_and_no_presets(self):
        catalog=write_fixture_catalog(self.path,(400,1000,1500))
        at=load_app(catalog)
        self.assertEqual(len(at.exception),0)
        self.assertEqual(at.selectbox(key='planet_param_teff_K').options,['400','1000','1500'])
        self.assertEqual(at.session_state['current']['mode'],'Atmosphere models')
        self.assertFalse(any('preset' in b.label.lower() for b in at.button))
        self.assertFalse(any('blackbody' in r.label.lower() for r in at.radio))
        self.assertFalse(any('Kevin' in e.value for e in at.markdown))

    def test_startup_has_no_synthetic_run_or_missing_model_fallback(self):
        at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        self.assertEqual(len(at.exception),0)
        self.assertNotIn('current',at.session_state)
        at.text_input(key='star_catalog_path').set_value(str(self.path/'missing.json')).run()
        at.button(key='run').click().run()
        self.assertNotIn('current',at.session_state)
        self.assertTrue(at.error)

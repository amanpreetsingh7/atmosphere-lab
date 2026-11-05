"""Synthetic GUI/runner and native-header fixtures for generalized inputs."""
from pathlib import Path
import contextlib
import copy
import io
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
from astropy.io import fits
from streamlit.testing.v1 import AppTest
from test_properties import write_library,bodies,response_mapping
from atmosphere_compare import ck04_install
import run_atmosphere_request as runner


class GeneralizedInputChecks(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.root=Path(self.directory.name)
        self.catalog,self.rows=write_library(self.root)

    def tearDown(self):self.directory.cleanup()

    def test_runner_reads_properties_and_preserves_an_existing_request(self):
        star,companion=bodies()
        request={'catalog':'configs/app_catalog.json','star':star,'companion':companion,
                 'response':response_mapping(),'distance_pc':10.}
        file=self.root/'configs/atmosphere_request.json';file.write_text(json.dumps(request));before=file.read_bytes()
        output=io.StringIO()
        with patch.object(runner,'ROOT',self.root),patch.object(runner.Path,'home',return_value=self.root/'no_desktop'),contextlib.redirect_stdout(output):
            prediction=runner.main(file)
        self.assertAlmostEqual(prediction['planet_star_contrast'],.001)
        self.assertEqual(file.read_bytes(),before)
        self.assertIn('UPLOAD RESULTS:',output.getvalue())
        self.assertEqual(len(list((self.root/'results').glob('*.zip'))),1)

    def test_generic_app_changes_parameters_exports_and_clears_stale_results(self):
        source='from pathlib import Path\nfrom general_inputs_ui import render\nrender(Path(%r))'%str(self.root)
        app=AppTest.from_string(source,default_timeout=45).run()
        self.assertEqual(len(app.exception),0)
        app.button(key='general_predict').click().run()
        self.assertEqual(len(app.exception),0)
        initial=app.session_state['general_prediction']['result']['planet_star_contrast']
        app.button(key='general_export_button').click().run()
        self.assertEqual(len(app.exception),0)
        self.assertGreater(len(app.session_state['general_export']['bytes']),1000)
        app.number_input(key='general_planet_radius').set_value(2.).run()
        self.assertNotIn('general_prediction',app.session_state)
        app.button(key='general_predict').click().run()
        self.assertAlmostEqual(app.session_state['general_prediction']['result']['planet_star_contrast']/initial,4.)
        app.button(key='general_save_request').click().run()
        saved=json.loads((self.root/'configs/atmosphere_request.json').read_text())
        self.assertEqual(saved['companion']['radius']['value'],2.)
        family=saved['companion']['family']
        app.number_input(key='general_planet_'+family+'_teff_K').set_value(1500.).run()
        app.button(key='general_predict').click().run()
        self.assertEqual(len(app.exception),0);self.assertNotIn('general_prediction',app.session_state)
        self.assertTrue(any('No exact' in error.value for error in app.error))

    def test_requested_ck04_node_has_dynamic_identity_and_no_catalog_mutation_on_failure(self):
        file=self.root/'ckm05_6250.fits'
        def write(teff):
            header=fits.Header({'TEFF':teff,'LOG_Z':-.5})
            table=fits.BinTableHDU.from_columns([fits.Column(name='WAVELENGTH',format='1E',unit='ANGSTROM',array=[90000,110000,140000]),
                fits.Column(name='g40',format='1E',unit='FLAM',array=[2,3,4])])
            fits.HDUList([fits.PrimaryHDU(header=header),table]).writeto(file,overwrite=True)
        write(6250)
        with patch.object(ck04_install,'download',return_value=file) as download:
            result=ck04_install.install_ck04(self.root,6250.,4.,-.5,expected_bytes=69120)
        self.assertEqual(result['parameters'],{'teff_K':6250.,'logg_cgs':4.,'metallicity_dex':-.5})
        self.assertIn('/ckm05/ckm05_6250.fits',download.call_args.args[0])
        self.assertEqual(result['spectrum']['gravity_column'],'g40')
        before=self.catalog.read_bytes();write(6000)
        with patch.object(ck04_install,'download',return_value=file):
            with self.assertRaises(ValueError):ck04_install.install_ck04(self.root,6250.,4.,-.5,expected_bytes=69120)
        self.assertEqual(self.catalog.read_bytes(),before)
        with patch.object(ck04_install,'download') as download:
            with self.assertRaises(ValueError):ck04_install.install_ck04(self.root,6250.,3.8,-.5)
            download.assert_not_called()

    def test_interpolated_property_result_shows_warning_and_source_corners(self):
        source='from pathlib import Path\nfrom general_inputs_ui import render\nrender(Path(%r))'%str(self.root)
        app=AppTest.from_string(source,default_timeout=45).run()
        app.selectbox(key='general_planet_family').set_value('planet_grid').run()
        app.selectbox(key='general_planet_mode').set_value('linear_surface_flux').run()
        app.number_input(key='general_planet_planet_grid_teff_K').set_value(1500.).run()
        app.button(key='general_predict').click().run()
        self.assertEqual(len(app.exception),0)
        self.assertTrue(any('Interpolated planet spectrum' in item.value for item in app.warning))
        self.assertTrue(app.session_state['general_prediction']['result']['source_spectra']['planet']['interpolation']['corners'])



if __name__=='__main__':unittest.main()

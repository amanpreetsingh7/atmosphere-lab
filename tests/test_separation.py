"""Analytic / import fixtures only; real model experiments run on the user's laptop."""
from pathlib import Path
import io
import json
import tempfile
import unittest
import zipfile
import numpy as np
from atmosphere_compare.separation import (Sensitivity, sensitivity_from_idl, load_sensitivity,
    heating, interpolate_band_signal, catalog_band_sequence, AU_M)
from atmosphere_compare.compare import RADII_M


class SeparationChecks(unittest.TestCase):
    def test_pa_mean_counts_and_missing_gaps(self):
        s = Sensitivity([.3,.1,.2], [[np.nan,1e-4,2e-4],[np.nan,3e-4,4e-4]], {})
        mean, counts = s.summary()
        np.testing.assert_allclose(mean[:2],[2e-4,3e-4])
        self.assertTrue(np.isnan(mean[2]))
        np.testing.assert_array_equal(counts,[2,2,0])

    def test_partial_idl_and_orientation(self):
        data = {'separations':[.1,.2,.3], 'curves':[[1e-3,2e-4,1e-4],[2e-3,4e-4,2e-4]], 'pas':[30,35,40,45], 'threshold':5.0}
        s = sensitivity_from_idl(data)
        self.assertTrue(s.metadata['partial_pa_run'])
        self.assertEqual(s.metadata['saved_pa_count'],2)
        data['curves'] = np.asarray(data['curves']).T
        np.testing.assert_allclose(sensitivity_from_idl(data).curves,s.curves)
        data['curves'] = np.ones((3,3))
        with self.assertRaisesRegex(ValueError,'ambiguous'):
            sensitivity_from_idl(data)

    def test_csv_long_form_and_rejections(self):
        content=b'separation_arcsec,contrast,pa_deg\n.1,.001,30\n.2,.0001,30\n.1,.003,35\n.2,,35\n'
        s=load_sensitivity(content,'curves.csv')
        np.testing.assert_allclose(s.summary()[0],[.002,.0001])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            load_sensitivity(content+b'.1,.002,30\n','curves.csv')
        with self.assertRaises(ValueError):
            Sensitivity([.1,.2],[-.001,.002],{})
        with self.assertRaises(ValueError):
            load_sensitivity(b'other,columns\n1,2\n','bad.csv')

    def test_heating_energy_and_projected_geometry(self):
        theta=np.array([.1,.2,1.0])
        projected,a,eq,effective=heating(theta,10,RADII_M['R_sun'],5772,100,.3,2)
        np.testing.assert_allclose(projected,[1,2,10])
        np.testing.assert_allclose(a,2*projected)
        np.testing.assert_allclose(effective**4,100**4+eq**4,rtol=1e-14)
        self.assertGreater(eq[0],eq[-1])
        _,_,zero,baseline=heating(theta,10,RADII_M['R_sun'],5772,100,1)
        np.testing.assert_allclose(zero,0)
        np.testing.assert_allclose(baseline,100)
        with self.assertRaises(ValueError):
            heating(theta,10,RADII_M['R_sun'],5772,100,.3,.5)
        # Terrestrial equilibrium temperature under A=0.3 / full redistribution.
        _,_,earth,_=heating(np.array([1,2]),1,RADII_M['R_sun'],5772,1,.3)
        self.assertAlmostEqual(earth[0],254.6,delta=.5)

    def test_interpolation_no_extrapolation(self):
        q=np.array([500,1000,1500,2000,2500,np.nan])
        f=interpolate_band_signal([1000,2000],[1000**4,2000**4],q)
        np.testing.assert_allclose(f[1:4],q[1:4]**4,rtol=1e-14)
        self.assertTrue(np.all(np.isnan(f[[0,4,5]])))
        with self.assertRaises(ValueError):
            interpolate_band_signal([1000,1000],[1,2],q)

    def test_sequence_holds_non_temperature_parameters(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            w=np.linspace(1,6,100)
            rows=[]
            for t,metal in [(1000,0),(1500,0),(2000,1)]:
                file='%s_%s.dat'%(t,metal)
                np.savetxt(path/file,np.column_stack((w,np.ones(len(w))*t**4)))
                rows.append({'role':'planet','family':'fixture','parameters':{'teff_K':t,'metallicity_dex':metal},
                    'spectrum':{'format':'ascii','file':file,'flux_convention':'surface_flux','flux_unit':'W_m2_um','wavelength_unit':'um'}})
            catalog=path/'catalog.json'
            catalog.write_text(json.dumps(rows))
            metadata={'catalog_path':str(catalog),'family':'fixture','parameters_declared':{'teff_K':1000,'metallicity_dex':0}}
            t,f,p=catalog_band_sequence(metadata,[2,3,4],[1,1,1],'energy')
            np.testing.assert_allclose(t,[1000,1500])
            np.testing.assert_allclose(f,2*t**4)
            self.assertEqual(p['held_parameters'],{'metallicity_dex':0})


class SeparationInterfaceChecks(unittest.TestCase):
    def setUp(self):
        from app_fixtures import write_fixture_catalog
        self.directory=tempfile.TemporaryDirectory()
        self.path=Path(self.directory.name)
        self.catalog=write_fixture_catalog(self.path,(400,1000,2400))

    def tearDown(self):
        self.directory.cleanup()

    def start(self):
        from app_fixtures import load_app
        at=load_app(self.catalog)
        at.selectbox(key='sep_band').set_value('Custom top-hat').run()
        self.assertEqual(len(at.exception),0)
        return at

    def test_catalog_sequence_heating_and_native_temperature_curves(self):
        at=self.start()
        at.selectbox(key='sep_route').set_value('Catalog temperature sequence').run()
        at.multiselect(key='sep_temperatures').set_value([400.0,1000.0]).run()
        at.button(key='sep_build').click().run()
        self.assertEqual(len(at.exception),0)
        result=at.session_state['separation_result']
        self.assertEqual(result['model_scope'],'atmosphere temperature sequence')
        self.assertEqual(len(result['models']),2)
        self.assertTrue(all(np.ptp(c['contrast'])==0 for c in result['models']))
        at.checkbox(key='sep_heating').check().run()
        at.button(key='sep_build').click().run()
        self.assertEqual(len(at.exception),0)
        result=at.session_state['separation_result']
        self.assertEqual(result['model_scope'],'approximate heating with atmosphere grids')
        self.assertGreater(result['models'][0]['contrast'][0],result['models'][0]['contrast'][-1])
        self.assertEqual(result['temperature_sequence']['temperature_nodes_K'],[400,1000,2400])

    def test_selected_atmosphere_export_and_changed_source(self):
        at=self.start()
        at.button(key='sep_build').click().run()
        self.assertEqual(len(at.exception),0)
        result=at.session_state['separation_result']
        self.assertEqual(len(result['models']),1)
        self.assertEqual(result['model_scope'],'selected atmosphere prediction')
        self.assertIsNone(result['observation'])
        at.button(key='sep_prepare').click().run()
        self.assertEqual(len(at.exception),0)
        z=zipfile.ZipFile(io.BytesIO(at.session_state['separation_export']['zip']))
        self.assertIn('contrast_vs_separation.pdf',z.namelist())
        json.loads(z.read('separation_report.json'))
        at.number_input(key='star_radius').set_value(2.0).run()
        at.button(key='run').click().run()
        self.assertTrue(any('comparison changed' in x.value for x in at.info))

    def test_calculated_filter_reused_with_exact_response(self):
        at=self.start()
        file=self.path/'response.csv'
        np.savetxt(file,[[2,0],[3,1],[4,0]],delimiter=',')
        at.radio(key='filter_source').set_value('Local throughput file').run()
        at.text_input(key='filter_path').set_value(str(file)).run()
        at.button(key='calculate_filter').click().run()
        expected=at.session_state['current']['report']['filter_results'][-1]
        at.selectbox(key='sep_band').set_value('Calculated filter').run()
        at.button(key='sep_build').click().run()
        self.assertEqual(len(at.exception),0)
        result=at.session_state['separation_result']
        self.assertEqual(result['bandpass']['wavelength_um'],[2,3,4])
        self.assertAlmostEqual(result['models'][0]['contrast'][0],expected['planet_star_contrast'])

    def test_failure_clears_separation_output(self):
        at=self.start()
        at.button(key='sep_build').click().run()
        self.assertIn('separation_result',at.session_state)
        at.number_input(key='sep_upper').set_value(.01).run()
        at.button(key='sep_build').click().run()
        self.assertEqual(len(at.exception),0)
        self.assertNotIn('separation_result',at.session_state)


if __name__ == '__main__':
    unittest.main()

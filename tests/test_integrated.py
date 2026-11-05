"""Integrated workflows with analytic spectra and offline model-job fixtures."""
from pathlib import Path
import io
import json
import tempfile
import time
import unittest
import zipfile
import numpy as np
from atmosphere_compare.evolution import read_table,evaluate
from atmosphere_compare.model_library import register_model,ModelJob
from atmosphere_compare.compare import equivalent_flux_density_Jy,C,RADII_M,PC_M
from atmosphere_compare.spectra import Spectrum
from app_fixtures import write_fixture_catalog,load_app


class IntegratedNumerics(unittest.TestCase):
    def test_evolution_exact_log_cell_and_no_extrapolation(self):
        data='mass_Mjup,age_Myr,intrinsic_temperature_K,radius_Rjup\n1,10,1000,1\n1,100,500,1\n10,10,2000,1\n10,100,1000,1\n'
        rows,meta=read_table(data.encode(),'fixture.csv')
        exact=evaluate(rows,1,10,meta)
        self.assertAlmostEqual(exact['intrinsic_temperature_K'],1000)
        mid=evaluate(rows,np.sqrt(10),np.sqrt(1000),meta)
        self.assertAlmostEqual(mid['intrinsic_temperature_K'],1000)
        self.assertAlmostEqual(sum(r['weight'] for r in mid['source_corners']),1)
        with self.assertRaises(ValueError):evaluate(rows,.5,10,meta)
        with self.assertRaisesRegex(ValueError,'incomplete'):evaluate(rows[:-1],np.sqrt(10),np.sqrt(1000),meta)

    def test_flux_density_flat_fnu_and_radius_distance_scaling(self):
        w=np.geomspace(1,6,80000);fnu=2e-12
        spec=Spectrum(w,fnu*C/(w*1e-6)**2*1e-6)
        expected=fnu*(RADII_M['R_sun']/(10*PC_M))**2/1e-26
        for detector in ('photon','energy'):
            result=equivalent_flux_density_Jy(spec,RADII_M['R_sun'],10,[2,3,4],[0,1,0],detector)
            self.assertAlmostEqual(result/expected,1,delta=1e-6)
            double=equivalent_flux_density_Jy(spec,2*RADII_M['R_sun'],20,[2,3,4],[0,1,0],detector)
            self.assertAlmostEqual(double,result,places=10)

    def test_import_validates_before_catalog_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);file=root/'spectrum.dat'
            np.savetxt(file,np.column_stack((np.linspace(1,6,100),np.ones(100))))
            mapping={'format':'ascii','file':str(file),'wavelength_unit':'um','flux_unit':'W_m2_um','flux_convention':'surface_flux'}
            row,meta=register_model(root,'planet','My exported grid',mapping,{'teff_K':1000,'cloud':'clear'})
            self.assertEqual(row['parameters']['cloud'],'clear')
            with self.assertRaises(ValueError):register_model(root,'planet','My exported grid',mapping,row['parameters'])
            mapping['file']=str(root/'missing.dat')
            before=(root/'configs/app_catalog.json').read_bytes()
            with self.assertRaises(FileNotFoundError):register_model(root,'planet','Another grid',mapping,{'teff_K':500})
            self.assertEqual(before,(root/'configs/app_catalog.json').read_bytes())

    def test_model_job_offline_progress_and_serialization(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'setup_real_models.py').write_text("import time\nprint('offline progress',flush=True)\ntime.sleep(.15)\nprint('offline complete',flush=True)\n")
            job=ModelJob(root);job.start('starter')
            with self.assertRaises(ValueError):job.start('starter')
            second=ModelJob(root)
            with self.assertRaises(ValueError):second.start('starter')
            deadline=time.monotonic()+5
            while job.status()['running'] and time.monotonic()<deadline:time.sleep(.02)
            state=job.status()
            self.assertEqual(state['exit_code'],0)
            self.assertIn('offline complete',state['log'])
            with self.assertRaises(ValueError):job.start('arbitrary_script')


class IntegratedInterface(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.path=Path(self.directory.name)
        self.catalog=write_fixture_catalog(self.path,(400,1000,2400))
    def tearDown(self):self.directory.cleanup()

    def test_full_reference_format_test_inside_app(self):
        at=load_app(self.catalog)
        before=at.session_state['current']['report']['star'].copy()
        at.button(key='sep_test_setup').click().run()
        at.button(key='sep_build').click().run()
        self.assertEqual(len(at.exception),0)
        result=at.session_state['separation_result']
        self.assertEqual(result['observation']['metadata']['kind'],'simulated_layout_test')
        self.assertFalse(result['observation']['metadata']['is_observation'])
        self.assertEqual(len(result['models']),4)
        self.assertEqual(before,at.session_state['current']['report']['star'])
        self.assertGreater(result['settings']['star_band_flux_Jy'],0)
        at.button(key='sep_check').click().run()
        self.assertEqual(len(at.exception),0)
        self.assertTrue(all(row['Heating identity passed'] for row in at.session_state['separation_checks']['rows']))
        at.button(key='sep_prepare').click().run()
        z=zipfile.ZipFile(io.BytesIO(at.session_state['separation_export']['zip']))
        report=json.loads(z.read('separation_report.json'))
        self.assertIn('SIMULATED',report['settings']['target_label'])
        at.button(key='prepare_export').click().run()
        self.assertEqual(len(at.exception),0)
        all_results=zipfile.ZipFile(io.BytesIO(at.session_state['export']['zip']))
        self.assertIn('separation/report.json',all_results.namelist())

    def test_physical_brightness_inverse_radius(self):
        at=load_app(self.catalog)
        file=self.path/'filter.csv';np.savetxt(file,[[2,0],[3,1],[4,0]],delimiter=',')
        at.radio(key='filter_source').set_value('Local throughput file').run()
        at.text_input(key='filter_path').set_value(str(file)).run()
        at.button(key='calculate_filter').click().run()
        at.radio(key='physical_kind').set_value('Measured contrast').run()
        expected=at.session_state['current']['report']['filter_results'][-1]['planet_star_contrast']
        at.number_input(key='physical_observed').set_value(expected).run()
        at.button(key='physical_build').click().run()
        self.assertEqual(len(at.exception),0)
        rows=at.session_state['physical_result']['rows']
        selected=[row for row in rows if row['temperature_K']==400 and row['radius_Rjup']==1][0]
        self.assertAlmostEqual(selected['radius_to_match_input_Rjup'],1,places=10)
        at.button(key='physical_prepare').click().run()
        self.assertEqual(len(at.exception),0)
        self.assertGreater(len(at.session_state['physical_figures']['png']),1000)
        at.button(key='prepare_export').click().run()
        self.assertEqual(len(at.exception),0)
        z=zipfile.ZipFile(io.BytesIO(at.session_state['export']['zip']))
        self.assertIn('physical_properties/temperature_and_radius.pdf',z.namelist())

    def test_mass_age_planet_used_in_separation_with_gravity_check(self):
        at=load_app(self.catalog)
        # Choose mass such that g=100 m/s^2 at 1 Rjup, matching the fixture grid.
        from atmosphere_compare.evolution import G,MJUP_KG
        mass=100*RADII_M['R_jup']**2/(G*MJUP_KG)
        file=self.path/'evolution.csv'
        file.write_text('mass_Mjup,age_Myr,intrinsic_temperature_K,radius_Rjup\n%.17g,10,700,1\n%.17g,100,600,1\n'%(mass,mass))
        at.text_input(key='evolution_path').set_value(str(file)).run()
        at.button(key='evolution_build').click().run()
        self.assertEqual(len(at.exception),0)
        at.selectbox(key='sep_band').set_value('Custom top-hat').run()
        at.checkbox(key='sep_evolution').check().run()
        at.checkbox(key='sep_heating').check().run()
        at.button(key='sep_build').click().run()
        self.assertEqual(len(at.exception),0)
        result=at.session_state['separation_result']
        self.assertAlmostEqual(result['models'][0]['intrinsic_temperature_K'],700)
        self.assertAlmostEqual(result['models'][0]['rplanet_m'],RADII_M['R_jup'])
        self.assertIn('mass_Mjup',result['evolution_planet'])

    def test_measurement_likelihoods_and_combined_export(self):
        at=load_app(self.catalog)
        file=self.path/'filter.csv';np.savetxt(file,[[2,0],[3,1],[4,0]],delimiter=',')
        at.radio(key='filter_source').set_value('Local throughput file').run()
        at.text_input(key='filter_path').set_value(str(file)).run()
        at.button(key='calculate_filter').click().run()
        at.button(key='inference_build').click().run()
        self.assertEqual(len(at.exception),0)
        result=at.session_state['inference_result']
        self.assertEqual(len(result['rows']),12)
        self.assertAlmostEqual(max(r['relative_likelihood'] for r in result['rows']),1)
        self.assertIn('No prior',result['likelihood']['interpretation'])
        at.button(key='inference_prepare').click().run()
        self.assertEqual(len(at.exception),0)
        at.button(key='prepare_export').click().run()
        z=zipfile.ZipFile(io.BytesIO(at.session_state['export']['zip']))
        self.assertIn('inference/measurement_likelihoods.pdf',z.namelist())
        at.button(key='run').click().run()
        self.assertEqual(len(at.exception),0)
        # A new source comparison hides the old conditional likelihood plot.
        self.assertNotIn('inference_prepare',[b.key for b in at.button])

    def test_evolved_gravity_interpolates_between_native_rows(self):
        rows=json.loads(self.catalog.read_text())
        planets=[r for r in rows if r['role']=='planet']
        rows=[r for r in rows if r['role']=='star']
        for row in planets:
            for g in (3.5,4.5):
                copied=json.loads(json.dumps(row));copied['parameters']['logg_cgs']=g
                # Signals are gravity-independent here, so the independent expected value is exact.
                rows.append(copied)
        self.catalog.write_text(json.dumps(rows))
        at=load_app(self.catalog)
        from atmosphere_compare.evolution import G,MJUP_KG
        mass=100*RADII_M['R_jup']**2/(G*MJUP_KG)
        file=self.path/'evolution.csv'
        file.write_text('mass_Mjup,age_Myr,intrinsic_temperature_K,radius_Rjup\n%.17g,10,700,1\n'%mass)
        at.text_input(key='evolution_path').set_value(str(file)).run()
        at.button(key='evolution_build').click().run()
        at.selectbox(key='sep_band').set_value('Custom top-hat').run()
        at.checkbox(key='sep_evolution').check().run()
        at.button(key='sep_build').click().run()
        self.assertEqual(len(at.exception),0)
        result=at.session_state['separation_result']
        self.assertAlmostEqual(result['temperature_sequence']['evaluated_logg_cgs'],4)
        self.assertEqual(len(result['temperature_sequence']['intrinsic_interpolation_corners']),4)

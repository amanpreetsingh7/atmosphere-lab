"""Worked-example checks with analytic fixtures only; real grids run on the laptop."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from atmosphere_compare.example import independent_band_signal,load_validation,validation_coverage
from atmosphere_compare.compare import band_signal,H,C
from atmosphere_compare.spectra import Spectrum
from atmosphere_compare.validation import audit_catalog
from app_fixtures import write_fixture_catalog,load_app


class ExampleNumerics(unittest.TestCase):
    def test_independent_integration_recovers_cubic_analytic_result(self):
        x=np.asarray([1.,1.7,2.4,3.]);spec=Spectrum(x,x)
        w,t=[1,2,3],[1/3,2/3,1]
        for detector,expected in [('energy',26/9),('photon',(20/3)*1e-6/(H*C))]:
            actual=independent_band_signal(spec,w,t,detector)
            self.assertAlmostEqual(actual/expected,1,places=13)
            self.assertAlmostEqual(actual/band_signal(spec,w,t,detector),1,places=13)
        with self.assertRaises(ValueError):independent_band_signal(spec,[.9,2,3],t,'energy')

    def test_diagnostic_report_reader_and_current_input_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);results=root/'results';results.mkdir()
            model=Spectrum([1,2,3],[1,1,1],{'files':[{'sha256':'source_hash'}],'parameters_declared':{'teff_K':1000}})
            band={'name':'fixture','response':{'wavelength_um':[1,2,3],'throughput':[1,1,1]},'detector':'energy','wavelength_frame':'vacuum'}
            record={'metadata':model.metadata,'parameters':{'teff_K':1000}}
            report={'reports':[{'records':[record]}],'responses':[band]}
            raw=json.dumps(report).encode();path=results/'report.zip'
            with zipfile.ZipFile(path,'w') as z:z.writestr('validation.json',raw)
            (results/'latest_validation.json').write_text(json.dumps({'path':str(path)}))
            loaded=load_validation(root)
            self.assertEqual(loaded['report_sha256'],hashlib.sha256(raw).hexdigest())
            coverage=validation_coverage(loaded,[model,model],band)
            self.assertTrue(all(coverage[k] for k in ('star','planet','response')))
            changed=dict(band,detector='photon')
            self.assertFalse(validation_coverage(loaded,[model,model],changed)['response'])
            changed_model=Spectrum([1,2,3],[1,1,1],{'files':[{'sha256':'different'}],'parameters_declared':{'teff_K':1000}})
            self.assertFalse(validation_coverage(loaded,[model,changed_model],band)['planet'])
            (results/'latest_validation.json').write_text(json.dumps({'path':str(root/'outside.zip')}))
            with self.assertRaisesRegex(ValueError,'outside'):load_validation(root)


class ExampleInterface(unittest.TestCase):
    def test_end_to_end_example_response_diagnostics_and_export(self):
        with tempfile.TemporaryDirectory() as d:
            catalog=write_fixture_catalog(Path(d),(500,1000,1500))
            at=load_app(catalog)
            star=at.session_state['current']['report']['star'].copy()
            at.radio(key='filter_source').set_value('Idealized top-hat (method check)').run()
            at.button(key='calculate_filter').click().run()
            self.assertEqual(len(at.exception),0)
            band=at.session_state['current']['report']['filter_results'][-1]
            self.assertEqual(band['source']['kind'],'idealized_top_hat')
            report={'reports':[audit_catalog(catalog,[band])],'responses':[band]}
            raw=json.dumps(report).encode();validation={'report':report,'report_sha256':hashlib.sha256(raw).hexdigest()}
            with patch('example_ui.load_validation',return_value=validation):
                at.button(key='example_run').click().run()
                self.assertEqual(len(at.exception),0)
                example=at.session_state['example_result']
                self.assertEqual(len(example['checks']),6);self.assertTrue(all(c['passed'] for c in example['checks']))
                at.button(key='example_separation').click().run()
                self.assertEqual(at.session_state['sep_sensitivity_source'],'None')
                at.button(key='sep_build').click().run()
                self.assertEqual(len(at.exception),0)
                self.assertIsNone(at.session_state['separation_result']['observation'])
                at.button(key='example_prepare').click().run()
                self.assertEqual(len(at.exception),0)
                z=zipfile.ZipFile(io.BytesIO(at.session_state['example_export']['zip']))
                saved=json.loads(z.read('example.json'))
                self.assertTrue(all(saved['diagnostic_coverage'][k] for k in ('star','planet','response')))
                self.assertFalse(saved['is_observation'])
                self.assertIn('installed_model_validation.json',z.namelist())
                self.assertIn('separation_curves_and_figures.zip',z.namelist())
                self.assertEqual(star,at.session_state['current']['report']['star'])
                at.button(key='prepare_export').click().run()
                combined=zipfile.ZipFile(io.BytesIO(at.session_state['export']['zip']))
                self.assertIn('worked_example/example.json',combined.namelist())
                at.button(key='run').click().run()
                self.assertNotIn('example_prepare',[b.key for b in at.button])

    def test_invalid_idealized_band_is_not_added_to_results(self):
        with tempfile.TemporaryDirectory() as d:
            at=load_app(write_fixture_catalog(Path(d)))
            at.radio(key='filter_source').set_value('Idealized top-hat (method check)').run()
            at.number_input(key='filter_top_lower').set_value(4.5).run()
            at.number_input(key='filter_top_upper').set_value(4.).run()
            at.button(key='calculate_filter').click().run()
            self.assertEqual(len(at.exception),0)
            self.assertFalse(at.session_state['current']['report']['filter_results'])
            self.assertTrue(any('end must exceed' in e.value for e in at.error))


if __name__=='__main__':unittest.main()

"""Current code identity must be distinguished from old file/response coverage."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from atmosphere_compare import __version__
from atmosphere_compare.spectra import Spectrum
from atmosphere_compare.example import validation_coverage
import validate_installed_models as validator


class DiagnosticProvenance(unittest.TestCase):
    def test_report_contains_current_version_and_science_code_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch.object(validator,'ROOT',root),patch.object(validator,'audit_catalog',return_value={'load_errors':[],'records':[]}):
                validator.main()
            latest=json.loads((root/'results/latest_validation.json').read_text())
            with zipfile.ZipFile(latest['path']) as archive:report=json.loads(archive.read('validation.json'))
            self.assertEqual(report['software_version'],__version__)
            self.assertIn('numpy',report['package_versions'])
            key='atmosphere_compare/spectra.py'
            code=Path(validator.__file__).parent/key
            self.assertEqual(report['source_code_sha256'][key],hashlib.sha256(code.read_bytes()).hexdigest())

    def test_legacy_and_old_version_coverage_are_distinct(self):
        spec=Spectrum([1,2,3],[1,1,1],{'files':[{'sha256':'fixture'}],'parameters_declared':{}})
        band={'response':{'wavelength_um':[1,2,3],'throughput':[0,1,0]},'detector':'photon','wavelength_frame':'vacuum'}
        report={'reports':[],'responses':[band]};validation={'report':report}
        self.assertIsNone(validation_coverage(validation,[spec,spec],band)['software_version_matches_current'])
        report['software_version']='0.0'
        self.assertFalse(validation_coverage(validation,[spec,spec],band)['software_version_matches_current'])
        report['software_version']=__version__
        self.assertTrue(validation_coverage(validation,[spec,spec],band)['software_version_matches_current'])

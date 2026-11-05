"""Updater must preserve the downloaded models and user configuration."""
from pathlib import Path
import json
import tempfile
import unittest
from apply_update import update


class UpdateChecks(unittest.TestCase):
    def test_code_backup_and_model_config_preservation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source,target=root/'source',root/'target'
            source.mkdir(); target.mkdir()
            for path,content in [('app.py','# new app\n'),('separation_ui.py','# new plots\n'),('launch_app.py','# new launch\n')]:
                (source/path).write_text(content)
                (target/path).write_text('old '+path)
            (target/'models/archives').mkdir(parents=True)
            archive=target/'models/archives/model.tar.gz.part'
            archive.write_bytes(b'precious partial model download')
            (target/'configs').mkdir()
            catalog=target/'configs/app_catalog.json'
            content=json.dumps([{'role':'star','parameters':{'teff_K':6000},'spectrum':{'label':'My custom star','file':'my.fits'}}])
            catalog.write_text(content)
            result=update(source,target)
            self.assertEqual(archive.read_bytes(),b'precious partial model download')
            self.assertEqual(catalog.read_text(),content)
            self.assertEqual((target/'app.py').read_text(),'# new app\n')
            self.assertEqual((Path(result['backup'])/'app.py').read_text(),'old app.py')

    def test_invalid_target_does_not_create_project_or_touch_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source=root/'source'; source.mkdir()
            (source/'app.py').write_text('new')
            with self.assertRaises(ValueError):
                update(source,root/'missing')
            self.assertFalse((root/'missing').exists())

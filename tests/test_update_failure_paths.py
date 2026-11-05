"""Update the installed project, not just the extracted source directory.

Synthetic project files and forced I/O failures; no real models or downloads.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import apply_update


class UpdateFailurePaths(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.root=Path(self.directory.name)
        self.source=self.root/'source';self.target=self.root/'target'
        self.source.mkdir();self.target.mkdir()
        for folder,label in ((self.source,'new'),(self.target,'old')):
            for name in ('app.py','separation_ui.py','launch_app.py','apply_update.py'):
                (folder/name).write_text('# '+label+' '+name+'\n')
        for name,data in {'models/archives/cached.tar.gz':b'archive',
                          'models/planet.dat':b'native model',
                          'configs/app_catalog.json':b'[{"spectrum":{"label":"custom"}}]',
                          'configs/real_models.json':b'{"custom":true}',
                          'results/saved.json':b'{"user_result":true}',
                          'pilot_recovery.py':b'# user research module\n'}.items():
            path=self.target/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    def tearDown(self):self.directory.cleanup()
    def original(self):
        return {str(p.relative_to(self.target)):p.read_bytes() for p in self.target.rglob('*') if p.is_file() and not any(part.startswith('code_backup_') for part in p.parts)}

    def test_installs_updater_itself_and_software_record(self):
        (self.source/'SOFTWARE_CHECKS.json').write_text('{}')
        (self.source/'RECHECK_RESULTS.json').write_text('{"checked":true}')
        (self.source/'software_checks.log').write_text('checked')
        protected={k:v for k,v in self.original().items() if k.startswith(('models/','configs/','results/','pilot_'))}
        apply_update.update(self.source,self.target)
        self.assertEqual((self.target/'apply_update.py').read_bytes(),(self.source/'apply_update.py').read_bytes())
        self.assertEqual((self.target/'SOFTWARE_CHECKS.json').read_text(),'{}')
        self.assertEqual((self.target/'RECHECK_RESULTS.json').read_text(),'{"checked":true}')
        self.assertEqual((self.target/'software_checks.log').read_text(),'checked')
        for name,data in protected.items():self.assertEqual((self.target/name).read_bytes(),data)

    def test_commit_failure_restores_all_replaced_files(self):
        before=self.original();real_replace=os.replace;failed=False
        def fail_once(source,destination):
            nonlocal failed
            if Path(destination).resolve()==(self.target/'separation_ui.py').resolve() and not failed:
                failed=True;raise OSError('forced commit failure')
            return real_replace(source,destination)
        with patch.object(apply_update.os,'replace',side_effect=fail_once):
            with self.assertRaisesRegex(OSError,'were restored'):apply_update.update(self.source,self.target)
        self.assertTrue(failed);self.assertEqual(before,self.original())

    def test_staging_failure_leaves_installed_files_unchanged(self):
        before=self.original();write=Path.write_bytes
        def fail_stage(path,data):
            if any(part.startswith('.atmosphere_update_stage_') for part in path.parts):raise OSError('forced staging failure')
            return write(path,data)
        with patch.object(Path,'write_bytes',new=fail_stage):
            with self.assertRaises(OSError):apply_update.update(self.source,self.target)
        self.assertEqual(before,self.original())

    def test_changed_checked_source_stops_before_install(self):
        expected=hashlib.sha256((self.source/'app.py').read_bytes()).hexdigest()
        (self.source/'SOFTWARE_CHECKS.json').write_text(json.dumps({'code_sha256':{'app.py':expected}}))
        (self.source/'app.py').write_text('# changed after checks\n')
        before=self.original()
        with self.assertRaisesRegex(ValueError,'differs'):apply_update.update(self.source,self.target)
        self.assertEqual(before,self.original())

    def test_nested_alias_cannot_update_itself(self):
        actual=self.source/'Atmosphere_Lab';actual.mkdir()
        (actual/'launch_app.py').write_text('pass\n')
        (self.source/'launch_app.py').unlink()
        with self.assertRaisesRegex(ValueError,'resolves'):apply_update.update(actual,self.source)

    def test_invalid_python_stops_before_any_replacement(self):
        (self.source/'app.py').write_text('def broken(\n')
        before=self.original()
        with self.assertRaisesRegex(ValueError,'invalid Python'):apply_update.update(self.source,self.target)
        self.assertEqual(before,self.original())

    def test_legacy_label_cleanup_is_rolled_back_on_failure(self):
        catalog=self.target/'configs/app_catalog.json'
        catalog.write_text(json.dumps([{'spectrum':{'label':"PHOENIX 5400 K (grid approximation to Kevin's 5401 K)"}}]))
        before=self.original();real_replace=os.replace;failed=False
        def fail_once(source,destination):
            nonlocal failed
            if Path(destination).resolve()==catalog.resolve() and not failed:failed=True;raise OSError('forced catalog commit failure')
            return real_replace(source,destination)
        with patch.object(apply_update.os,'replace',side_effect=fail_once):
            with self.assertRaisesRegex(OSError,'were restored'):apply_update.update(self.source,self.target)
        self.assertTrue(failed);self.assertEqual(before,self.original())

    def test_unrecognized_research_hook_is_not_overwritten(self):
        (self.target/'app.py').write_text('# research_pilot_stage1\n# customized hook\n')
        (self.target/'pilot_ui.py').write_text('# custom research module\n')
        before=self.original()
        with self.assertRaisesRegex(ValueError,'hook has changed'):apply_update.update(self.source,self.target)
        self.assertEqual(before,self.original())

    def test_active_model_job_blocks_update_without_changes(self):
        import fcntl
        lock=self.target/'models/install_logs/installation.lock';lock.parent.mkdir(parents=True);lock.touch()
        before=self.original()
        with lock.open('a') as stream:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError,'owns the model cache'):apply_update.update(self.source,self.target)
        self.assertEqual(before,self.original())


if __name__=='__main__':unittest.main()

"""PyCharm software validation with environment preflight and traceable reports.

Science fixtures in this suite are synthetic. Native installed-model validation
and observational PSF/recovery checks are separate stages.
"""
from pathlib import Path
from datetime import datetime
import hashlib
import importlib
from importlib.metadata import version, PackageNotFoundError
import io
import json
import platform
import shlex
import shutil
import sys
import unittest
import zipfile

ROOT = Path(__file__).resolve().parent
DEPENDENCIES = {
    'numpy': 'numpy', 'matplotlib': 'matplotlib', 'astropy': 'astropy',
    'streamlit': 'streamlit', 'streamlit.testing.v1': 'streamlit',
    'plotly': 'plotly', 'pandas': 'pandas', 'scipy.integrate': 'scipy',
    'scipy.special': 'scipy', 'netCDF4': 'netCDF4', 'pyarrow': 'pyarrow',
}


def environment_check():
    packages, errors = {}, []
    for module, distribution in DEPENDENCIES.items():
        try:
            imported = importlib.import_module(module)
            if module == 'streamlit.testing.v1' and not hasattr(imported, 'AppTest'):
                raise ImportError('Streamlit AppTest is unavailable in this version.')
        except Exception as exc:
            errors.append({'module': module, 'error': type(exc).__name__+': '+str(exc)})
        if distribution not in packages:
            try:packages[distribution] = version(distribution)
            except PackageNotFoundError:packages[distribution] = 'not installed'
    return {'python_executable': sys.executable, 'python_version': platform.python_version(),
            'platform': platform.platform(), 'packages': packages, 'import_errors': errors}


class Tee:
    def __init__(self, first, second):self.streams = (first, second)
    def write(self, text):
        for stream in self.streams:stream.write(text)
        return len(text)
    def flush(self):
        for stream in self.streams:stream.flush()


def main(root=None):
    root = Path(root).resolve() if root is not None else ROOT
    print('Validation interpreter:', sys.executable, flush=True)
    print('Validation project:', root, flush=True)
    environment = environment_check()
    summary = {'tests_run': 0, 'passed': False, 'failures': [], 'errors': [],
               'scope': 'Synthetic/analytic software and interface checks; not native-grid accuracy or observational validation.'}
    log, failure_text = io.StringIO(), ''
    if environment['import_errors']:
        summary['status'] = 'environment_not_ready'
        summary['errors'] = environment['import_errors']
        print('Environment preflight failed. The suite was not started.', flush=True)
        for error in environment['import_errors']:print(error['module']+': '+error['error'], flush=True)
        print('Run install_requirements.py in this same PyCharm interpreter, then rerun this file.', flush=True)
        print('Terminal alternative:', shlex.join([sys.executable, '-m', 'pip', 'install', '-r', str(root/'requirements.txt')]), flush=True)
        exit_code = 2
    else:
        suite = unittest.defaultTestLoader.discover(str(root/'tests'))
        result = unittest.TextTestRunner(stream=Tee(sys.stderr, log), verbosity=2).run(suite)
        summary.update(tests_run=result.testsRun, passed=result.wasSuccessful(),
            status='passed' if result.wasSuccessful() else 'tests_failed',
            failures=[str(t) for t, _ in result.failures], errors=[str(t) for t, _ in result.errors],
            skipped=[{'test': str(t), 'reason': reason} for t, reason in result.skipped],
            discovery_import_errors=[str(t) for t, _ in result.errors if isinstance(t, unittest.loader._FailedTest)])
        failure_text = '\n\n'.join(trace for _, trace in result.failures+result.errors)
        exit_code = 0 if summary['passed'] else 1
    from atmosphere_compare import __version__
    summary['software_version'] = __version__
    source = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob('*.py'))
              if not any(part.startswith(('code_backup_', '.')) or part in ('__pycache__', 'venv', 'models', 'results')
                         for part in p.relative_to(root).parts)}
    folder = root/'results';folder.mkdir(exist_ok=True)
    path = folder/('atmosphere_validation_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, value in [('summary.json', summary), ('environment.json', environment), ('source_sha256.json', source)]:
            archive.writestr(name, json.dumps(value, indent=2))
        archive.writestr('failures.txt', failure_text)
        archive.writestr('test_log.txt', log.getvalue())
    (folder/'latest_software_checks.json').write_text(json.dumps({'path': str(path), 'summary': summary}, indent=2))
    desktop = Path.home()/'Desktop'
    if desktop.is_dir():
        try:shutil.copy2(path, desktop/path.name);path = desktop/path.name
        except OSError:pass
    print('Results ZIP:', path, flush=True)
    return exit_code


if __name__ == '__main__':raise SystemExit(main())

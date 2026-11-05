"""Run from the extracted update to update the existing laptop project in place.

Models, archive downloads, and configs are retained. Replaced code is backed up.
"""
from pathlib import Path
from datetime import datetime
import ast
import hashlib
import json
import os
import shutil
import tempfile

TARGET = Path.home()/"Downloads"/"Atmosphere_Lab 3"
SOURCE = Path(__file__).resolve().parent

FILES = ("apply_update.py", "update_current_example.py", "app.py", "parameter_ui.py", "setup_parameter_models.py", "separation_ui.py", "model_library_ui.py", "physical_ui.py", "inference_ui.py", "example_ui.py", "validation_ui.py", "launch_app.py", "install_requirements.py", "requirements.txt", "setup_real_models.py",
    "setup_evolution_models.py", "setup_gravity_models.py", "setup_cholla_models.py", "setup_atmo_models.py", "validate_installed_models.py",
    "setup_separation_models.py", "run_models.py", "run_laptop_validation.py", "run_demo.py",
    "README.md", "START_HERE.txt", "CONTRAST_PLOT_GUIDE.md", "CHANGELOG.md", "MODEL_DOWNLOADS.md",
    "VALIDATION_STATUS.json", "SOFTWARE_CHECKS.json", "RECHECK_RESULTS.json", "software_checks.log", "compatible_stack_checks.log", "installed_update_checks.log", "PROJECT_SCOPE.md", "SCIENCE_REVIEW.md", "SHOW_THE_LAB.md", "FULL_EXAMPLE.txt", "PARAMETER_GUIDE.txt")
FOLDERS = ("atmosphere_compare", "tests", ".streamlit", "templates", "evidence/review_checks")


def update(source=SOURCE, target=TARGET):
    destination=Path(target).expanduser().resolve()
    if not (destination/'launch_app.py').is_file() and (destination/'Atmosphere_Lab/launch_app.py').is_file():
        destination=destination/'Atmosphere_Lab'
    lock=destination/'models/install_logs/installation.lock'
    if lock.is_file():
        import fcntl
        with lock.open('a') as stream:
            try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('An installation or diagnostic job still owns the model cache. Finish or stop that job before updating. No code was changed.')
            return _perform_update(source,destination)
    return _perform_update(source,target)


def _perform_update(source, target):
    source, target = Path(source).resolve(), Path(target).expanduser().resolve()
    if source == target:
        raise ValueError("This folder already contains the update. Run launch_app.py here, or change TARGET to your previous project.")
    if not (target/"launch_app.py").is_file():
        nested = target/"Atmosphere_Lab"
        if (nested/"launch_app.py").is_file():
            target = nested
        else:
            raise ValueError("Existing project not found at %s. Set TARGET to the folder containing its launch_app.py." % target)
    if source == target:
        raise ValueError("The selected target resolves to this extracted update folder; no files were changed.")
    paths = [Path(name) for name in FILES if (source/name).is_file()]
    for folder in FOLDERS:
        paths.extend(p.relative_to(source) for p in (source/folder).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    if not (source/"app.py").is_file() or not (source/"separation_ui.py").is_file():
        raise ValueError("The extracted update is incomplete.")
    # Preserve the separately installed research panel without shipping it as
    # part of this atmosphere-comparison release or replacing its modules.
    old_app=(target/'app.py').read_text()
    new_app=(source/'app.py').read_text()
    research_insert='\n\n# research_pilot_stage1\nfrom pilot_ui import render as render_research_pilot\nrender_research_pilot(ROOT)'
    if '# research_pilot_stage1' in old_app:
        if research_insert not in old_app or not (target/'pilot_ui.py').is_file():
            raise ValueError('The installed research-panel hook has changed or its module is missing. No files were updated; review that hook before applying this update.')
        if '# research_pilot_stage1' not in new_app:
            if new_app.count('render_model_library(ROOT)')!=1:
                raise ValueError('Cannot preserve the installed research-panel hook. No files were updated.')
            new_app=new_app.replace('render_model_library(ROOT)','render_model_library(ROOT)'+research_insert,1)
        ast.parse(new_app,feature_version=(3,9))
    proposed={relative:(source/relative).read_bytes() for relative in paths}
    # Validate the distributed source before adding a retained research hook.
    manifest=source/'SOFTWARE_CHECKS.json'
    if manifest.is_file():
        for name,expected in json.loads(manifest.read_text()).get('code_sha256',{}).items():
            relative=Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Invalid source path in software-check record.')
            file=source/relative
            if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=expected:
                raise ValueError('Update source is incomplete or differs from its checked code: '+name+'. No files were updated.')
    proposed[Path('app.py')]=new_app.encode('utf-8')
    for relative,data in proposed.items():
        if relative.suffix=='.py':
            try:ast.parse(data,filename=str(relative),feature_version=(3,9))
            except SyntaxError as exc:raise ValueError('Update contains invalid Python 3.9 syntax: '+str(relative)) from exc
    # Remove only the exact autogenerated label from the superseded release.
    catalog = target/"configs/app_catalog.json"
    label_changed = False
    if catalog.is_file():
        try:
            rows = json.loads(catalog.read_text())
            for row in rows:
                spectrum = row.get("spectrum", {})
                if spectrum.get("label") == "PHOENIX 5400 K (grid approximation to Kevin's 5401 K)":
                    spectrum["label"] = "PHOENIX: 5400 K, log g 4.5, solar metallicity"
                    label_changed = True
            if label_changed:
                proposed[Path('configs/app_catalog.json')]=json.dumps(rows,indent=2).encode('utf-8')
        except (ValueError, TypeError, AttributeError):
            print("Catalog format was not recognized; left it unchanged.", flush=True)
    for relative in proposed:
        destination=target/relative
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise ValueError('A code destination is a symlink or directory: '+str(relative)+'. No files were updated.')
    backup = target/("code_backup_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    # Prepare every file before changing existing code. Replace each file
    # atomically; if any commit fails, restore all already-replaced files.
    with tempfile.TemporaryDirectory(prefix='.atmosphere_update_stage_',dir=target) as folder:
        staged=Path(folder)
        for relative,data in proposed.items():
            file=staged/relative;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(data)
        backup.mkdir()
        existed={relative:(target/relative).is_file() for relative in proposed}
        for relative in proposed:
            if existed[relative]:
                saved=backup/relative;saved.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(target/relative,saved)
        written=[]
        try:
            for relative in proposed:
                destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True)
                os.replace(staged/relative,destination);written.append(relative)
        except BaseException as exc:
            # Free staging space before restoration, including on a full disk.
            shutil.rmtree(staged,ignore_errors=True)
            failures=[]
            for relative in reversed(written):
                destination=target/relative
                try:
                    if existed[relative]:
                        restored=destination.with_name(destination.name+'.restore.part')
                        shutil.copy2(backup/relative,restored);os.replace(restored,destination)
                    else:destination.unlink()
                except OSError as restore_error:failures.append(str(relative)+': '+str(restore_error))
            if failures:
                raise OSError('Update failed and automatic restoration was incomplete. Keep the backup at '+str(backup)+'. '+ '; '.join(failures)) from exc
            raise OSError('Update failed; replaced files were restored. Backup: '+str(backup)) from exc
    print("UPDATED:", target, flush=True)
    print("Downloaded models and archive cache were retained. Code backup:", backup, flush=True)
    print("Open this existing project in PyCharm and Run launch_app.py.", flush=True)
    return {"target": str(target), "backup": str(backup), "files_updated": len(proposed), "obsolete_label_cleaned": label_changed}


if __name__ == "__main__":
    try:
        update()
    except (OSError, ValueError) as exc:
        print("UPDATE STOPPED:", exc, flush=True)
        raise SystemExit(1)

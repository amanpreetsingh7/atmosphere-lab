"""PyCharm entry point for an editable property request; no atmosphere constants."""
from pathlib import Path
from datetime import datetime
import json
import shutil
from atmosphere_compare.properties import predict_request

ROOT = Path(__file__).resolve().parent
REQUEST_FILE = ROOT/'configs/atmosphere_request.json'


def main(request_file=None):
    from prediction_exports import model_access, export_result
    request_file = Path(request_file).expanduser().resolve() if request_file else REQUEST_FILE
    if not request_file.exists():
        if request_file != REQUEST_FILE:raise FileNotFoundError('Request JSON not found: '+str(request_file))
        request_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/'templates/atmosphere_request.json', request_file)
        print('Created editable request:', request_file, flush=True)
    request = json.loads(request_file.read_text())
    print('Reading explicit model properties from:', request_file, flush=True)
    print('Star:', request['star'], flush=True)
    print('Companion:', request['companion'], flush=True)
    with model_access(ROOT):
        result = predict_request(request, ROOT)
        name = 'atmosphere_prediction_'
        output = export_result(result, ROOT/'results'/(name+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
        desktop = Path.home()/'Desktop'
        def copy_to_desktop(path):
            if desktop.is_dir():
                try:shutil.copy2(path, desktop/path.name)
                except OSError:return path
                return desktop/path.name
            return path
        print('Predicted band contrast:', result['planet_star_contrast'], flush=True)
        print('Delta magnitude:', result['delta_magnitude'], flush=True)
        print('UPLOAD RESULTS:', copy_to_desktop(output), flush=True)
        print('The request is reusable and editable; inspect source selections/interpolation assumptions in prediction.json.', flush=True)
    return result


if __name__ == '__main__':
    try:main()
    except Exception as exc:
        print('ATMOSPHERE REQUEST STOPPED:', exc, flush=True)
        raise SystemExit(1)

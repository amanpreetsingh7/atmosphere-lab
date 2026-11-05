"""Install available Bobcat temperature nodes without assuming missing filenames."""
from setup_gravity_models import main

if __name__ == '__main__':
    try:main(gravities=(100,))
    except Exception as exc:
        print('TEMPERATURE INSTALL STOPPED:',exc,flush=True)
        raise SystemExit(1)

from pathlib import Path
import gzip
import hashlib
import io
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from atmosphere_compare.downloads import download, extract_selected
from atmosphere_compare.spectra import load_spectrum


class FakeResponse(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}

    def getcode(self):
        return self.status


class DownloadChecks(unittest.TestCase):
    def test_resume_checks_byte_range(self):
        payload = b"abcdef0123456789"
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/"file.dat"
            target.with_name("file.dat.part").write_bytes(payload[:5])
            response = FakeResponse(payload[5:],206,{"Content-Range":"bytes 5-15/16"})
            with patch("atmosphere_compare.downloads.urllib.request.urlopen",return_value=response) as opened:
                download("https://example.invalid/model",target,len(payload),hashlib.md5(payload).hexdigest())
                self.assertEqual(opened.call_args.args[0].get_header("Range"),"bytes=5-")
            self.assertEqual(target.read_bytes(),payload)

    def test_server_ignoring_range_restarts_file(self):
        payload = b"correct_complete_payload"
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/"file.dat"
            target.with_name("file.dat.part").write_bytes(b"wrong")
            with patch("atmosphere_compare.downloads.urllib.request.urlopen",return_value=FakeResponse(payload)):
                download("https://example.invalid/model",target,len(payload),hashlib.md5(payload).hexdigest())
            self.assertEqual(target.read_bytes(),payload)

    def test_checksum_failure_does_not_finalize(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/"file.dat"
            with patch("atmosphere_compare.downloads.urllib.request.urlopen",return_value=FakeResponse(b"bad")):
                with self.assertRaises(ValueError):
                    download("https://example.invalid/model",target,3,hashlib.md5(b"yes").hexdigest())
            self.assertFalse(target.exists())

    def test_only_selected_archive_member_extracted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive=root/"models.tar.gz"
            with tarfile.open(archive,"w:gz") as t:
                for name,value in (("../../outside.dat",b"ignore"),("spectra/model.gz",b"selected")):
                    info=tarfile.TarInfo(name);info.size=len(value);t.addfile(info,io.BytesIO(value))
            result=extract_selected(archive,{"spectra/model.gz":root/"chosen.gz"})
            self.assertEqual((root/"chosen.gz").read_bytes(),b"selected")
            self.assertEqual(set(result),{"spectra/model.gz"})

    def test_actual_bobcat_header_layout_in_gzip_fixture(self):
        header="1000. 100. 0.28 0.00 1.0000E+05 0.00 1.00 1.00 Teff, grav (MKS), Y, f_rain, Kzz, [Fe/H], C/O, f_hole\n"
        data=header+"microns Flux (erg/cm^2/s/Hz)\n5.0 1D-8\n4.0 2D-8\n3.0 3D-8\n"
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"sp_t1000g100nc_m0.0.gz"
            with gzip.open(path,"wt") as f:f.write(data)
            spec=load_spectrum({"format":"sonora_bobcat","file":path.name,
                                "parameters":{"teff_K":1000,"logg_cgs":4.0,"metallicity_dex":0.0,"co_relative_solar":1.0}},folder)
            self.assertEqual(spec.metadata["header_lines_skipped"],2)
            self.assertEqual(spec.metadata["parameters_verified_from_header"]["teff_K"],1000)
            np.testing.assert_allclose(spec.wavelength_um,[3,4,5])


if __name__ == "__main__":
    unittest.main()

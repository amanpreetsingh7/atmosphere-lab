"""Regression checks for scientific edge cases found in the October 15 audit.

Only analytic arrays and synthetic file fixtures are used here.
"""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from atmosphere_compare.spectra import Spectrum, load_spectrum
from atmosphere_compare.compare import integral, band_contrast, equivalent_flux_density_Jy, PC_M, C
from atmosphere_compare.inference import likelihood_grid
from apply_update import update


class ScienceAuditChecks(unittest.TestCase):
    def test_faint_bin_survives_a_bright_region(self):
        spec=Spectrum([1,2,3,4],[1e20,1e20,1.,1.])
        np.testing.assert_allclose(integral(spec,[3,3.5,4]),[.5,.5],rtol=1e-15)
        all_bins=integral(spec,[1,2,3,3.5,4])
        np.testing.assert_allclose(all_bins[-2:],[.5,.5],rtol=1e-15)

    def test_local_bins_integrate_a_linear_function_exactly(self):
        x=np.array([1.,1.3,2.,2.8,3.]);spec=Spectrum(x,2*x+3)
        e=np.array([1.01,1.22,1.8,2.81,2.99])
        expected=np.diff(e**2+3*e)
        np.testing.assert_allclose(integral(spec,e),expected,rtol=1e-13)

    def test_direct_band_ratio_rejects_air_vacuum_mixing(self):
        air=Spectrum([1,2,3],[1,1,1],{'wavelength_frame':'air'})
        vacuum=Spectrum([1,2,3],[1,1,1],{'wavelength_frame':'vacuum'})
        with self.assertRaisesRegex(ValueError,'vacuum'):
            band_contrast(air,vacuum,1,[1,2,3],[1,1,1])

    def test_phoenix_wrong_or_missing_units_are_rejected_at_load(self):
        from astropy.io import fits
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);settings={'format':'phoenix_hires','file':'flux.fits','wavelength_file':'wave.fits'}
            fits.writeto(root/'flux.fits',np.ones(3),fits.Header({'BUNIT':'erg/s/cm^2/Angstrom'}))
            fits.writeto(root/'wave.fits',np.array([10000.,20000.,30000.]),fits.Header({'UNIT':'Angstrom'}))
            with self.assertRaisesRegex(ValueError,'units'):load_spectrum(settings,root)
            with fits.open(root/'flux.fits',mode='update') as h:h[0].header['BUNIT']='erg/s/cm^2/cm'
            load_spectrum(settings,root)
            with fits.open(root/'wave.fits',mode='update') as h:h[0].header.pop('UNIT')
            with self.assertRaisesRegex(ValueError,'headers'):load_spectrum(settings,root)

    def test_bobcat_wrong_native_units_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'model.dat'
            p.write_text('1000 100 .28 0 0 0 1 0\nwavelength in microns, flux in W/m2/m\n1 1\n2 1\n3 1\n')
            with self.assertRaisesRegex(ValueError,'unit header'):
                load_spectrum({'format':'sonora_bobcat','file':str(p)},folder)

    def test_original_atmo_filename_and_units_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'spec_T500_lg4.0_CEQ.txt'
            header='# ATMO emission spectrum\n# Wavelength [microns]  |  Flux [W/m2/micron]\n'
            p.write_text(header+'1 1\n2 2\n3 3\n')
            settings={'format':'atmo2020_ascii','file':str(p)}
            result=load_spectrum(settings,folder)
            self.assertEqual(result.metadata['parameters_verified_from_filename']['logg_cgs'],4.)
            with self.assertRaisesRegex(ValueError,'teff_K'):
                load_spectrum(dict(settings,parameters={'teff_K':600}),folder)
            p.write_text(header.replace('W/m2/micron','W/m2/m')+'1 1\n2 2\n3 3\n')
            with self.assertRaisesRegex(ValueError,'header'):load_spectrum(settings,folder)

    def test_cholla_and_diamondback_reject_per_micron_native_headers(self):
        with tempfile.TemporaryDirectory() as folder:
            for fmt,name in (('sonora_cholla','1000K_100g_logkzz4.spec'),
                             ('sonora_diamondback','t1000g100f2_m+0.0_co1.0.spec')):
                p=Path(folder)/name
                for unit in ('W/m2/micron','W/m2/um','W/m2/nm','W/m2/m/sr','W/m2/m / sr','W/m²/micron'):
                    with self.subTest(format=fmt,unit=unit):
                        p.write_text('wavelength (microns) flux ('+unit+')\n1 1\n2 2\n3 3\n')
                        with self.assertRaisesRegex(ValueError,'header'):
                            load_spectrum({'format':fmt,'file':name},folder)

    def test_cholla_and_diamondback_accept_complete_per_metre_units(self):
        with tempfile.TemporaryDirectory() as folder:
            for fmt,name in (('sonora_cholla','1000K_100g_logkzz4.spec'),
                             ('sonora_diamondback','t1000g100f2_m+0.0_co1.0.spec')):
                p=Path(folder)/name
                for unit in ('W/m2/m','W/m^2/m','W/m²/m','W / m^2 / m'):
                    with self.subTest(format=fmt,unit=unit):
                        p.write_text('wavelength (microns) flux '+unit+'\n1 1000000\n2 2000000\n3 3000000\n')
                        spec=load_spectrum({'format':fmt,'file':name},folder)
                        np.testing.assert_allclose(spec.surface_flux,[1.,2.,3.],rtol=1e-15)

    def test_converted_cholla_can_use_explicit_ascii_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'converted.txt'
            p.write_text('1 1\n2 2\n3 3\n')
            spec=load_spectrum({'format':'ascii','file':str(p),'wavelength_unit':'um',
                                'flux_unit':'W_m2_um','flux_convention':'surface_flux'},folder)
            np.testing.assert_allclose(spec.surface_flux,[1.,2.,3.],rtol=1e-15)

    def test_atmo_header_cannot_contradict_its_filename(self):
        from astropy.io import fits
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'spec_t500_lg5.0_ceq.fits'
            cols=[fits.Column(name='WAVELENGTH',array=[10000.,20000.,30000.],format='D',unit='ANGSTROMS'),
                  fits.Column(name='FLUX',array=[1.,2.,3.],format='D',unit='FLAM')]
            hdu=fits.BinTableHDU.from_columns(cols);hdu.header['TEFF']=600;hdu.writeto(p)
            with self.assertRaisesRegex(ValueError,'teff_K'):
                load_spectrum({'format':'atmo2020_stsci_fits','file':str(p)},folder)

    def test_likelihood_requires_measurement_information(self):
        with self.assertRaises(ValueError):likelihood_grid(np.empty((2,0)),[])

    def test_likelihood_model_error_keeps_gaussian_normalization(self):
        predictions=np.array([[1.],[2.]])
        sigma=.1;fraction=.4;obs=[{'contrast':1.,'sigma':sigma,'kind':'detection'}]
        result=likelihood_grid(predictions,obs,model_fractional_error=fraction)
        noise=np.sqrt(sigma**2+(fraction*predictions[:,0])**2)
        loglike=-.5*((1-predictions[:,0])/noise)**2-np.log(noise)
        np.testing.assert_allclose(result['relative_likelihood'],np.exp(loglike-loglike.max()),rtol=1e-14)

    def test_equivalent_jy_for_an_analytic_flat_fnu_source(self):
        wave=np.geomspace(1.,5.,30000)
        # One W m^-2 Hz^-1 surface Fnu; dilution chosen independently.
        spec=Spectrum(wave,C/(wave*1e-6)**2*1e-6)
        radius=7e7;distance=10.;expected=(radius/(distance*PC_M))**2/1e-26
        for detector in ('photon','energy'):
            actual=equivalent_flux_density_Jy(spec,radius,distance,[1,2,3,4,5],[0,.4,1,.7,0],detector)
            self.assertAlmostEqual(actual/expected,1.,places=6)

    def test_updater_preserves_installed_research_panel(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';target=root/'target';source.mkdir();target.mkdir()
            anchor='render_model_library(ROOT)'
            hook='\n\n# research_pilot_stage1\nfrom pilot_ui import render as render_research_pilot\nrender_research_pilot(ROOT)'
            (source/'app.py').write_text(anchor+'\n')
            (target/'app.py').write_text(anchor+hook+'\n')
            for base in (source,target):
                (base/'launch_app.py').write_text('pass\n');(base/'separation_ui.py').write_text('pass\n')
            (target/'pilot_ui.py').write_text('research_code = True\n')
            update(source,target)
            self.assertIn(hook,(target/'app.py').read_text())
            self.assertEqual((target/'pilot_ui.py').read_text(),'research_code = True\n')

    def test_gravity_installer_can_register_the_atmo_overlap_node(self):
        import gzip,io,tarfile
        from unittest.mock import patch
        import setup_gravity_models as installer
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);archive=root/'fixture.tar.gz'
            data=gzip.compress(b'1000 1000 .28 0 0 0 1 0\nwavelength in microns; flux in erg/cm^2/s/Hz\n1 1e-15\n2 2e-15\n3 3e-15\n')
            with tarfile.open(archive,'w:gz') as stream:
                member=tarfile.TarInfo('spectra/sp_t1000g1000nc_m0.0.gz');member.size=len(data);stream.addfile(member,io.BytesIO(data))
            with patch.object(installer,'ROOT',root),patch.object(installer,'download',return_value=archive):
                installer.main()
            import json
            rows=json.loads((root/'configs/app_catalog.json').read_text())
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['parameters']['logg_cgs'],5.)


if __name__=='__main__':unittest.main()

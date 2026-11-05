"""Analytic spectra, manufactured NetCDFs and fake HTTP ranges; no native science runs."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from netCDF4 import Dataset
from atmosphere_compare.spectra import load_spectrum,Spectrum
from atmosphere_compare.parameter_exploration import explore,export_bundle
from atmosphere_compare.remote_zip import RangeFile,selected_members,validate_member
from atmosphere_compare.model_library import register_folder
from app_fixtures import write_fixture_catalog,load_app
from setup_parameter_models import wanted_members


def manufactured_netcdf(path,flame=True,gravity_unit='cm / s2',flux_unit='erg/cm**2/s/cm',m=0,co=1):
    with Dataset(path,'w',format='NETCDF4') as ds:
        ds.createDimension('wavelength',101)
        w=ds.createVariable('wavelength','f8',('wavelength',));w.units='micron';w[:]=np.linspace(6,1,101)
        f=ds.createVariable('flux_emission' if flame else 'flux','f8',('wavelength',));f.units=flux_unit;f[:]=np.ones(101)*1e7
        if flame:
            params={'effective_temp':{'value':1000,'unit':'K'},'gravity':{'value':10000 if gravity_unit.startswith('cm') else 100,'unit':gravity_unit},'logmh':m,'cto':{'value':co,'unit':'relative to Solar (=0.458)'}}
            ds.rebinned_resolution='R=3000'
        else:
            params={'teff':{'value':1000,'unit':'K'},'logg':{'value':100,'unit':'m / s2'},'mh':m,'cto':co,'logkzz':4}
        ds.planet_params=json.dumps(params)


class NativeParameters(unittest.TestCase):
    def setUp(self):self.directory=tempfile.TemporaryDirectory();self.root=Path(self.directory.name)
    def tearDown(self):self.directory.cleanup()
    def load(self,path,fmt,parameters=None):return load_spectrum({'file':str(path),'format':fmt,'parameters':parameters or {}},self.root)

    def test_flame_units_gravity_and_co_are_verified(self):
        for unit in ('cm / s2','m / s2'):
            p=self.root/'spectra_teff_1000_grav_100_mh_+0.0_co_1.0_R3000.nc';manufactured_netcdf(p,gravity_unit=unit)
            s=self.load(p,'sonora_flame_skimmer_v1_r3k')
            self.assertTrue(np.all(s.surface_flux==1)) # 1e7 erg/cm²/s/cm -> 1 W/m²/um, no 4π
            self.assertEqual(s.metadata['parameters_verified_from_header']['logg_cgs'],4)
            self.assertEqual(s.metadata['parameters_verified_from_header']['co_relative_solar'],1)
            self.assertEqual(s.metadata['native_resolving_power'],3000)
            self.assertGreater(s.wavelength_um[-1],s.wavelength_um[0])

    def test_flame_rejects_converted_units_identity_and_resolution(self):
        p=self.root/'spectra_teff_1000_grav_100_mh_+0.0_co_1.0_R3000.nc'
        manufactured_netcdf(p,flux_unit='W/m2/um')
        with self.assertRaises(ValueError):self.load(p,'sonora_flame_skimmer_v1_r3k')
        manufactured_netcdf(p,m=.5)
        with self.assertRaisesRegex(ValueError,'disagrees'):self.load(p,'sonora_flame_skimmer_v1_r3k')
        manufactured_netcdf(p)
        with self.assertRaisesRegex(ValueError,'disagrees'):self.load(p,'sonora_flame_skimmer_v1_r3k',{'logg_cgs':5})
        with Dataset(p,'a') as ds:ds.rebinned_resolution='R=30000'
        with self.assertRaisesRegex(ValueError,'resolution'):self.load(p,'sonora_flame_skimmer_v1_r3k')

    def test_flame_rejects_scalar_temperature_and_absolute_co(self):
        p=self.root/'spectra_teff_1000_grav_100_mh_+0.0_co_1.0_R3000.nc';manufactured_netcdf(p)
        with Dataset(p,'a') as ds:
            params=json.loads(ds.planet_params);params['effective_temp']=958;ds.planet_params=json.dumps(params)
        with self.assertRaisesRegex(ValueError,'quantity'):self.load(p,'sonora_flame_skimmer_v1_r3k')
        manufactured_netcdf(p)
        with Dataset(p,'a') as ds:
            params=json.loads(ds.planet_params);params['cto']['unit']='absolute';ds.planet_params=json.dumps(params)
        with self.assertRaisesRegex(ValueError,'relative'):self.load(p,'sonora_flame_skimmer_v1_r3k')

    def test_elf_owl_native_coordinate_and_declared_release(self):
        p=self.root/'spectra_logzz_4.0_teff_1000.0_grav_100.0_mh_0.0_co_1.0.nc';manufactured_netcdf(p,flame=False)
        s=self.load(p,'sonora_elf_owl_v2')
        self.assertTrue(np.all(s.surface_flux==1));self.assertEqual(s.metadata['parameters_verified_from_header']['logKzz_cgs'],4)
        self.assertIn('user-selected',s.metadata['source_release'])

    def test_diamondback_cloudfree_is_not_a_numeric_sedimentation_node(self):
        for cloud,expected in [('nc','cloud_free'),('f2',2)]:
            p=self.root/('t1000g100%s_m0.0_co1.0.spec'%cloud)
            p.write_text('wavelength(micron) flux W/m2/m\n1 1000000\n2 1000000\n3 1000000\n')
            s=self.load(p,'sonora_diamondback')
            self.assertEqual(s.metadata['parameters_verified_from_filename']['fsed'],expected)
            np.testing.assert_allclose(s.surface_flux,1)

    def test_folder_import_rolls_back_if_any_header_disagrees(self):
        a=self.root/'spectra_teff_1000_grav_100_mh_+0.0_co_1.0_R3000.nc';manufactured_netcdf(a)
        b=self.root/'spectra_teff_1000_grav_100_mh_+0.5_co_1.0_R3000.nc';manufactured_netcdf(b,m=0)
        with self.assertRaisesRegex(ValueError,'disagrees'):
            register_folder(self.root,'planet','fixture',self.root,'*.nc','sonora_flame_skimmer_v1_r3k',{}, {})
        self.assertFalse((self.root/'configs/app_catalog.json').exists())

    def test_release_metadata_selects_expected_nodes_without_native_arrays(self):
        source=json.loads((Path(__file__).parent/'fixtures/parameter_release_metadata.json').read_text())
        for kind,count in [('diamondback',8),('flame',11)]:
            self.assertEqual(len(wanted_members([r['name'] for r in source[kind]['members']],kind)),count)
        native=source['flame']['netcdf_metadata']
        p=self.root/'spectra_teff_1000_grav_100_mh_+0.0_co_1.0_R3000.nc';manufactured_netcdf(p)
        with Dataset(p,'a') as ds:
            for key,value in native['attrs'].items():ds.setncattr(key,value)
            for key in ('wavelength','flux_emission'):
                for name,value in native['variables'][key]['attrs'].items():ds.variables[key].setncattr(name,value)
        model=self.load(p,'sonora_flame_skimmer_v1_r3k')
        self.assertTrue(np.all(model.surface_flux==1))


class ControlledComparison(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.root=Path(self.directory.name)
        self.catalog=write_fixture_catalog(self.root,(1000,))
        rows=json.loads(self.catalog.read_text());planet=rows[-1];self.rows=[]
        for m,f in [(-.5,2),(0,3),(.5,4)]:
            name='metal_%g.dat'%m;np.savetxt(self.root/name,np.column_stack((np.linspace(1,5,800),np.ones(800)*f)))
            row=json.loads(json.dumps(planet));row['parameters']['metallicity_dex']=m;row['spectrum']['file']=name;self.rows.append(row)
        self.catalog.write_text(json.dumps([rows[0]]+self.rows))
        self.star=Spectrum(np.linspace(.5,6,500),np.ones(500)*100,{'label':'analytic star','wavelength_frame':'vacuum'})
    def tearDown(self):self.directory.cleanup()

    def test_shared_bins_and_native_band_ratios(self):
        bands=[{'name':'analytic response','detector':'photon','response':{'wavelength_um':[2,3,4],'throughput':[0,1,0]}}]
        result=explore(self.star,self.rows,self.root,'metallicity_dex',10,1,50,[.5,6],bands)
        self.assertEqual(result['shared_wavelength_range_um'],[1,5])
        for curve,expected,band in zip(result['curves'],[.0002,.0003,.0004],result['band_rows']):
            np.testing.assert_allclose(curve['result']['planet_star_contrast'],expected,rtol=1e-12)
            self.assertAlmostEqual(band['planet_star_contrast'],expected,places=14)
            self.assertEqual(curve['result']['wavelength_um'],result['curves'][0]['result']['wavelength_um'])
        scaled=explore(self.star,self.rows,self.root,'metallicity_dex',10,2,50)
        np.testing.assert_allclose(np.array(scaled['curves'][0]['result']['planet_star_contrast'])/result['curves'][0]['result']['planet_star_contrast'],4)
        z=zipfile.ZipFile(io.BytesIO(export_bundle(result)))
        self.assertIn('parameter_comparison.pdf',z.namelist());self.assertIn('band_contrasts.csv',z.namelist())
        self.assertEqual(json.loads(z.read('report.json'))['held_parameters']['teff_K'],1000)

    def test_rejects_mixed_gravity_format_duplicates_missing_and_partial_bands(self):
        import copy
        bad=copy.deepcopy(self.rows);bad[1]['parameters']['logg_cgs']=5
        with self.assertRaisesRegex(ValueError,'hold every other'):explore(self.star,bad,self.root,'metallicity_dex',10,1)
        bad=copy.deepcopy(self.rows);bad[1]['spectrum']['format']='sonora_bobcat'
        with self.assertRaisesRegex(ValueError,'formats'):explore(self.star,bad,self.root,'metallicity_dex',10,1)
        with self.assertRaisesRegex(ValueError,'Duplicate'):explore(self.star,[self.rows[0]]*2,self.root,'metallicity_dex',10,1)
        bad=copy.deepcopy(self.rows);bad[1]['spectrum']['file']='absent.dat'
        with self.assertRaises(FileNotFoundError):explore(self.star,bad,self.root,'metallicity_dex',10,1)
        with self.assertRaisesRegex(ValueError,'Full supplied filter coverage'):
            explore(self.star,self.rows,self.root,'metallicity_dex',10,1,bands=[{'name':'bad','detector':'energy','response':{'wavelength_um':[.5,2,3],'throughput':[0,1,0]}}])

    def test_in_app_build_export_and_stale_controls(self):
        at=load_app(self.catalog);self.assertEqual(len(at.exception),0)
        at.button(key='parameter_build').click().run();self.assertEqual(len(at.exception),0)
        self.assertEqual(len(at.session_state['parameter_result']['curves']),3)
        at.button(key='parameter_prepare').click().run();self.assertEqual(len(at.exception),0)
        at.button(key='prepare_export').click().run();self.assertEqual(len(at.exception),0)
        z=zipfile.ZipFile(io.BytesIO(at.session_state['export']['zip']))
        self.assertIn('parameter_exploration/report.json',z.namelist())
        at.checkbox(key='parameter_include_bands').uncheck().run()
        # Include-toggle does not change an empty response set. Changing values does.
        widget=[m for m in at.multiselect if str(m.key).startswith('parameter_values_')][0]
        widget.set_value(widget.value[:2]).run()
        self.assertNotIn('parameter_result',at.session_state)
        at.button(key='parameter_build').click().run();self.assertEqual(len(at.exception),0)
        self.assertEqual(len(at.session_state['parameter_result']['curves']),2)

    def test_native_flame_in_app_defaults_allow_metal_co_and_mixing(self):
        star=json.loads(self.catalog.read_text())[0];rows=[]
        for m,co,k in [(-.5,1,None),(0,1,None),(.5,1,None),(0,.5,None),(0,1.5,None),(0,2.5,None)]+[(0,1,k) for k in (2,4,7,8,9)]:
            name='spectra_%steff_1000_grav_100_mh_%+.1f_co_%.1f_R3000.nc'%('' if k is None else 'logkzz_%d_'%k,m,co)
            file=self.root/name;manufactured_netcdf(file,m=m,co=co)
            metadata=load_spectrum({'file':name,'format':'sonora_flame_skimmer_v1_r3k'},self.root).metadata
            params=dict(metadata['parameters_verified_from_filename']);params.update(metadata['parameters_verified_from_header'])
            rows.append({'role':'planet','family':'sonora_flame_skimmer_v1_r3k','parameters':params,'spectrum':{'format':'sonora_flame_skimmer_v1_r3k','file':name}})
        self.catalog.write_text(json.dumps([star]+rows));at=load_app(self.catalog)
        self.assertEqual(len(at.exception),0)
        # The sidebar's first chemistry is disequilibrium; the controlled tab
        # can still select a solar equilibrium baseline explicitly.
        at.selectbox(key='parameter_hold_sonora_flame_skimmer_v1_r3k_metallicity_dex_chemistry').set_value(json.dumps('equilibrium')).run()
        at.button(key='parameter_build').click().run();self.assertEqual(len(at.exception),0)
        self.assertEqual(len(at.session_state['parameter_result']['curves']),3)
        at.selectbox(key='parameter_axis_sonora_flame_skimmer_v1_r3k').set_value('co_relative_solar').run()
        at.selectbox(key='parameter_hold_sonora_flame_skimmer_v1_r3k_co_relative_solar_chemistry').set_value(json.dumps('equilibrium')).run()
        at.button(key='parameter_build').click().run();self.assertEqual(len(at.exception),0)
        self.assertEqual(len(at.session_state['parameter_result']['curves']),4)
        at.selectbox(key='parameter_axis_sonora_flame_skimmer_v1_r3k').set_value('logKzz_cgs').run()
        at.button(key='parameter_build').click().run();self.assertEqual(len(at.exception),0)
        self.assertEqual(len(at.session_state['parameter_result']['curves']),5)


class FakeResponse(io.BytesIO):
    def __init__(self,data,status,headers):super().__init__(data);self.status=status;self.headers=headers


class RemoteArchive(unittest.TestCase):
    def fixture(self,change=None,compression=zipfile.ZIP_DEFLATED):
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w',compression) as z:
            z.writestr('spectra/a.spec',b'a fixture spectrum\n'*10000)
            z.writestr('spectra/b.spec',b'another fixture\n')
        payload=output.getvalue();calls=[]
        def opener(request,**kwargs):
            bounds=request.get_header('Range').removeprefix('bytes=');a,b=map(int,bounds.split('-'));calls.append((a,b))
            data=payload[a:b+1];headers={'Content-Range':'bytes %d-%d/%d'%(a,b,len(payload)),'ETag':'fixture-tag'};status=206
            if change:data,status,headers=change(data,status,headers,len(calls))
            return FakeResponse(data,status,headers)
        return RangeFile('https://fixture.invalid/archive.zip',len(payload),opener),calls

    def test_range_zip_crc_cache_and_receipt(self):
        r,calls=self.fixture()
        with zipfile.ZipFile(r) as z,tempfile.TemporaryDirectory() as folder:
            source={'record':'fixture','official_archive_md5':'not-whole-file-verified'}
            selected_members(z,['spectra/a.spec'],folder,source)
            receipt=json.loads((Path(folder)/'installation_receipt.json').read_text())
            self.assertIn('Full archive MD5 was not checked',receipt['checksum_scope'])
            self.assertEqual(len(receipt['members'][0]['sha256']),64)
            count=len(calls);selected_members(z,['spectra/a.spec'],folder,source);self.assertEqual(count,len(calls))
            (Path(folder)/'a.spec').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Existing model differs'):selected_members(z,['spectra/a.spec'],folder,source)

    def test_refuses_full_response_wrong_range_encoding_and_short_body(self):
        changes=[lambda d,s,h,n:(d,200,h),lambda d,s,h,n:(d,s,dict(h,**{'Content-Range':'bytes 0-1/2'})),
                 lambda d,s,h,n:(d,s,dict(h,**{'Content-Encoding':'gzip'})),lambda d,s,h,n:(d[:-1],s,h)]
        for change in changes:
            r,_=self.fixture(change)
            with self.assertRaises((ValueError,OSError)):r.read(10)

    def test_detects_changed_remote_identity_and_oversized_read(self):
        def change(d,s,h,n):return d,s,dict(h,ETag='changed' if n>1 else 'fixture-tag')
        r,_=self.fixture(change);r.read(10)
        with self.assertRaisesRegex(ValueError,'changed'):r.read(10)
        r=RangeFile('https://fixture.invalid',50_000_000,lambda *a,**k:None)
        with self.assertRaisesRegex(ValueError,'oversized'):r.read()

    def test_path_and_member_bounds(self):
        for name in ('../bad','/bad','folder/../../bad','folder/\\bad','__MACOSX/._bad'):
            info=zipfile.ZipInfo(name);info.file_size=10
            with self.assertRaises(ValueError):validate_member(info)
        info=zipfile.ZipInfo('large.spec');info.file_size=50_000_001
        with self.assertRaises(ValueError):validate_member(info)

    def test_forced_zip64_and_corrupt_member(self):
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w') as z:
            with z.open('spectrum.spec','w',force_zip64=True) as f:f.write(b'checked body')
        with zipfile.ZipFile(io.BytesIO(output.getvalue())) as z,tempfile.TemporaryDirectory() as folder:
            selected_members(z,['spectrum.spec'],folder,{})
            self.assertEqual((Path(folder)/'spectrum.spec').read_bytes(),b'checked body')
        corrupted=output.getvalue().replace(b'checked body',b'corrupt body')
        with zipfile.ZipFile(io.BytesIO(corrupted)) as z,tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(zipfile.BadZipFile):selected_members(z,['spectrum.spec'],folder,{})
            self.assertFalse((Path(folder)/'spectrum.spec').exists())
            self.assertFalse((Path(folder)/'spectrum.spec.part').exists())

    def test_exact_installer_selection_and_missing_node_rejection(self):
        diamond=['spectra/t1000g100%s_m0.0_co1.0.spec'%c for c in ('f1','f2','f3','f4','f8','nc')]
        diamond+=['spectra/t1000g100f2_m%s_co1.0.spec'%m for m in ('-0.5','+0.5')]
        self.assertEqual(len(wanted_members(diamond,'diamondback')),8)
        with self.assertRaisesRegex(ValueError,'complete'):wanted_members(diamond[:-1],'diamondback')
        flame=['eq/mh/spectra_teff_1000_grav_100_mh_%s_co_%s_R3000.nc'%(m,c) for m,c in [('-0.5','1.0'),('+0.0','1.0'),('+0.5','1.0'),('+0.0','0.5'),('+0.0','1.5'),('+0.0','2.5')]]
        flame+=['deq/mh/spectra_logkzz_%d_teff_1000_grav_100_mh_+0.0_co_1.0_R3000.nc'%k for k in (2,4,7,8,9)]
        self.assertEqual(len(wanted_members(flame,'flame')),11)

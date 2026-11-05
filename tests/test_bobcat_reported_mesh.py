"""Observed header metadata + synthetic spectra; no native grid calculations."""
from pathlib import Path
import gzip
import io
import json
import math
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from atmosphere_compare.spectra import load_spectrum
from atmosphere_compare.grid import BandGrid,catalog_band_grid
from atmosphere_compare.validation import audit_catalog
from atmosphere_compare.evolution import G,MJUP_KG
from atmosphere_compare.compare import RADII_M
from app_fixtures import write_fixture_catalog,load_app
import setup_gravity_models as installer


INVENTORY=json.loads((Path(__file__).parent/'fixtures/bobcat_header_inventory.json').read_text())['rows']


def parameters(record):
    return {'teff_K':record['header_temperature_K'],
            'logg_cgs':math.log10(record['header_gravity_m_s2']*100),
            'metallicity_dex':record['header_metallicity_dex'],'co_relative_solar':record['header_co_relative_solar']}


def synthetic_signal(p):
    return (p['teff_K']/1000.)**2.3*np.exp(.8*(p['logg_cgs']-4))


def native_fixture_bytes(record):
    # Convert an analytic constant F_lambda to synthetic cgs F_nu samples,
    # independently of the app converter. Only the first two lines are native.
    p=parameters(record);signal=synthetic_signal(p)
    lines=[record['original_parameter_line'],record['original_units_line']]
    for wavelength in (1.,3.,5.):
        fnu=signal*(wavelength*1e-6)**2/299792458.*1e9
        lines.append('%.17g %.17g'%(wavelength,fnu))
    return gzip.compress(('\n'.join(lines)+'\n').encode())


def rows_from_inventory(records=INVENTORY):
    return [{'parameters':parameters(r),'grid_coordinates':{'nominal_teff_K':r['filename_temperature_K']}}
            for r in records]


def write_archive(root,records=INVENTORY):
    path=root/'fixture.tar.gz'
    with tarfile.open(path,'w:gz') as tar:
        for record in records:
            data=native_fixture_bytes(record);member=tarfile.TarInfo('spectra/'+record['filename'])
            member.size=len(data);tar.addfile(member,io.BytesIO(data))
    return path


class BobcatReportedMesh(unittest.TestCase):
    def test_reported_offsets_are_preserved_at_load(self):
        offsets=[r for r in INVENTORY if r['temperature_header_minus_filename_K']]
        self.assertEqual(len(offsets),5)
        with tempfile.TemporaryDirectory() as folder:
            for r in offsets:
                path=Path(folder)/r['filename'];path.write_bytes(native_fixture_bytes(r))
                spec=load_spectrum({'format':'sonora_bobcat','file':str(path)},folder)
                self.assertEqual(spec.metadata['parameters_verified_from_header']['teff_K'],r['header_temperature_K'])
                self.assertEqual(spec.metadata['grid_coordinates']['nominal_teff_K'],r['filename_temperature_K'])
                self.assertAlmostEqual(spec.metadata['parameters_verified_from_header']['logg_cgs'],math.log10(3200))
                np.testing.assert_allclose(spec.surface_flux,synthetic_signal(parameters(r)),rtol=1e-14)

    def test_large_identity_or_composition_mismatches_still_fail(self):
        template=next(r for r in INVENTORY if r['filename']=='sp_t1200g31nc_m0.0.gz')
        with tempfile.TemporaryDirectory() as folder:
            for field,value,pattern in (('header_temperature_K',1203,'temperature'),
                    ('header_gravity_m_s2',45,'gravity'),('header_metallicity_dex',.5,'metallicity'),
                    ('header_co_relative_solar',.5,'co_relative')):
                r=dict(template);r[field]=value
                p=parameters(r);vals=list(r['header_numeric_values'])
                for i,key in ((0,'header_temperature_K'),(1,'header_gravity_m_s2'),(5,'header_metallicity_dex'),(6,'header_co_relative_solar')):vals[i]=r[key]
                r['original_parameter_line']=' '.join(map(str,vals))
                path=Path(folder)/r['filename'];path.write_bytes(native_fixture_bytes(r))
                with self.subTest(field=field),self.assertRaisesRegex(ValueError,pattern):
                    load_spectrum({'format':'sonora_bobcat','file':str(path)},folder)

    def test_nominal_temperature_cannot_be_declared_as_reported_temperature(self):
        r=next(r for r in INVENTORY if r['filename']=='sp_t1200g31nc_m0.0.gz')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/r['filename'];path.write_bytes(native_fixture_bytes(r))
            with self.assertRaisesRegex(ValueError,'Declared teff_K'):
                load_spectrum({'format':'sonora_bobcat','file':str(path),'parameters':{'teff_K':1200}},folder)

    def test_all_273_installer_rows_register_without_renaming_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);archive=write_archive(root)
            # An old nominal-parameter row for the same file must be replaced,
            # while unrelated catalog content remains.
            (root/'configs').mkdir()
            old=[{'role':'planet','family':'sonora_bobcat','parameters':{'teff_K':1200,'logg_cgs':3.5},
                  'spectrum':{'file':'../models/bobcat/sp_t1200g31nc_m0.0.gz'}},
                 {'role':'star','family':'unrelated','parameters':{},'spectrum':{'file':'elsewhere'}}]
            (root/'configs/app_catalog.json').write_text(json.dumps(old))
            with patch.object(installer,'ROOT',root),patch.object(installer,'download',return_value=archive):installer.main()
            rows=json.loads((root/'configs/app_catalog.json').read_text());planets=[r for r in rows if r['role']=='planet']
            self.assertEqual(len(planets),273);self.assertIn(old[1],rows)
            row=next(r for r in planets if r['grid_coordinates']['nominal_teff_K']==1200 and r['grid_coordinates']['filename_gravity_m_s2']==31)
            self.assertEqual(row['parameters']['teff_K'],1201)
            record=json.loads((root/'models/gravity_installation.json').read_text())
            self.assertEqual(len(record['header_temperature_offsets']),5)
            signals=[synthetic_signal(row['parameters']) for row in planets]
            result,_=BandGrid(planets,signals).evaluate({'teff_K':1200.5,'logg_cgs':3.8})
            self.assertAlmostEqual(result,synthetic_signal({'teff_K':1200.5,'logg_cgs':3.8}),places=12)

    def test_failed_install_keeps_catalog_atomic(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'configs').mkdir();catalog=root/'configs/app_catalog.json';catalog.write_text('[]')
            record=dict(INVENTORY[0]);record['header_temperature_K']=1300.
            vals=list(record['header_numeric_values']);vals[0]=1300.;record['original_parameter_line']=' '.join(map(str,vals))
            archive=write_archive(root,[record]);before=catalog.read_bytes()
            with patch.object(installer,'ROOT',root),patch.object(installer,'download',return_value=archive),self.assertRaises(ValueError):installer.main()
            self.assertEqual(catalog.read_bytes(),before)

    def test_actual_coordinate_power_law_and_nonnegative_weights(self):
        rows=rows_from_inventory();grid=BandGrid(rows,[synthetic_signal(r['parameters']) for r in rows])
        cases=[(1201,math.log10(3200)),(1200,math.log10(3200)),(1200.5,3.8),(2300.3,3.6),(2400.4,3.55),(1000,5.)]
        for t,g in cases:
            with self.subTest(t=t,g=g):
                result,corners=grid.evaluate({'teff_K':t,'logg_cgs':g})
                self.assertAlmostEqual(result,synthetic_signal({'teff_K':t,'logg_cgs':g}),places=12)
                self.assertAlmostEqual(sum(c['weight'] for c in corners),1.,places=14)
                self.assertTrue(all(c['weight']>=0 for c in corners))
                self.assertAlmostEqual(sum(c['weight']*np.log(c['parameters']['teff_K']) for c in corners),np.log(t),places=13)
                self.assertAlmostEqual(sum(c['weight']*c['parameters']['logg_cgs'] for c in corners),g,places=13)

    def test_native_points_recovered_exactly_for_all_reported_headers(self):
        rows=rows_from_inventory();signals=[synthetic_signal(r['parameters']) for r in rows];grid=BandGrid(rows,signals)
        for row,signal in zip(rows,signals):
            actual,corners=grid.evaluate(row['parameters'])
            self.assertAlmostEqual(actual,signal,places=12);self.assertEqual(len(corners),1)

    def test_no_missing_cell_bridge_or_boundary_extrapolation(self):
        rows=rows_from_inventory();rows=[r for r in rows if not (r['parameters']['teff_K']==1200 and r['parameters']['logg_cgs']==4.)]
        grid=BandGrid(rows,[synthetic_signal(r['parameters']) for r in rows])
        with self.assertRaisesRegex(ValueError,'incomplete'):grid.evaluate({'teff_K':1150,'logg_cgs':3.9})
        for t,g in ((2400.2,4.),(199.9,4.),(1000,5.01)):
            with self.subTest(t=t,g=g),self.assertRaisesRegex(ValueError,'outside'):grid.evaluate({'teff_K':t,'logg_cgs':g})

    def test_native_temperature_list_is_specific_to_selected_gravity(self):
        rows=rows_from_inventory();grid=BandGrid(rows,[1.]*len(rows))
        self.assertIn(1200,grid.temperatures_at_gravity(4.));self.assertNotIn(1201,grid.temperatures_at_gravity(4.))
        self.assertIn(1201,grid.temperatures_at_gravity(math.log10(3200)));self.assertNotIn(1200,grid.temperatures_at_gravity(math.log10(3200)))
        self.assertEqual(len(grid.temperatures_at_gravity(4.)),39)

    def test_regular_grid_limit_matches_previous_interpolation(self):
        tagged=[r for r in rows_from_inventory() if r['parameters']['logg_cgs'] in (4.,5.)]
        plain=[{'parameters':r['parameters']} for r in tagged];signals=[synthetic_signal(r['parameters']) for r in tagged]
        a,b=BandGrid(tagged,signals),BandGrid(plain,signals)
        for t in np.geomspace(200,2400,17):
            self.assertAlmostEqual(a.evaluate({'teff_K':t,'logg_cgs':4.37})[0],b.evaluate({'teff_K':t,'logg_cgs':4.37})[0],places=12)

    def test_full_reported_mesh_gravity_holdouts_use_same_interpolator(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'models').mkdir();rows=[]
            for record in INVENTORY:
                file=root/'models'/record['filename'];file.write_bytes(native_fixture_bytes(record))
                rows.append({'role':'planet','family':'sonora_bobcat','parameters':parameters(record),
                             'spectrum':{'format':'sonora_bobcat','file':str(file)}})
            catalog=root/'catalog.json';catalog.write_text(json.dumps(rows))
            band={'name':'analytic','response':{'wavelength_um':[1,3,5],'throughput':[1,1,1]},'detector':'energy'}
            report=audit_catalog(catalog,[band])
            self.assertEqual(len(report['records']),273);self.assertFalse(report['load_errors'])
            self.assertEqual(len(report['heldout_temperature_band_tests']),259)
            self.assertEqual(len(report['heldout_gravity_band_tests']),195)
            self.assertFalse(report['unsupported_gravity_holdouts'])
            for row in report['heldout_temperature_band_tests']+report['heldout_gravity_band_tests']:
                self.assertLess(abs(row['fractional_error']),1e-12)

    def test_unverified_or_contradictory_mesh_tags_rejected(self):
        rows=rows_from_inventory()[:2];rows[0]['grid_coordinates']['nominal_teff_K']+=4
        with self.assertRaisesRegex(ValueError,'coordinates'):BandGrid(rows,[1,1])
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);record=INVENTORY[0];file=root/record['filename'];file.write_bytes(native_fixture_bytes(record))
            row={'role':'planet','family':'sonora_bobcat','parameters':parameters(record),'spectrum':{'format':'sonora_bobcat','file':str(file)},
                 'grid_coordinates':{'nominal_teff_K':800}}
            path=root/'catalog.json';path.write_text(json.dumps([row]))
            meta={'catalog_path':str(path),'family':'sonora_bobcat','parameters_declared':row['parameters'],'wavelength_frame':'vacuum'}
            with self.assertRaisesRegex(ValueError,'connectivity'):catalog_band_grid(meta,[1,3,5],[1,1,1],'energy')


class BobcatMeshInterface(unittest.TestCase):
    def test_reported_mesh_in_evolution_and_measurement_app_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=write_fixture_catalog(root,(1100,))
            rows=[r for r in json.loads(path.read_text()) if r['role']=='star']
            for record in INVENTORY:
                if record['filename_temperature_K'] not in (1100,1200,1300) or record['filename_gravity_m_s2'] not in (31,100,1000):continue
                file=root/record['filename'];file.write_bytes(native_fixture_bytes(record))
                rows.append({'role':'planet','family':'sonora_bobcat','parameters':parameters(record),'spectrum':{'format':'sonora_bobcat','file':str(file),'label':'Synthetic Bobcat fixture'}})
            path.write_text(json.dumps(rows));at=load_app(path)
            at.selectbox(key='planet_param_logg_cgs').set_value(json.dumps(4.)).run()
            at.selectbox(key='planet_param_teff_K').set_value(json.dumps(1200.)).run();at.button(key='run').click().run()
            at.radio(key='filter_source').set_value('Idealized top-hat (method check)').run()
            at.button(key='calculate_filter').click().run();at.button(key='inference_build').click().run()
            self.assertEqual(len(at.exception),0)

            tested={r['temperature_K'] for r in at.session_state['inference_result']['rows']}
            self.assertEqual(tested,{1100.,1200.,1300.})
            gravity=3.8;mass=(10**gravity/100)*RADII_M['R_jup']**2/(G*MJUP_KG)
            evolution=root/'evolution.csv';evolution.write_text('mass_Mjup,age_Myr,intrinsic_temperature_K,radius_Rjup\n%.17g,100,1200.5,1\n'%mass)
            at.text_input(key='evolution_path').set_value(str(evolution)).run();at.button(key='evolution_build').click().run()
            at.selectbox(key='sep_band').set_value('Calculated filter').run();at.checkbox(key='sep_evolution').check().run()
            at.button(key='sep_build').click().run();self.assertEqual(len(at.exception),0)
            result=at.session_state['separation_result'];self.assertEqual(result['models'][0]['omitted_out_of_grid_samples'],0)
            at.button(key='sep_check').click().run();at.button(key='prepare_export').click().run()
            self.assertEqual(len(at.exception),0)


if __name__=='__main__':unittest.main()

"""Synthetic property-to-model controls. No real native atmosphere calculation."""
from pathlib import Path
import copy
import json
import shutil
import tempfile
import unittest
import numpy as np
from atmosphere_compare.properties import ModelLibrary, predict_from_properties, predict_request, predict_many, predict_separation, radius_at_contrast, response_from_request
from atmosphere_compare.bandpass import Response

ROOT = Path(__file__).resolve().parents[1]


def response_mapping():
    return {'kind': 'filter_and_sky_csv', 'name': 'Synthetic unity response (method demonstration only)', 'limits_um': [9, 14], 'filter': {'file': 'templates/demo_filter.csv', 'wavelength_unit': 'um', 'transmission_unit': 'fraction', 'wavelength_frame': 'vacuum'}, 'sky': {'file': 'templates/demo_sky.csv', 'wavelength_unit': 'um', 'transmission_unit': 'fraction', 'wavelength_frame': 'vacuum'}}


def write_library(root):
    root = Path(root);(root/'configs').mkdir();(root/'models').mkdir();(root/'assets').mkdir()
    (root/'templates').mkdir()
    for key in ('filter', 'sky'):
        (root/'templates'/('demo_'+key+'.csv')).write_text('wavelength,transmission\n9,1\n11,1\n14,1\n')
    (root/'templates/response_demo.json').write_text(json.dumps(response_mapping()))
    rows = []
    def add(role, family, params, values, suffix=''):
        name=role+'_'+family+'_'+str(len(rows))+suffix+'.txt'
        np.savetxt(root/'models'/name, np.column_stack(([8., 11., 15.], values)))
        rows.append({'role': role, 'family': family, 'parameters': params,
            'spectrum': {'format': 'ascii', 'file': '../models/'+name, 'label': 'SYNTHETIC '+name,
                         'wavelength_unit': 'um', 'flux_unit': 'W_m2_um', 'flux_convention': 'surface_flux'}})
    add('star', 'star_grid', {'teff_K': 5000., 'logg_cgs': 4.5, 'metallicity_dex': 0.}, [10, 10, 10])
    add('star', 'other_star', {'teff_K': 5000., 'logg_cgs': 4.5, 'metallicity_dex': 0.}, [20, 20, 20])
    for temp in (1000., 2000.):
        for gravity in (4., 5.):
            add('planet', 'planet_grid', {'teff_K': temp, 'logg_cgs': gravity, 'metallicity_dex': 0., 'cloud': 'clear'}, [temp/1000+gravity-4]*3)
    add('planet', 'cloud_grid', {'teff_K': 1000., 'logg_cgs': 4., 'cloud': 'clear', 'co_relative_solar': 1.}, [1, 1, 1])
    add('planet', 'cloud_grid', {'teff_K': 1000., 'logg_cgs': 4., 'cloud': 'cloudy', 'co_relative_solar': 1.}, [2, 2, 2])
    catalog=root/'configs/app_catalog.json';catalog.write_text(json.dumps(rows))
    return catalog,rows


def bodies():
    star={'family': 'star_grid', 'parameters': {'teff_K': 5000., 'logg_cgs': 4.5, 'metallicity_dex': 0.}, 'radius': {'value': 10., 'unit': 'm'}}
    planet={'family': 'planet_grid', 'parameters': {'teff_K': 1000., 'logg_cgs': 4., 'metallicity_dex': 0., 'cloud': 'clear'}, 'radius': {'value': 1., 'unit': 'm'}}
    return star,planet


class PropertyPredictionChecks(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.root=Path(self.directory.name)
        self.catalog,self.rows=write_library(self.root);self.star,self.planet=bodies()
        self.response=response_from_request(response_mapping(),self.root)

    def tearDown(self):self.directory.cleanup()

    def predict(self, star=None, planet=None, **options):
        return predict_from_properties(self.catalog, star or self.star, planet or self.planet, self.response, **options)

    def test_exact_parameters_family_switch_radius_distance_and_source_provenance(self):
        one=self.predict(distance_pc=10.)
        self.assertAlmostEqual(one['planet_star_contrast'], .001)
        other=copy.deepcopy(self.star);other['family']='other_star'
        two=self.predict(star=other,distance_pc=20.)
        self.assertAlmostEqual(two['planet_star_contrast'], .0005)
        larger=copy.deepcopy(self.planet);larger['radius']['value']=2.
        self.assertAlmostEqual(self.predict(planet=larger)['planet_star_contrast'], .004)
        self.assertEqual(one['source_spectra']['star']['family'],'star_grid')
        self.assertEqual(one['selection_provenance']['catalog_kind'],'catalog_file')
        self.assertEqual(one['source_spectra']['star']['parameters_requested'],self.star['parameters'])

    def test_unsupported_parameters_no_nearest_neighbor_and_missing_files(self):
        for field,value in [('teff_K',1200.),('unknown',1.),('teff_K',True)]:
            body=copy.deepcopy(self.planet);body['parameters'][field]=value
            with self.assertRaises(ValueError):self.predict(planet=body)
        row=self.rows[2];(self.catalog.parent/row['spectrum']['file']).unlink()
        with self.assertRaises(FileNotFoundError):self.predict()

    def test_categorical_parameters_are_explicit_and_not_interpolated(self):
        body=copy.deepcopy(self.planet);body['family']='cloud_grid';body['parameters']={'teff_K':1000.,'logg_cgs':4.}
        with self.assertRaisesRegex(ValueError,'ambiguous'):self.predict(planet=body)
        body['parameters'].update(cloud='cloudy',co_relative_solar=1.)
        self.assertAlmostEqual(self.predict(planet=body)['planet_star_contrast'],.002)
        body['parameters']['co_relative_solar']=.5
        with self.assertRaises(ValueError):self.predict(planet=body)

    def test_complete_cell_surface_flux_interpolation(self):
        body=copy.deepcopy(self.planet);body['parameters'].update(teff_K=np.sqrt(1000.*2000.),logg_cgs=4.5);body['selection']='linear_surface_flux'
        result=self.predict(planet=body)
        # Equal log-temperature and linear-gravity weights give flux (1+2+2+3)/4=2.
        self.assertAlmostEqual(result['planet_star_contrast'],.002)
        source=result['source_spectra']['planet']
        self.assertEqual(source['construction'],'linear_surface_flux_interpolation')
        self.assertEqual(len(source['interpolation']['corners']),4)
        self.assertIsNone(result['native_samples_inside_band']['planet'])
        self.assertFalse(result['spectral_weights']['integration_uses_native_samples'])
        self.assertIn('not_computed',result['energy_diagnostics']['planet'])

    def test_narrow_band_with_no_interior_native_knot_keeps_linear_flux(self):
        body=copy.deepcopy(self.planet);body['parameters'].update(teff_K=np.sqrt(1000.*2000.),logg_cgs=4.5);body['selection']='linear_surface_flux'
        response=Response(np.array([12.,12.5,13.]),np.ones(3),np.array([12.,12.5,13.]),np.ones(3),(12.,13.),{'name':'Synthetic narrow band','wavelength_frame':'vacuum'})
        result=predict_from_properties(self.catalog,self.star,body,response)
        self.assertAlmostEqual(result['planet_star_contrast'],.002)
        self.assertTrue(result['source_spectra']['planet']['interpolation']['redundant_midpoint_inserted'])
        self.assertIsNone(result['native_samples_inside_band']['planet'])

    def test_interpolation_refuses_missing_corner_and_extrapolation(self):
        body=copy.deepcopy(self.planet);body['selection']='linear_surface_flux';body['parameters'].update(teff_K=1500.,logg_cgs=4.5)
        self.catalog.write_text(json.dumps([r for r in self.rows if not (r['family']=='planet_grid' and r['parameters']['teff_K']==2000 and r['parameters']['logg_cgs']==5.)]))
        with self.assertRaisesRegex(ValueError,'incomplete'):self.predict(planet=body)
        self.catalog.write_text(json.dumps(self.rows));body['parameters']['teff_K']=2100
        with self.assertRaisesRegex(ValueError,'outside'):self.predict(planet=body)

    def test_custom_filter_sky_units_and_no_extrapolation(self):
        (self.root/'filter.csv').write_text('wavelength,transmission\n9000,100\n11000,100\n14000,100\n')
        (self.root/'sky.csv').write_text('wavelength,transmission\n80000,1\n110000,1\n150000,1\n')
        mapping={'kind':'filter_and_sky_csv','name':'Synthetic unity response','limits_um':[9,14],
            'filter':{'file':'filter.csv','wavelength_unit':'nm','transmission_unit':'percent','wavelength_frame':'vacuum'},
            'sky':{'file':'sky.csv','wavelength_unit':'angstrom','transmission_unit':'fraction','wavelength_frame':'vacuum'}}
        response=response_from_request(mapping,self.root)
        np.testing.assert_allclose(response.at([9,11,14]),1.)
        result=predict_from_properties(self.catalog,self.star,self.planet,response)
        self.assertAlmostEqual(result['planet_star_contrast'],.001)
        mapping['limits_um']=[7,14]
        with self.assertRaises(ValueError):response_from_request(mapping,self.root)
        mapping['limits_um']=[9,14];mapping['sky']['wavelength_frame']='air'
        with self.assertRaises(ValueError):response_from_request(mapping,self.root)

    def test_json_request_and_direct_inputs_compute_new_properties(self):
        request={'catalog':'configs/app_catalog.json','star':self.star,'companion':self.planet,
                 'response':response_mapping(),'distance_pc':10.}
        one=predict_request(request,self.root)
        two=predict_from_properties(self.catalog,self.star,self.planet,self.response,distance_pc=10.)
        self.assertEqual(one['planet_star_contrast'],two['planet_star_contrast'])
        request['star']['family']='other_star'
        self.assertAlmostEqual(predict_request(request,self.root)['planet_star_contrast'],one['planet_star_contrast']/2)
        request['unexpected']=3
        with self.assertRaises(ValueError):predict_request(request,self.root)

    def test_fixed_and_heated_separation_and_conditional_radius(self):
        star=copy.deepcopy(self.star);star['radius']={'value':1.,'unit':'R_sun'}
        body=copy.deepcopy(self.planet);body['selection']='linear_surface_flux'
        one=predict_separation(self.catalog,star,body,self.response,[.01,.02],10.)
        self.assertEqual(one['contrast'][0],one['contrast'][1])
        two=predict_separation(self.catalog,star,body,self.response,[.01,.02],10.,
            heating_options={'intrinsic_teff_K':1000.,'bond_albedo':0.,'true_to_projected':1.})
        self.assertGreater(two['effective_temperature_K'][0],two['effective_temperature_K'][1])
        self.assertGreater(two['contrast'][0],two['contrast'][1])
        self.assertEqual(two['projected_separation_au'],[.1,.2])
        prediction=self.predict();limit=radius_at_contrast(prediction,prediction['planet_star_contrast']*4)
        self.assertAlmostEqual(limit['radius_m'],2.)

    def test_model_comparison_uses_each_explicit_family_and_parameters(self):
        a=copy.deepcopy(self.planet);b=copy.deepcopy(a);b['parameters']['teff_K']=2000
        result=predict_many(self.catalog,self.star,[a,b],self.response)
        self.assertAlmostEqual(result[1]['planet_star_contrast']/result[0]['planet_star_contrast'],2.)
        inventory=ModelLibrary(self.catalog).inventory()
        self.assertEqual(sum(r['registered_models'] for r in inventory),len(self.rows))
        self.assertEqual(sum(r['installed_models'] for r in inventory),len(self.rows))


if __name__=='__main__':unittest.main()

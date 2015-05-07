'''
Created on 7 May 2015

@author: ppeglar
'''
from iris import tests

import math

import numpy as np

import spherical_geometry.vectorised_spherical as sphz


def spt(*args, **kwargs):
    # Wrap sph.sph_point, but default to in_degrees=True
    in_degrees = kwargs.pop('in_degrees', True)
    kwargs['in_degrees'] = in_degrees
    return sphz.sph_pointz(*args, **kwargs)


def spts(latlon_list):
    # Convert a list of latlons (in degrees) to SphPoints
    return sphz.sph_pointz(latlon_list, in_degrees=True)

# def spoly(points):
#     # Create a polygon from list of latlons (in degrees)
#     return sph.SphAcwConvexPolygon(points, in_degrees=True)

def d2r(degrees):
    return degrees / 180.0 * math.pi


def r2d(radians):
    return radians * 180.0 / math.pi


class TestConverts(tests.IrisTest):
    def test_convert_xyz_to_latlon(self):
        def _test_xyz_latlon(test_xyz, expect_latlon):
            test_xyz = np.array(test_xyz)
            result_latlon = sphz.convert_xyzs_to_latlons(test_xyz)
            self.assertArrayAllClose(result_latlon, expect_latlon)

        # Single-point testcases, borrowed from the old scalar version.
        _test_xyz_latlon((0, 0, 1), (d2r(90), 0.0))
        _test_xyz_latlon((0, 0, 0.01), (d2r(90), 0.0))
        _test_xyz_latlon((1, 0, 1.0), (d2r(45), 0.0))
        _test_xyz_latlon((-1, 0, 1.0), (d2r(45), d2r(180)))
        _test_xyz_latlon((1.0, 0, 0), (0.0, 0.0))
        _test_xyz_latlon((1.0, 1.0, 0), (0.0, d2r(45)))
        _test_xyz_latlon((1.0, -1.0, 0), (0.0, -d2r(45)))
        _test_xyz_latlon((-1.0, -1.0, 0), (0.0, -d2r(180 - 45)))
        _test_xyz_latlon((0.0, -1.0, 0), (0.0, -d2r(90)))
        _test_xyz_latlon((1, 1, -math.sqrt(2)), (-d2r(45), d2r(45)))
        _test_xyz_latlon((math.sqrt(1.5), math.sqrt(0.5), math.sqrt(2.0)),
                         (d2r(45), d2r(30)))

        # Single-point testcase for shapes
        xyzs = np.array([0.0, 0.0, 1.0])
        expect_latlons = np.array([d2r(90), 0.0])
        expect_lats = expect_latlons[..., 0]
        expect_lons = expect_latlons[..., 1]
        self.assertEqual(xyzs.shape, (3,))
        lats, lons = sphz.convert_xyzs_to_latlons(xyzs)
        self.assertEqual(lats.shape, ())
        self.assertArrayAllClose(lats, expect_lats)
        self.assertArrayAllClose(lons, expect_lons)

        # Multi-point testcase
        xyzs = np.array([(0.0, 0.0, 1.0),
                         (1.0, 0.0, 1.0),
                         (-1.0, 0.0, 1.0)])
        expect_latlons = np.array([(d2r(90), 0.0),
                                   (d2r(45), 0.0),
                                   (d2r(45), d2r(180))])
        expect_lats = expect_latlons[..., 0]
        expect_lons = expect_latlons[..., 1]
        self.assertEqual(xyzs.shape, (3, 3))
        lats, lons = sphz.convert_xyzs_to_latlons(xyzs)
        self.assertEqual(lats.shape, (3,))
        self.assertArrayAllClose(lats, expect_lats)
        self.assertArrayAllClose(lons, expect_lons)

        # Multi-dimensional testcase
        xyzs = np.array([[(0.0, 0.0, 1.0),
                          (1.0, 0.0, 1.0),
                          (-1.0, 0.0, 1.0),
                          (0.0, 0.0, 1.0)],
                         [(1.0, 0.0, 1.0),
                          (-1.0, 0.0, 1.0),
                          (0.0, 0.0, 1.0),
                          (1.0, 0.0, 1.0)]])
        expect_latlons = np.array([[(d2r(90), 0.0),
                                    (d2r(45), 0.0),
                                    (d2r(45), d2r(180)),
                                    (d2r(90), 0.0)],
                                   [(d2r(45), 0.0),
                                    (d2r(45), d2r(180)),
                                    (d2r(90), 0.0),
                                    (d2r(45), 0.0)]])
        expect_lats = expect_latlons[..., 0]
        expect_lons = expect_latlons[..., 1]
        self.assertEqual(xyzs.shape, (2, 4, 3))
        lats, lons = sphz.convert_xyzs_to_latlons(xyzs)
        self.assertEqual(lats.shape, (2, 4))
        self.assertArrayAllClose(lats, expect_lats)
        self.assertArrayAllClose(lons, expect_lons)


    def test_convert_latlon_to_xyz(self):
        def _test_latlon_xyz(test_latlon, expect_xyz):
            test_latlon = [np.array(vals) for vals in test_latlon]
            result_xyz = sphz.convert_latlons_to_xyzs(*test_latlon)
            self.assertArrayAllClose(result_xyz, expect_xyz, atol=1e-7)

        # Single-point testcases, borrowed from the old scalar version.
        rv2 = 1.0 / math.sqrt(2)
        _test_latlon_xyz((d2r(90), 0.0), (0, 0, 1.0))
        _test_latlon_xyz((d2r(45), 0.0), (rv2, 0, rv2))
        _test_latlon_xyz((d2r(45), d2r(180)), (-rv2, 0, rv2))
        _test_latlon_xyz((0.0, 0.0), (1.0, 0, 0))
        _test_latlon_xyz((0.0, d2r(45)), (rv2, rv2, 0))
        _test_latlon_xyz((0.0, -d2r(45)), (rv2, -rv2, 0))
        _test_latlon_xyz((0.0, -d2r(180 - 45)), (-rv2, -rv2, 0))
        _test_latlon_xyz((0.0, -d2r(90)), (0.0, -1.0, 0))
        _test_latlon_xyz((-d2r(45), d2r(45)), (0.5, 0.5, -rv2))


if __name__ == '__main__':
    tests.main()

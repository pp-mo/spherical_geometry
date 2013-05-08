'''
Created on May 8, 2013

@author: itpp
'''

from iris import tests

import spherical_geometry as sph

class TestSphGeom(tests.IrisTest):
    def test_convert_xyz_to_latlon(self):
        self.assertEqual(sph.convert_xyz_to_latlon(0, 0, 1), (math.pi/2, 0.0))

if __name__ == '__main__':
    tests.main()


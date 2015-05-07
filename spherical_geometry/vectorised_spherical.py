'''
Created on 7 May 2015

@author: ppeglar
'''
import numpy as np
 
from spherical_geometry.vectorised_array_objects import VectorisedArrayObject

# Control validity checking (turn off for speed)
ENABLE_CHECKING = True

POINT_ZERO_MAGNITUDE = 1e-15
ANGLE_ZERO_MAGNITUDE = 1e-8
COS_ANGLE_ZERO_MAGNITUDE = 1e-8


class ZeroPointLatlonError(ValueError):
    def __init__(self, *args, **kwargs):
        if not args:
            args = ['Point too close to zero for lat-lon conversion.']
        super(ZeroPointLatlonError, self).__init__(*args, **kwargs)


def convert_latlons_to_xyzs(lats, lons):
    if ENABLE_CHECKING:
        assert lons.shape == lats.shape
    shape = lats.shape
    zs = np.sin(lats)
    cos_lats = np.cos(lats)
    xs = cos_lats * np.cos(lons)
    ys = cos_lats * np.sin(lons)
    return np.concatenate([aa[..., None] for aa in (xs, ys, zs)],
                          axis=len(shape))


def convert_xyzs_to_latlons(xyzs, error_any_zeros=False):
    """
    Convert arrays of XYZ to LAT,LON.

    Input contains X, Y, Z values in shape [..., 3]

    Returns 2 arrays (lats, lons).
    Underflows are avoided, but return values are not specified.

    """
    if ENABLE_CHECKING:
        assert xyzs.shape[-1] == 3
    xs = xyzs[..., 0:1]  # Trailing dim here ensures these are always arrays.
    ys = xyzs[..., 1:2]
    zs = xyzs[..., 2:3]
    mod_sqs = xs * xs + ys * ys + zs * zs
    zero_points = np.nonzero(np.abs(mod_sqs) < POINT_ZERO_MAGNITUDE)
    if error_any_zeros and np.any(zero_points):
        raise ZeroPointLatlonError()
    # Fake zero points to avoid overflow warnings
    # N.B. this is why we added a dimension -- so we always have an array here.
    mod_sqs[zero_points] = POINT_ZERO_MAGNITUDE
    # Calculate angles by inverse trig.
    lats = np.arcsin(zs / np.sqrt(mod_sqs))
    lons = np.arctan2(ys, xs)
    if ENABLE_CHECKING:
        assert lats.shape[-1] == 1
    return (lats[..., 0], lons[..., 0])


class PointZ(VectorisedArrayObject):
    array_names = ('lats', 'lons', '_xyzs')

    def __init__(self, lats=None, lons=None, xyzs=None, in_degrees=False,
                 shape=None, arrays=None):
        """
        Initialise PointsZ.

        TODO: full description

        """
        # If passed raw 'arrays', as per parent constructor, we don't do the
        # points setting:  This is new object creation for copy, slicing etc.
        if arrays is None:
            # No arrays passed : points must be specified as lat/lon or xyz.
            arrays = {}
            if ENABLE_CHECKING:
                msg = 'either "xyzs", or both "lats and "lons" must be given'
                if ((lats is not None and xyzs is not None) or
                    (lats is not None and lons is None) or
                    (lats is None and lons is not None) or
                    (lats is None and lons is None and xyzs is None)):
                        raise ValueError(msg)
            if lats is not None:
                lats = np.asanyarray(lats)
                lons = np.asanyarray(lons)
                if in_degrees:
                    lats = np.deg2rad(lats)
                    lons = np.deg2rad(lons)
            else:
                # Note "xyzs" are optional - only calculated if needed.
                xyzs = np.asanyarray(xyzs)
                # Note: convert xyz to latlon + back, to normalise values.
                lats, lons = convert_xyzs_to_latlons(xyzs)
                xyzs = convert_latlons_to_xyzs(lats, lons)
                arrays['_xyzs'] = xyzs
            arrays['lats'], arrays['lons'] = lats, lons
            if shape is None:
                shape = lats.shape
        # Call main init.
        super(PointZ, self).__init__(arrays=arrays, shape=shape)

    @property
    def xyzs(self):
        if self._xyzs is None:
            self._xyzs = convert_latlons_to_xyzs(self.lats, self.lons)
        return self._xyzs

    def antipodes(self):
        return PointZ(xyzs=-self.xyzs)

    def __eq__(self, others):
        # Note: array-style : result is boolean array
        return np.prod(np.abs(self.xyzs - others.xyzs) < POINT_ZERO_MAGNITUDE,
                       axis=-1) == 1

    def __ne__(self, others):
        return ~self.__eq__(others)

    def dot_products(self, other):
        results = self.xyzs.dot(other.xyzs)
        # Clip to valid range, so small errors don't break inverse-trig usage
        results = np.max((-1.0, np.min((1.0, results))))
        return results

    def cross_products(self, other):
        axyz = self.xyzs
        bxyz = other.xyzs
        # Note: retain last *1 dimension, to ensure all values are arrays.
        ax, ay, az = axyz[..., 0:1], axyz[..., 1:2], axyz[..., 2:3]
        bx, by, bz = bxyz[..., 0:1], bxyz[..., 1:2], bxyz[..., 2:3]
        x, y, z = ((ay * bz - az * by),
                   (az * bx - ax * bz),
                   (ax * by - ay * bx))
        xyzs = np.concatenate((x, y, z), axis=len(x.shape)-1)
        return PointZ(xyzs=xyzs)

    def distances_to(self, other):
        return np.arccos(self.dot_products(other))

#    def __str__(self):
#        def r2d(radians):
#            return radians * 180.0 / math.pi
#        return 'SphPoint({})'.format(self._ll_str())
#
#    def __repr__(self):
#        return '{}({!r}, {!r})'.format(self.__class__.__name__,
#                                       self.lat, self.lon)
#
#    def _ll_str(self):
#        def r2d(radians):
#            return radians * 180.0 / math.pi
#        return '({:+06.1f}d, {:+06.1f}d)'.format(r2d(self.lat), r2d(self.lon))


def sph_pointz(points_or_latlons, in_degrees=False):
    """Make (lats,lons) into PointZ, or return a PointZ unchanged."""
    if hasattr(points_or_latlons, 'lats'):
        result = points_or_latlons
    else:
        if len(points_or_latlons) != 2:
            raise ValueError('sph_points argument not a PointZ. '
                             'Expected (lats, lons), got : {!r}'.format(
                                 points_or_latlons))
        result = PointZ(latlons=points_or_latlons, in_degrees=in_degrees)
    return result




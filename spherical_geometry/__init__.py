'''
Created on May 7, 2013

@author: itpp

Spherical gridcell area-overlap calculations.
'''

import itertools
import math
import numpy as np


def convert_latlon_to_xyz(lat, lon):
    z = math.sin(lat)
    cos_lat = math.cos(lat)
    x = cos_lat * math.cos(lon)
    y = cos_lat * math.sin(lon)
    return (x, y, z)


class ZeroPointLatlonError(ValueError):
    def __init__(self, *args, **kwargs):
        if not args:
            args = ['Point too close to zero for lat-lon conversion.']
        super(ZeroPointLatlonError, self).__init__(*args, **kwargs)

POINT_ZERO_MAGNITUDE = 1e-15
ANGLE_ZERO_MAGNITUDE = 1e-8
COS_ANGLE_ZERO_MAGNITUDE = 1e-8

def convert_xyz_to_latlon(x, y, z):
    mod_sq = (x * x + y * y + z * z)
    if abs(mod_sq) < POINT_ZERO_MAGNITUDE:
        raise ZeroPointLatlonError()
    lat = math.asin(z / math.sqrt(mod_sq))
    lon = math.atan2(y, x)
    return (lat, lon)


class SphPoint(object):
    """ A 2d point on the unit sphere. """
    def __init__(self, lat_or_point, lon=None):
        if lon is None:
            lat, lon = lat_or_point.lat, lat_or_point.lon
        else:
            lat = lat_or_point
        self.lat = lat
        self.lon = lon
        self._p3d = None

    def as_xyz(self):
        if self._p3d is None:
            self._p3d = convert_latlon_to_xyz(self.lat, self.lon)
        return self._p3d

    def as_latlon(self):
        return (self.lat, self.lon)

    def copy(self):
        return SphPoint(self)

    def antipode(self):
        x, y, z = self.as_xyz()
        return SphPoint(*convert_xyz_to_latlon(-x, -y, -z))

    def __eq__(self, other):
        return np.allclose(self.as_xyz(), other.as_xyz(),
                           atol=POINT_ZERO_MAGNITUDE)

    def __ne__(self, other):
        return not self.__eq__(other)

    def dot_product(self, other):
        other = sph_point(other)
        result = sum([a * b for a, b in zip(self.as_xyz(), other.as_xyz())])
        # Clip to valid range, so small errors don't break inverse-trig usage
        return max(-1.0, min(1.0, result))

    def cross_product(self, other):
        ax, ay, az = self.as_xyz()
        bx, by, bz = sph_point(other).as_xyz()
        x, y, z = ((ay * bz - az * by),
                   (az * bx - ax * bz),
                   (ax * by - ay * bx))
        return sph_point(convert_xyz_to_latlon(x, y, z))

    def distance_to(self, other):
        return math.acos(self.dot_product(other))

    def __str__(self):
        def r2d(radians):
            return radians * 180.0 / math.pi
        return 'SphPoint({})'.format(self._ll_str())

    def __repr__(self):
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self.lat, self.lon)

    def _ll_str(self):
        def r2d(radians):
            return radians * 180.0 / math.pi
        return '({:+06.1f}d, {:+06.1f}d)'.format(r2d(self.lat), r2d(self.lon))


def sph_point(point, in_degrees=False):
    """ Make (lat,lon) into SphPoint, or return SphPoint unchanged. """
    if hasattr(point, 'lat'):
        return point
    if len(point) != 2:
        raise ValueError('sph_point argument not a SphPoint. '
                         'Expected (lat, lon), got : {!r}'.format(point))
    if in_degrees:
        point = [x * math.pi/180.0 for x in point]
    return SphPoint(*point)


class SphGcSeg(object):
    """ A great circle line segment, from 'A' to 'B' on the unit sphere. """
    def __init__(self, point_a, point_b):
        self.point_a = sph_point(point_a)
        self.point_b = sph_point(point_b)
        self.pole = self.point_b.cross_product(self.point_a)

    def reverse(self):
        return SphGcSeg(self.point_b, self.point_a)

    def has_point_on_left_side(self, point):
        """
        Returns >0 (left), <1 (right) or =0.0 (close to the line).

        'COS_ANGLE_ZERO_MAGNITUDE' defines a tolerance zone near the line
        (i.e. 'nearly colinear'), where 0.0 is always returned.
        So the caller can use, for example. '>' or '>=' as required, which will
        automatically ignore 'small' values of the wrong sign.

        """
        dot = self.pole.dot_product(point)
        if abs(dot) < COS_ANGLE_ZERO_MAGNITUDE:
            return 0.0
        return -dot

    def _cos_angle_to_other(self, other):
        # Cosine of angle between self + other
        return self.pole.dot_product(other.pole)

    def _cos_angle_to_point(self, point):
        # cosine(angle from AB to AP), where P = given point
        if point == self.point_a:
            return 1.0
        seg2 = SphGcSeg(self.point_a, point)
        result = self._cos_angle_to_other(seg2)
        return result

    def angle_to_other(self, other):
        # Angle between self and another segment
        # NOTE: at present relies on dot product, so cannot resolve -ve angles.
        # As these are segments, not just GCs, this could be fixed.
        # The existing is sufficient for the area calculations, which is all
        # it's currently used for.
        return math.acos(self._cos_angle_to_other(other))

    def angle_to_point(self, point):
        # Angle from AB to AP
        result = math.acos(self._cos_angle_to_point(point))
        if abs(result) > ANGLE_ZERO_MAGNITUDE \
                and self.has_point_on_left_side(point) < 0.0:
            result = -result
        return result

    def pseudoangle_to_point(self, point):
        # An "alternative" angle measure from AB to AP.
        # Ordering properties same as real angles.
        # Varies -2..0..2 as true angle varies -180..0..+180.
        # Not linear in true angle, but quicker to calculate.
        result = 1.0 - self._cos_angle_to_point(point)
        if abs(result) > COS_ANGLE_ZERO_MAGNITUDE \
                and self.has_point_on_left_side(point) < 0.0:
            result = -result
        return result

    def intersection_points_with_other(self, other):
        # Return the (two) intersection points with the other segment.
        # N.B. returns None if the two are parallel
        try:
            point_a = self.pole.cross_product(other.pole)
        except ZeroPointLatlonError:
            return None
        point_b = point_a.antipode()
        return (point_a, point_b)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                   repr(self.point_a),
                                   repr(self.point_b))

    def __str__(self):
        return 'SphSeg({!s} -> {!s}, pole={!s}>'.format(
            self.point_a._ll_str(),
            self.point_b._ll_str(),
            self.pole._ll_str())


class TooFewPointsForPolygonError(ValueError):
    def __init__(self, *args, **kwargs):
        if not args:
            args = ['Polygon must have at least 3 points.']
        super(TooFewPointsForPolygonError, self).__init__(*args, **kwargs)


class NonConvexPolygonError(ValueError):
    def __init__(self, *args, **kwargs):
        if not args:
            args = ['Polygon points cannot be reordered to make it convex.']
        super(NonConvexPolygonError, self).__init__(*args, **kwargs)


class SphAcwConvexPolygon(object):
    def __init__(self, points=[], in_degrees=False):
        self._set_points([sph_point(point, in_degrees=in_degrees)
                          for point in points])
        self._make_anticlockwise_convex()

    def _set_points(self, points):
        # Assign to our points, check length and uncache edges
        self.points = points
        self._edges = None
        self._centre_and_max_radius = None
        self.n_points = len(points)
        if self.n_points < 3:
            raise TooFewPointsForPolygonError()

    def _remove_duplicate_points(self):
        # Remove duplicate adjacent points so all angles can be calculated.
        # Only within current ordering: can still have duplicates elsewhere.
        points = self.points
        prev_points = [None] + points[:-1]
        points = [this for this, prev in zip(points, prev_points)
                  if prev is None or this != prev]
        self._set_points(points)

    def _is_anticlockwise_convex(self):
        # Check if our points are arranged in a convex anticlockwise chain.
        # To use this, must be able to calculate edges --> must have no
        # adjacent duplicated points.
        edges = self.edge_gcs()
        points = self.points
        points = points[2:] + points[:2]
        return all(edge.has_point_on_left_side(point) >= 0.0
                   for edge, point in zip(edges, points))

    def _make_anticlockwise_convex(self):
        # Reorder points as required, or raise an error if not possible.
        # Calculate a centre point
        points = self.points
        centre_xyz = [np.sum(x)
                      for x in zip(*[p.as_xyz() for p in points])]
        centre_point = sph_point(convert_xyz_to_latlon(*centre_xyz))
        # Make a reference edge from the centre to points[0]
        edge0 = SphGcSeg(centre_point, points[0])
        # Calculate (pseudo)angles from reference edge to all points
        angles = [edge0.pseudoangle_to_point(p) for p in points]
        # Sort points into this order, with original [0] at start
        angles = [(angle + 4.0) % 4.0 for angle in angles]
        points_and_angles = zip(points, angles)
        points_and_angles_sorted = sorted(points_and_angles,
                                          key=lambda p_and_a: p_and_a[1])
        new_points = [p_and_a[0] for p_and_a in points_and_angles_sorted]
        # Set new points, and remove any duplicates
        self._set_points(new_points)
        self._remove_duplicate_points()
        # Should now be as required
        if not self._is_anticlockwise_convex():
            raise NonConvexPolygonError()

    def edge_gcs(self):
        if self._edges is None:
            points = self.points
            next_points = points[1:] + points[:1]
            self._edges = [SphGcSeg(this_pt, next_pt)
                           for this_pt, next_pt in zip(points, next_points)]
        return self._edges

    def centre_and_max_radius(self):
        if self._centre_and_max_radius is None:
            xyz_all = np.array([point.as_xyz() for point in self.points])
            xyz_centre = np.mean(xyz_all, axis=0)
            centre_point = SphPoint(*convert_xyz_to_latlon(*xyz_centre))
            radius_cosines = [centre_point.dot_product(point)
                              for point in self.points]
            max_radius = math.acos(min(radius_cosines))
            self._centre_and_max_radius = (centre_point, max_radius)
        return self._centre_and_max_radius

    def contains_point(self, point, in_degrees=False):
        point = sph_point(point, in_degrees=in_degrees)
        return all(gc.has_point_on_left_side(point) >= 0.0
                   for gc in self.edge_gcs())

    def area(self):
        edges = self.edge_gcs()
        preceding_edges = edges[-1:] + edges[:-1]
        angle_total = sum([prev.reverse().angle_to_other(this)
                           for prev, this in zip(preceding_edges, edges)])
        angle_total -= math.pi * (self.n_points - 2)
        return angle_total

    def intersection_with_polygon(self, other):
        # Do fast check to exclude ones which are well separated
        centre_this, radius_this = self.centre_and_max_radius()
        centre_other, radius_other = other.centre_and_max_radius()
        spacing = centre_this.distance_to(centre_other)
        if spacing - radius_this - radius_other > 0:
            return None
        # Add output candidates: points from A that are in B, and vice versa
        result_points = [p for p in self.points
                         if other.contains_point(p) and p not in other.points]
        result_points += [p for p in other.points
                          if self.contains_point(p) and p not in result_points]
        # Calculate all intersections of (extended) edges between A and B
        inters_ab = [gc_a.intersection_points_with_other(gc_b)
                     for gc_a in self.edge_gcs() for gc_b in other.edge_gcs()]
        # remove 'None' cases, leaving a list of antipode pairs
        inters_ab = [x for x in inters_ab if x is not None]
        # flatten the pairs to a single list of points
        inters_ab = [x for x in itertools.chain.from_iterable(inters_ab)]
        # Add any intersections which are: inside both, not already seen
        result_points += [p for p in inters_ab
                          if (p not in result_points
                              and self.contains_point(p)
                              and other.contains_point(p))]
        if len(result_points) < 3:
            return None
        else:
            # Convert this bundle of points into a new SphAcwConvexPolygon
            return SphAcwConvexPolygon(points=result_points)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                               ', '.join([repr(p) for p in self.points]))

    def __str__(self):
        return 'SphPoly({})'.format(
            ', '.join([str(p) for p in self.points]))

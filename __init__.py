'''
Created on May 7, 2013

@author: itpp

Spherical gridcell area-overlap calculations.
'''

import itertools
import math


def convert_latlon_to_xyz(lat, lon):
    z = math.sin(lat)
    x = z * math.cos(lon)
    y = z * math.sin(lon)
    return (x, y, z)


def convert_xyz_to_latlon(x, y, z):
    mod_sq = (x * x + y * y + z * z)
    if abs(mod_sq) < 1e-7:
        raise ValueError('Point too close to zero for lat-lon conversion.')
    mod = math.sqrt(mod_sq)
    x, y, z = [p / mod for p in (x, y, z)]
    lat = math.asin(z)
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

    @property
    def as_xyz(self):
        if self._p3d is None:
            self._p3d = convert_latlon_to_xyz(self.lat, self.lon)
        return self._p3d

    def copy(self):
        return SphPoint(self)

    def antipode(self):
        x, y, z = self.as_xyz()
        return SphPoint(*convert_xyz_to_latlon(-x, -y, -z))


def sph_point(point, in_degrees=False):
    """ Convert lat,lon to SphPoint, or return SphPoint unchanged. """
    if hasattr(point, 'lat'):
        return point
    if len(point) != 2:
        raise ValueError('sph_point argument not a SphPoint. '
                         'Expected (lat, lon), got : {!r}'.format(point))
    if in_degrees:
        point = [x * math.pi/180.0 for x in point]
    return SphPoint(*point)


def sphpoints_dot_product(point_a, point_b):
    xyz_a = sph_point(point_a).as_xyz()
    xyz_b = sph_point(point_b).as_xyz()
    return sum([a * b for a, b in zip(xyz_a, xyz_b)])


def sphpoints_cross_product(point_a, point_b):
    xa, ya, za = sph_point(point_a).as_xyz()
    xb, yb, zb = sph_point(point_b).as_xyz()
    x, y, z = ((ay * bz - az * by),
               (az * bx - ax * bz),
               (ax * by - ay * bx))
    return sph_point(convert_xyz_to_latlon(x, y, z))


class SphGcLineseg(object):
    """ A great circle line segment, from 'A' to 'B' on the unit sphere. """
    def __init__(self, point_a, point_b):
        self.point_a = sph_point(point_a)
        self.point_b = sph_point(point_b)
        self.pole = sphpoints_cross_product(self.point_b, self.point_a)
        self.antipole = vector_negative(self.pole);

    def has_point_on_left(self, point):
        return sphpoints_dot_product(self.pole.as_xyz, point.as_xyz) > 0.0

    def cos_angle_to_other(self, other):
        # Angle from AB to AP
        return math.acos(sphpoints_dot_product(self.pole, other.pole))

    def angle_to_other(self, other):
        # Angle from AB to AP
        return math.acos(self.cos_angle_to_other(other))

    def cos_angle_to_point(self, point):
        # Angle from AB to AP
        seg2 = SphGcLineseg(self.point_a, point)
        return math.acos(sphpoints_dot_product(self.pole, seg2.pole))

    def angle_to_point(self, point):
        # Angle from AB to AP
        return math.acos(self.cos_angle_to_point(point))

    def intersect_points_with_other(self, other):
        point_a = sphpoints_cross_product(self.pole, other.pole)
        point_b = point_a.antipode()
        return (point_a, point_b)


class SphAcwConvexPolygon(object):
    def __init__(self, points=[], in_degrees=False, ordering_from_edge=None):
        self.points = [sph_point(point, in_degrees=in_degrees)
                       for point in points]
        if len(self.points) < 3:
            raise ValueError('Polygon must have at least 3 points.')
        self._edges = None
        self._make_anticlockwise_convex(ordering_from_edge=ordering_from_edge)

    def _is_anticlockwise_convex(self):
        # Check if our points are arranged in a convex anticlockwise chain
        # Optionally, raise an error if not
        edges = self.edge_gcs()
        result = True
        preceding_edge = edges[-1]
        for this_edge in edges:
            if not preceding_edge.has_point_on_left(this_edge.point_b):
                result = False
                break
            preceding_edge = this_edge
        return result

    def _make_anticlockwise_convex(self, ordering_from_edge=None):
        if self._is_anticlockwise_convex(error_if_not=False):
            return
        # Grab first edge if not given one
        if ordering_from_edge is not None:
            edge0 = ordering_from_edge
        else:
            edge0 = self.edge_gcs()[0]
        # Invalidate edges cache as we are about to reorder all the points
        self._edges = None
        # Calculate angles from reference edge to all other points
        points = self.points
        other_angles = [edge0.angle_to_point(p) for p in points[2:]]
        eps = 1.0e-7
        max_valid_angle = math.pi + eps
        min_valid_angle = -eps
        # Sort into angle order (forward or reverse as seems best)
        # Fix the first two angles, to avoid any problems.
        if all([x > min_valid_angle for x in other_angles]):
            # Try this with first 2 points in correct order
            angles = [-0.02, -0.01] + other_angles
        else:
            # Try with 2 points in reverse order
            points = points[2::-1] + points[2:]
            angles = [-0.02, -0.01] + [-x for x in other_angles]
        # Check for all valid angles : NOTE first two points must be suitable.
        if any([a > max_valid_angle or a < min_valid_angle
                for a in angles[2:]]):
            raise ValueError('SphAcwConvexPolygon cannot be made from given '
                             'points, starting with first two.')
        points_and_angles = zip(self.points, angles)
        points_and_angles_sorted = sorted(points_and_angles,
                                          key=lambda p_and_a: p_and_a[1])
        self.points = [p_and_a[0] for p_and_a in points_and_angles_sorted]
        if not self._is_anticlockwise_convex():
            raise ValueError(
                'SphAcwConvexPolygon points cannot be reordered to make it '
                'convex.')

    @property
    def edge_gcs(self):
        if self._edges is None:
            self._edges = [SphGcLineseg(points[i:i+2])
                         for i in range(len(self.points))]
            self._edges += [SphGcLineseg(points[-1], points[0])]
        return self._edges

    def contains_point(self, point, in_degrees=False):
        point = sph_point(point, in_degrees=in_degrees)
        return all(gc.has_point_on_left(point) for gc in self.edge_gcs)

    def area(self):
        angle_total = 0.0
        edges = self.edge_gcs()
        previous_edge = edges[-1]
        for this_edge in edges:
            angle_total += previous_edge.angle_to_other(this_edge)
        return angle_total - math.pi

    def intersection_with_polygon(self, other):
        # Add output candidates: points from A that are in B, and vice versa
        result_points = [p for p in self.points if other.contains_point(p)]
        result_points += [p for p in other.points if self.contains_point(p)]
        # Calculate all intersections of (extended) edges between A and B
        inters_ab += [gc_a.intersect_points_with_gc(gc_b)
                      for gc_a in self.edge_gcs for gc_b in other.edge_gcs]
        inters_ab = itertools.chain.from_iterable(inters_ab)  # ==flatten
        # Add to output: all A/B intersections within both areas
        result_points += [p for p in inters_ab
                        if (self.contains_point(p) 
                            and other.contains_point(p))]
        # Convert this bundle of points into a new SphAcwConvexPolygon
        # NOTE: only works because all points are inside original polygon
        edge0 = self.edge_gcs()[0]
        return SphAcwConvexPolygon(points=result_points,
                                   ordering_from_edge=edge0)



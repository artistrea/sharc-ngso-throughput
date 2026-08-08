# class CoordinateSystem():
#     """Class for transforming coordinates to local ENU using a reference lat, lon, alt.

#     This class receives a reference lat, lon, alt and may transform other coordinate types to local ENU.
#     """

#     def __init__(self, num_of_references: int = 1):
#         """
#         Initialize CoordinateSystem with unset reference coordinates.
#         You may create a specific number of references to vectorize calculations
#         """
#         self.n_refs = num_of_references

#         # geodesical
#         self.ref_lat = None
#         self.ref_long = None
#         self.ref_alt = None

#         # cartesian
#         self.ref_x = None
#         self.ref_y = None
#         self.ref_z = None

#         # rotation matrix used
#         self.translation = np.array([self.ref_x, self.ref_y, self.ref_z])
#         self.rotation = None

#     def get_translation(self):
#         """Return the translation value for the reference altitude and Earth's radius."""
#         """Return the translation value for the reference altitude.

#         Returns
#         -------
#         float
#             The sum of the reference altitude and Earth's radius in meters.
#         """
#         return self.ref_alt + EARTH_RADIUS_M

#     def validate(self):
#         """Validate that the reference coordinates for transformation are set."""
#         """Validate that the reference for coordinate transformation is set.

#         Raises
#         ------
#         ValueError
#             If the reference latitude, longitude, or altitude is not set.
#         """
#         if None in [self.ref_lat, self.ref_long, self.ref_alt]:
#             raise ValueError(
#                 "You need to set a reference for coordinate transformation before using it")

#     def set_reference(self, ref_lat: np.ndarray, ref_long: np.ndarray, ref_alt: np.ndarray):
#         """Set the reference latitude, longitude, and altitude for coordinate transformation."""
#         """Set the reference latitude, longitude, and altitude for coordinate transformation.

#         Parameters
#         ----------
#         ref_lat : np.ndarray (float)
#             Reference latitudes in degrees.
#         ref_long : np.ndarray (float)
#             Reference longitudes in degrees.
#         ref_alt : np.ndarray (float)
#             Reference altitudes in meters.
#         """
#         ref_lat, ref_long, ref_alt = map(np.atleast_1d, [ref_lat, ref_long, ref_alt])
#         if (
#             self.n_refs != len(ref_lat)
#             or self.n_refs != len(ref_long)
#             or self.n_refs != len(ref_alt)
#         ):
#             raise ValueError(
#                 f"CoordinateSytem has {self.n_refs} number of references, "
#                 "but different number of coordinates were passed"
#             )
#         self.ref_lat = to_scalar(ref_lat)
#         self.ref_long = to_scalar(ref_long)
#         self.ref_alt = to_scalar(ref_alt)
#         ref_x, ref_y, ref_z = lla2ecef(
#             self.ref_lat, self.ref_long, self.ref_alt)
#         self.ref_x = to_scalar(ref_x)
#         self.ref_y = to_scalar(ref_y)
#         self.ref_z = to_scalar(ref_z)

#         # ECEF considers xy plane with x axis pointing at lon = 0,
#         # local coords considers x axis pointing towards East
#         # and y pointing towards North

#         # translate everything so ES is at (0, 0, 0)
#         self.translation = np.array([self.ref_x, self.ref_y, self.ref_z])

#         # considering:
#         # - that by definition the vector pointing East
#         #     is already orthogonal to ECEF z axis,
#         #     in other words, it is fully contained in the ECEF xy plane;
#         # Then, a single rotation around ECEF z can align local East and positive x
#         # Local east and ECEF x axis are parallel and with same direction at long=-90
#         # rotation around ECEF z = -90 - ref_lon
#         rotation_around_z = -self.ref_long - 90

#         # considering:
#         # - that the local zenith is orthogonal to local east;
#         # - that the local zenith unit vector can be found by using geodetic (lat, lon)
#         #    as polar coordinates (as follows from geodetic lat, lon definition)
#         # - that the local east is now fully contained in the x axis;
#         # Then the local zenith is fully contained in the yz plane,
#         # and a single rotation around the x axis aligns it with global z
#         # More specifically, z_vec = polar(lat, ref_long - ref_long) = polar(lat, 0)
#         # and so a rotation of 90 - lat is what is necessary
#         rotation_around_x = self.ref_lat - 90

#         # since ECEF follows left hand rule and local coordinates also do,
#         # x axis points to local East and z points to local zenith,
#         # y axis already is aligned with local North after transformation
#         self.rotation = scipy.spatial.transform.Rotation.from_euler(
#             'zx',
#             np.stack([rotation_around_z, rotation_around_x], axis=-1),
#             degrees=True
#         )
#         # (M,3,3) or (3,3)
#         rot_mtx = self.rotation.as_matrix()
#         if rot_mtx.ndim == 2:
#             # guarantee (M, 3, 3)
#             rot_mtx = rot_mtx[None, ...]
#         # broadcastable (M, 1, 3, 3)
#         self.rotation_mtx = rot_mtx[:, None, :, :]

#         inv_rot_mtx = self.rotation.inv().as_matrix()
#         if inv_rot_mtx.ndim == 2:
#             # guarantee (M, 3, 3)
#             inv_rot_mtx = inv_rot_mtx[None, ...]
#         # broadcastable (M, 1, 3, 3)
#         self.inv_rotation_mtx = inv_rot_mtx[:, None, :, :]

#         # self.inv_rotation_mtx = self.rotation.inv().as_matrix()[:, None, :, :]   # (N,1,3,3)
#         # can also be confirmed comparing to here:
#         # https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates

#     def apply_rotation(self, xyz: np.ndarray, *, inverse=False):
#         """
#         Parameters
#         ----------
#         xyz: np.ndarray
#             xyz coordinates to rotate with rotation matrix
#             shape should be (N, 3) or (M, N, 3)

#         Returns
#         -------
#             rotated_xyz: np.ndarray shape = (M, N, 3)
#         """
#         if xyz.ndim == 3:
#             pts = xyz[:, :, :, None]              # (M,N,3,1)
#         else:
#             pts = xyz[None, :, :, None]              # (1,N,3,1)

#         rotated = np.matmul(self.rotation_mtx, pts)          # (M,N,3,1)
#         rotated = rotated[..., 0]                   # (M,N,3)

#         return rotated

#     def ecef2enu(
#         self, x, y, z, *, translate=None
#     ):
#         """Transform points by the same transformation required to bring reference to (0,0,0).

#         You can only rotate by specifying translate=0.

#         Parameters
#         ----------
#         x, y, z : array-like
#             Cartesian coordinates to transform.
#         translate : array-like or None, optional
#             Translation vector to use (default: use reference translation).

#         Returns
#         -------
#         np.ndarray
#             Transformed coordinates.
#         """
#         self.validate()
#         x, y, z = map(np.atleast_1d, [x, y, z])

#         translate_val = self.translation
#         if translate is not None:
#             translate_val = np.atleast_1d(translate)

#         # broadcast translate to have same number of dimensions as expected
#         xyz = np.stack([x, y, z], axis=-1)  # Nx3

#         xyz = xyz - translate_val[np.newaxis, :]

#         # rotate so axis are same as ENU
#         ret =  self.apply_rotation(xyz)
#         # print("ret.shape", ret.shape)
#         # print("ret[0].T.shape", ret[0].T.shape)

#         return ret

#     def enu2ecef(
#         self, x2, y2, z2, *, translate=None
#     ):
#         """Reverse transformed points by the same transformation required to bring reference to (0,0,0).

#         You can only rotate by specifying translate=0. You need to use the same 'translate' value used
#         in transformation if you wish to reverse the transformation correctly.

#         Parameters
#         ----------
#         x2, y2, z2 : array-like
#             Transformed cartesian coordinates to revert.
#         translate : array-like or None, optional
#             Translation vector to use (default: use reference translation).

#         Returns
#         -------
#         np.ndarray
#             Reverted coordinates.
#         """
#         self.validate()
#         x2, y2, z2 = map(np.atleast_1d, [x2, y2, z2])

#         # translate everything so ES is at (0, 0, 0)
#         translate_val = self.translation
#         if translate is not None:
#             translate_val = np.atleast_1d(translate)

#         # broadcast translate to have same number of dimensions as expected
#         xyz = np.stack([x2, y2, z2], axis=-1)  # Nx3

#         # rotate xyz back to ecef coord system
#         xyz = self.apply_rotation(xyz, inverse=True)

#         # translate earth reference back to its original ecef coord
#         return (xyz + translate_val[np.newaxis, np.newaxis, :])

#     def lla2enu(
#         self, lat: np.array, long: np.array, alt: np.array
#     ):
#         """Convert latitude, longitude, altitude to transformed cartesian coordinates.

#         This rotates and translates every point considering the reference that was set
#         and a geodesical coordinate system.

#         Parameters
#         ----------
#         lat : np.array
#             Latitude values.
#         long : np.array
#             Longitude values.
#         alt : np.array
#             Altitude values.

#         Returns
#         -------
#         np.ndarray
#             Transformed cartesian coordinates.
#         """
#         # get cartesian position by geodesical
#         x, y, z = lla2ecef(lat, long, alt)

#         return self.ecef2enu(x, y, z)

#     def station_ecef2enu(
#         self, station: StationManager, idx=None
#     ) -> None:
#         """In-place rotate and translate all coordinates so that reference parameters end up in (0,0,0).

#         Stations end up in the same relative position according to each other, adapting their angles to the rotation.
#         If idx is specified, only stations[idx] will be converted.

#         Parameters
#         ----------
#         station : StationManager
#             The station manager whose stations will be transformed.
#         idx : array-like or None, optional
#             Indices of stations to convert (default: all).
#         """
#         # transform positions
#         if idx is None:
#             nx, ny, nz = self.ecef2enu(
#                 station.x, station.y, station.z)[0].T
#         else:
#             nx, ny, nz = self.ecef2enu(
#                 station.x[idx], station.y[idx], station.z[idx])[0].T

#         if idx is None:
#             azim = station.azimuth
#             elev = station.elevation
#         else:
#             azim = station.azimuth[idx]
#             elev = station.elevation[idx]

#         r = 1
#         # then get pointing vec
#         pointing_vec_x, pointing_vec_y, pointing_vec_z = polar_to_cartesian(
#             r, azim, elev)

#         # transform pointing vectors, without considering geodesical earth
#         # coord system
#         pointing_vec_x, pointing_vec_y, pointing_vec_z = self.ecef2enu(
#             pointing_vec_x, pointing_vec_y, pointing_vec_z, translate=0)[0].T

#         if idx is None:
#             station.x = nx
#             station.y = ny
#             station.z = nz

#             _, station.azimuth, station.elevation = cartesian_to_polar(
#                 pointing_vec_x, pointing_vec_y, pointing_vec_z)
#         else:
#             station.x[idx] = nx
#             station.y[idx] = ny
#             station.z[idx] = nz

#             _, azimuth, elevation = cartesian_to_polar(
#                 pointing_vec_x, pointing_vec_y, pointing_vec_z)

#             station.azimuth[idx] = azimuth
#             station.elevation[idx] = elevation

#     def station_enu2ecef(
#         self, station: StationManager, idx=None
#     ) -> None:
#         """In-place rotate and translate all coordinates so that reference parameters end up in (0,0,0).

#         Stations end up in the same relative position according to each other, adapting their angles to the rotation.
#         If idx is specified, only stations[idx] will be converted.

#         Parameters
#         ----------
#         station : StationManager
#             The station manager whose stations will be transformed.
#         idx : array-like or None, optional
#             Indices of stations to convert (default: all).
#         """
#         # transform positions
#         if idx is None:
#             nx, ny, nz = self.enu2ecef(
#                 station.x, station.y, station.z)
#         else:
#             nx, ny, nz = self.enu2ecef(
#                 station.x[idx], station.y[idx], station.z[idx])

#         if idx is None:
#             azim = station.azimuth
#             elev = station.elevation
#         else:
#             azim = station.azimuth[idx]
#             elev = station.elevation[idx]

#         r = 1
#         # then get pointing vec
#         pointing_vec_x, pointing_vec_y, pointing_vec_z = polar_to_cartesian(
#             r, azim, elev)

#         # transform pointing vectors, without considering geodesical earth
#         # coord system
#         pointing_vec_x, pointing_vec_y, pointing_vec_z = self.enu2ecef(
#             pointing_vec_x, pointing_vec_y, pointing_vec_z, translate=0)

#         if idx is None:
#             station.x = nx
#             station.y = ny
#             station.z = nz

#             _, station.azimuth, station.elevation = cartesian_to_polar(
#                 pointing_vec_x, pointing_vec_y, pointing_vec_z)
#         else:
#             station.x[idx] = nx
#             station.y[idx] = ny
#             station.z[idx] = nz

#             _, azimuth, elevation = cartesian_to_polar(
#                 pointing_vec_x, pointing_vec_y, pointing_vec_z)

#             station.azimuth[idx] = azimuth
#             station.elevation[idx] = elevation
#
# TEST:
#
# import unittest
# import numpy as np
# import numpy.testing as npt
# from sharc.support.sharc_geom import CoordinateSystem
# from sharc.satellite.utils.sat_utils import ecef2lla, lla2ecef
# from sharc.station_manager import StationManager


# class TestGeometryConverter(unittest.TestCase):
#     """Unit tests for the CoordinateSystem class and related coordinate transformations."""

#     def setUp(self):
#         """Set up test fixtures for CoordinateSystem tests."""
#         self.conv0_0km = CoordinateSystem()
#         self.conv0_0km.set_reference(
#             0, 0, 0
#         )
#         self.conv0_52km = CoordinateSystem()
#         self.conv0_52km.set_reference(
#             0, 0, 52e3
#         )

#         self.conv1_0km = CoordinateSystem()
#         self.conv1_0km.set_reference(
#             -15, -47, 0
#         )
#         self.conv1_10km = CoordinateSystem()
#         self.conv1_10km.set_reference(
#             -15, -47, 10e3
#         )

#         self.all_converters = [
#             self.conv0_0km,
#             self.conv0_52km,
#             self.conv1_0km,
#             self.conv1_10km,
#         ]

#     def test_set_reference(self):
#         """Check if set_reference sets both LLA and ECEF coordinates correctly."""
#         # negative x in xaxis
#         self.conv0_0km.set_reference(0, 180, 1200)
#         self.assertEqual(self.conv0_0km.ref_alt, 1200)
#         self.assertEqual(self.conv0_0km.ref_long, 180)
#         self.assertEqual(self.conv0_0km.ref_lat, 0)

#         # almost radius of earth
#         self.assertAlmostEqual(self.conv0_0km.ref_x, -
#                                6378145, delta=self.conv0_0km.ref_alt)
#         self.assertAlmostEqual(self.conv0_0km.ref_y, 0)
#         self.assertAlmostEqual(self.conv0_0km.ref_z, 0)

#         # positive x in xaxis
#         self.conv0_0km.set_reference(0, 0, 200)
#         self.assertEqual(self.conv0_0km.ref_alt, 200)
#         self.assertEqual(self.conv0_0km.ref_long, 0)
#         self.assertEqual(self.conv0_0km.ref_lat, 0)

#         # almost radius of earth
#         self.assertAlmostEqual(
#             self.conv0_0km.ref_x,
#             6378145,
#             delta=self.conv0_0km.ref_alt)
#         self.assertAlmostEqual(self.conv0_0km.ref_y, 0)
#         self.assertAlmostEqual(self.conv0_0km.ref_z, 0)

#     def test_reference_ecef(self):
#         """Test ECEF to LLA conversion for reference points."""
#         for coord_sys in self.all_converters:
#             lat, lon, alt = ecef2lla(coord_sys.ref_x, coord_sys.ref_y, coord_sys.ref_z)
#             # ecef2lla approximation requires "almost equal" directive
#             print("coord_sys.ref_lat", coord_sys.ref_lat)
#             self.assertAlmostEqual(lat[0], coord_sys.ref_lat, places=8)
#             self.assertAlmostEqual(lon[0], coord_sys.ref_long, places=8)
#             self.assertAlmostEqual(alt[0], coord_sys.ref_alt, places=8)

#     def test_ecef_to_enu(self):
#         """Test ECEF to ENU coordinate transformation."""
#         # for each converter defined at the setup
#         for coord_sys in self.all_converters:
#             # check if reference point always goes to (0,0,0)
#             enus = coord_sys.ecef2enu(
#                 coord_sys.ref_x, coord_sys.ref_y, coord_sys.ref_z)
#             self.assertEqual(enus.shape, (1, 1, 3))
#             x, y, z = enus[0][0]
#             self.assertEqual(x, 0)
#             self.assertEqual(y, 0)
#             self.assertEqual(z, 0)

#     def test_lla_to_enu(self):
#         """Test LLA to ENU coordinate transformation."""
#         # for each converter defined at the setup
#         for coord_sys in self.all_converters:
#             enus = coord_sys.lla2enu(
#                 coord_sys.ref_lat, coord_sys.ref_long, coord_sys.ref_alt)
#             self.assertEqual(enus.shape, (1, 1, 3))
#             x, y, z = enus[0][0]
#             self.assertEqual(x, 0)
#             self.assertEqual(y, 0)
#             self.assertEqual(z, 0)

#     def test_enu_to_ecef(self):
#         """Test ENU to ECEF coordinate transformation."""
#         # for each converter defined at the setup
#         for coord_sys in self.all_converters:
#             # check if the reverse is true
#             ecefs = coord_sys.enu2ecef(0, 0, 0)
#             self.assertEqual(ecefs.shape, (1, 1, 3))
#             x, y, z = ecefs[0][0]
#             self.assertEqual(x, coord_sys.ref_x)
#             self.assertEqual(y, coord_sys.ref_y)
#             self.assertEqual(z, coord_sys.ref_z)

#     def test_station_converter(self):
#         """Test station coordinate and orientation conversions between ECEF and ENU."""
#         # for each converter defined at the setup
#         for coord_sys in self.all_converters:
#             rng = np.random.default_rng(0)
#             n_samples = 100
#             stations = StationManager(n_samples)
#             # place stations randomly with ecef
#             xyz_bef = lla2ecef(
#                 rng.uniform(-90, 90, n_samples),
#                 rng.uniform(-180, 180, n_samples),
#                 rng.uniform(0, 35e3, n_samples),
#             )
#             # set first station to be the reference
#             xyz_bef[0][0] = coord_sys.ref_x
#             xyz_bef[1][0] = coord_sys.ref_y
#             xyz_bef[2][0] = coord_sys.ref_z
#             # point them randomly
#             azim_bef = rng.uniform(-180, 180, n_samples)
#             elev_bef = rng.uniform(-90, 90, n_samples)

#             stations.x, stations.y, stations.z = xyz_bef
#             stations.azimuth = azim_bef
#             stations.elevation = elev_bef

#             # get relative distances and off axis while in ecef
#             dists_bef = stations.geom.get_3d_distance_to(stations.geom)
#             off_axis_bef = stations.geom.get_off_axis_angle(stations.geom)

#             # convert stations to enu
#             coord_sys.station_ecef2enu(stations)

#             # check if reference origin
#             self.assertEqual(stations.x[0], 0)
#             self.assertEqual(stations.y[0], 0)
#             self.assertEqual(stations.z[0], 0)

#             # get relative distances and off axis while in enu
#             dists_aft = stations.geom.get_3d_distance_to(stations.geom)
#             off_axis_aft = stations.geom.get_off_axis_angle(stations.geom)

#             # all stations should maintain same relative distances and off axis
#             # since their relative positioning and pointing should eq in ECEF
#             # and ENU
#             npt.assert_allclose(dists_aft, dists_bef)
#             npt.assert_allclose(off_axis_aft, off_axis_bef)

#             # NOTE: the next set of tests may not pass and the code still be
#             # correct...

#             # we can try to check if there are differences between stations before
#             # and after transformation.
#             # TODO: It would be more correct to not force equality for all cases,
#             # but only for most of them

#             # sometimes some values can be really similar before and after the
#             # transformation

#             # for example, reference may be on the same axis after and before transformation
#             # so we ignore the first station (used as reference) on these
#             # checks
#             npt.assert_equal(
#                 np.abs(stations.x[1:] - xyz_bef[0][1:]) > 1e3,
#                 True
#             )
#             npt.assert_equal(
#                 np.abs(stations.y[1:] - xyz_bef[1][1:]) > 1e3,
#                 True
#             )
#             npt.assert_equal(
#                 np.abs(stations.z[1:] - xyz_bef[2][1:]) > 1e3,
#                 True
#             )
#             # and the elevation angle may not change much if pointing vector is to the east/west
#             # since the pointing vector is aligned with the x axis, the rotation along it
#             # won't change the value much
#             npt.assert_equal(
#                 np.abs(stations.azimuth - azim_bef) > 0.4,
#                 True
#             )
#             npt.assert_equal(
#                 np.abs(stations.elevation - elev_bef) > 0.4,
#                 True
#             )

#             # return stations to starting case:
#             coord_sys.station_enu2ecef(stations)

#             # check if their position is the same as at the start
#             # some precision error occurs, so "almost equal" is needed
#             npt.assert_almost_equal(stations.x, xyz_bef[0])
#             npt.assert_almost_equal(stations.y, xyz_bef[1])
#             npt.assert_almost_equal(stations.z, xyz_bef[2])
#             npt.assert_almost_equal(stations.azimuth, azim_bef)
#             npt.assert_almost_equal(stations.elevation, elev_bef)


# if __name__ == '__main__':
#     unittest.main()


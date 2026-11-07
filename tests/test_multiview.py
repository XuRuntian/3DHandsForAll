import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from mano_io import default_parameters, validate, read_annotation, save_revision
from multiview_data import Camera, MultiViewSequence, mesh_overlay

ROOT = os.environ.get('MULTIVIEW_TEST_DATA', '/home/xrt/Downloads/Perspective_Generalization_09/rgbd')


class InterchangeTests(unittest.TestCase):
    def setUp(self):
        self.state = default_parameters('000012', 'left', 'master_camera:123')

    def test_original_and_versions_survive_roundtrip(self):
        corrected = copy.deepcopy(self.state)
        corrected['pose'] = np.linspace(-1.3, 1.2, 45).tolist()
        corrected['betas'] = np.linspace(-2, 2, 10).tolist()
        corrected['transl'] = [.1, -.2, .7]
        with tempfile.TemporaryDirectory() as directory:
            first = save_revision(directory, self.state, corrected)
            contents = first.read_bytes()
            second = save_revision(directory, self.state, corrected)
            self.assertNotEqual(first, second)
            self.assertEqual(contents, first.read_bytes())
            original, loaded = read_annotation(first, 'master_camera:123', '000012')
            self.assertEqual(original, self.state)
            self.assertEqual(loaded, corrected)

    def test_predictions_are_not_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prediction.json'
            path.write_text(json.dumps(self.state))
            before = path.read_bytes()
            original, corrected = read_annotation(path)
            corrected['pose'][0] = .25
            save_revision(directory, original, corrected, str(path))
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(original, self.state)

    def test_reject_ambiguous_or_invalid_parameters(self):
        for key, value in [('pose', [0]*48), ('betas', [0]*9), ('units', 'mm'),
                           ('rotation_representation', 'matrix'), ('pose_convention', 'pca'),
                           ('transl', [0, float('nan'), 1]), ('frame_id', '../x')]:
            state = dict(self.state, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(state)
        with self.assertRaises(ValueError):
            validate(self.state, 'master_camera:other')
        with self.assertRaises(ValueError):
            validate(self.state, frame_id='99')


class ProjectionTests(unittest.TestCase):
    def camera(self, transform=None):
        meta = dict(serial='123', original_calibration=dict(intrinsics=dict(color=dict(
            width=100, height=80, fx=100, fy=100, ppx=50, ppy=40,
            model='distortion.none', coeffs=[0]*5))))
        return Camera('cam00', meta, np.eye(4) if transform is None else transform)

    def test_extrinsic_direction_and_meter_translation(self):
        transform = np.eye(4)
        transform[:3,:3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        transform[:3,3] = [1, 2, 3]
        camera = self.camera(transform)
        uv, visible = camera.project([[1, 2, 4], [1, 2.1, 4]])
        np.testing.assert_allclose(uv, [[50,40], [60,40]], atol=1e-10)
        self.assertTrue(visible.all())

    def test_near_plane_and_source_image_immutable(self):
        camera = self.camera()
        points = np.array([[0,0,-1], [0,0,0], [.1,0,1.]])
        _, visible = camera.project(points)
        self.assertEqual(visible.tolist(), [False, False, True])
        image = np.zeros((80,100,3), dtype=np.uint8)
        overlay = mesh_overlay(image, camera, points, np.array([[0,1,2]]))
        self.assertFalse(overlay.any())
        self.assertFalse(image.any())

    def test_reject_nonrigid_transform(self):
        transform = np.eye(4)
        transform[0,0] = -1
        with self.assertRaises(ValueError):
            self.camera(transform)


@unittest.skipUnless(Path(ROOT).exists(), 'Real six-camera data not available')
class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sequence = MultiViewSequence(ROOT)

    def test_default_center_visible_in_all_cameras(self):
        point = self.sequence.initial_translation()
        for camera in self.sequence.cameras.values():
            uv, valid = camera.project([point])
            self.assertTrue(valid[0])
            self.assertTrue((uv[0] >= 0).all() and (uv[0] < camera.size).all())

    def test_sync_and_image_sizes(self):
        self.assertEqual(self.sequence.master_id, 'cam03')
        self.assertEqual(len(self.sequence.cameras), 6)
        for index in [0, len(self.sequence.frames)-1]:
            self.assertEqual(len(self.sequence.images(index)), 6)

    def test_against_realsense_sdk(self):
        try:
            import pyrealsense2 as rs
        except ImportError:
            self.skipTest('Optional pyrealsense2 SDK unavailable')
        rng = np.random.default_rng(8)
        for camera in self.sequence.cameras.values():
            intr = rs.intrinsics()
            intr.width, intr.height = camera.size
            intr.fx, intr.fy = camera.K[0,0], camera.K[1,1]
            intr.ppx, intr.ppy = camera.K[:2,2]
            intr.coeffs = camera.D.tolist()
            intr.model = rs.distortion.inverse_brown_conrady
            pixels = rng.uniform([40,40], [1240,680], (100,2))
            points = np.array([rs.rs2_deproject_pixel_to_point(intr,p.tolist(),1.) for p in pixels])
            master = points @ camera.master_from_camera[:3,:3].T + camera.master_from_camera[:3,3]
            uv, valid = camera.project(master)
            self.assertTrue(valid.all())
            np.testing.assert_allclose(uv, pixels, atol=.002)


if __name__ == '__main__':
    unittest.main()

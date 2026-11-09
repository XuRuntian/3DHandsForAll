"""Projection, opacity, and display-only GUI regression tests."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from multiview_data import Camera, HAND_CHAINS, mesh_overlay, skeleton_overlay
from test_multiview_gui import AVAILABLE, TestGeometry


class OverlayTests(unittest.TestCase):
    def setUp(self):
        meta = dict(serial='0', original_calibration=dict(intrinsics=dict(color=dict(
            width=100, height=80, fx=100, fy=100, ppx=50, ppy=40,
            model='none', coeffs=[0]*5))))
        self.camera = Camera('cam00', meta, np.eye(4))
        self.image = np.full((80,100,3), 90, dtype=np.uint8)
        self.joints = np.zeros((21,3))
        self.joints[:,2] = 1.
        for finger in range(5):
            for joint in range(4):
                self.joints[1+finger*4+joint,:2] = [(finger-2)*.07, -(joint+1)*.06]
        self.vertices = self.joints[[0,4,20]]
        self.faces = np.array([[0,1,2]])

    def test_openpose_topology_and_all_projected_joints(self):
        self.assertEqual(len(HAND_CHAINS), 5)
        self.assertEqual([i for chain in HAND_CHAINS for i in chain[1:]], list(range(1,21)))
        result = skeleton_overlay(self.image,self.camera,self.joints,alpha=1)
        xy,_ = self.camera.project(self.joints)
        for x,y in np.rint(xy).astype(int):
            self.assertFalse(np.array_equal(result[y,x],self.image[y,x]))
        with patch('multiview_data.cv2.line', wraps=cv2.line) as line:
            skeleton_overlay(self.image,self.camera,self.joints)
            self.assertEqual(line.call_count,20)

    def test_opacity_endpoints_blending_and_input_immutability(self):
        before = self.image.copy()
        for draw, args in [(mesh_overlay,(self.vertices,self.faces)),
                           (skeleton_overlay,(self.joints,))]:
            with self.subTest(renderer=draw.__name__):
                hidden = draw(self.image,self.camera,*args,alpha=0)
                opaque = draw(self.image,self.camera,*args,alpha=1)
                half = draw(self.image,self.camera,*args,alpha=.5)
                np.testing.assert_array_equal(hidden,self.image)
                self.assertTrue(np.any(opaque != self.image))
                np.testing.assert_array_equal(half,cv2.addWeighted(self.image,.5,opaque,.5,0))
                np.testing.assert_array_equal(self.image,before)
                for alpha in [-.1,1.1,float('nan')]:
                    with self.assertRaises(ValueError):draw(self.image,self.camera,*args,alpha=alpha)
        np.testing.assert_array_equal(mesh_overlay(self.image,self.camera,self.vertices,self.faces),
                                      mesh_overlay(self.image,self.camera,self.vertices,self.faces,alpha=.75))

    def test_invalid_near_plane_and_extreme_points_are_not_drawn(self):
        joints = self.joints.copy()
        joints[:,2] = -1
        np.testing.assert_array_equal(skeleton_overlay(self.image,self.camera,joints),self.image)
        joints[0] = [np.nan,0,1]
        joints[1] = [np.inf,0,1]
        joints[2] = [1e8,0,1]
        np.testing.assert_array_equal(skeleton_overlay(self.image,self.camera,joints),self.image)
        with self.assertRaises(ValueError):skeleton_overlay(self.image,self.camera,joints[:16])
        joints = self.joints.copy()
        joints[1,2] = -1
        with patch('multiview_data.cv2.line', wraps=cv2.line) as line:
            skeleton_overlay(self.image,self.camera,joints)
            self.assertEqual(line.call_count,18)

    def test_scaled_image_and_camera_transform(self):
        transform = np.eye(4)
        transform[:3,:3] = [[0,-1,0],[1,0,0],[0,0,1]]
        transform[:3,3] = [.4,.2,.3]
        camera = copy.deepcopy(self.camera)
        camera.R = transform[:3,:3].T
        camera.t = -camera.R @ transform[:3,3]
        master = self.joints @ transform[:3,:3].T + transform[:3,3]
        result = skeleton_overlay(self.image,camera,master,size=(200,160),alpha=1)
        self.assertEqual(result.shape,(160,200,3))
        xy,_ = self.camera.project(self.joints)
        for x,y in np.rint(xy*2).astype(int):
            self.assertFalse(np.array_equal(result[y,x], [90]*3))


@unittest.skipUnless(AVAILABLE, 'Needs Tk/display and torch')
class DisplayGuiTests(unittest.TestCase):
    def test_modes_opacity_details_and_navigation_do_not_edit_annotations(self):
        from multiview_annotator import MultiViewApp
        from test_sequence_contract import gui_args
        from sequence_store import ManoSequence
        with tempfile.TemporaryDirectory() as root, patch('multiview_annotator.CalibratedMano',TestGeometry):
            args = gui_args(root)
            app = MultiViewApp(args)
            errors = []
            app.window.report_callback_exception = lambda *e: errors.append(e)
            def widgets(parent):
                for child in parent.winfo_children():
                    yield child
                    yield from widgets(child)
            try:
                app.window.update();app.render()
                original = copy.deepcopy(app.mano_fit_tool.state)
                before = copy.deepcopy(app.document.arrays)
                mesh = app.display_images['cam00'].copy()
                radios = [w for w in widgets(app.window) if w.winfo_class()=='TRadiobutton']
                next(w for w in radios if w.cget('value')=='skeleton').invoke()
                app.window.update();app.render()
                skeleton = app.display_images['cam00'].copy()
                self.assertFalse(np.array_equal(mesh,skeleton))
                app.open_detail('cam00');app.window.update()
                np.testing.assert_array_equal(np.asarray(app.details[0]['canvas'].source),skeleton)
                # The detail window uses the same mode and transparency variables.
                detail_radios = [w for w in widgets(app.details[0]['window']) if w.winfo_class()=='TRadiobutton']
                next(w for w in detail_radios if w.cget('value')=='mesh').invoke()
                app.window.update();app.render()
                np.testing.assert_array_equal(app.display_images['cam00'],mesh)
                scales = [w for w in widgets(app.window) if w.winfo_class()=='Scale'
                          and str(w.cget('variable'))==str(app.transparency)]
                for scale in scales:
                    scale.set(100);app.window.update();app.render()
                    self.assertEqual(app.transparency_caption.get(),'100%')
                    for cid,image in app.display_images.items():
                        np.testing.assert_array_equal(image,app.images[cid])
                    np.testing.assert_array_equal(np.asarray(app.details[0]['canvas'].source),app.images['cam00'])
                    scale.set(0);app.window.update();app.render()
                app.overlay.set(False);app.update_display();app.render()
                np.testing.assert_array_equal(app.display_images['cam00'],app.images['cam00'])
                app.overlay.set(True)
                app.display_mode.set('skeleton');app.update_display();app.render()
                app.details[0]['overlay'].set(False);app.draw_detail(app.details[0])
                np.testing.assert_array_equal(np.asarray(app.details[0]['canvas'].source),app.images['cam00'])
                self.assertEqual(app.mano_fit_tool.state,original)
                self.assertEqual(app.undo_stack,[])
                self.assertEqual(app.redo_stack,[])
                for key in before:np.testing.assert_array_equal(app.document.arrays[key],before[key])
                self.assertFalse(Path(args.output).exists())
                # Missing images suppress even a valid hand overlay.
                app.sequence.frames[0]['images']['cam01'] = None
                np.testing.assert_array_equal(app.view_image('cam01',(100,80),True),app.images['cam01'])
                app.navigate(1);app.window.update();app.render()
                self.assertIsNone(app.vertices);self.assertIsNone(app.joints)
                for cid,image in app.display_images.items():
                    np.testing.assert_array_equal(image,app.images[cid])
                app.navigate(1);app.switch_side();app.window.update();app.render()
                self.assertEqual(app.display_mode.get(),'skeleton')
                self.assertEqual(app.transparency.get(),0)
                saved = ManoSequence.load(app.save())
                for key in before:np.testing.assert_array_equal(saved.arrays[key],before[key])
                self.assertFalse(errors,errors)
            finally:
                if app.pending_render is not None:app.window.after_cancel(app.pending_render)
                app.window.destroy()

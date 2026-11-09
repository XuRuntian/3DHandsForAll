"""Optional integration tests using licensed MANO assets supplied locally."""
import copy
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
import numpy as np
from mano_io import default_parameters, read_annotation, save_revision
from multiview_data import MultiViewSequence, mesh_overlay, skeleton_overlay

MODEL_ROOT = Path(os.environ.get('MANO_TEST_ROOT', '/home/xrt/models/mano_v1_2/models'))
DATA_ROOT = Path(os.environ.get('MULTIVIEW_TEST_DATA', '/home/xrt/Downloads/Perspective_Generalization_09/rgbd'))
try:
    from multiview_mano import CalibratedMano
    import tkinter
    from multiview_annotator import MultiViewApp
    AVAILABLE = all((MODEL_ROOT / f'MANO_{s}.pkl').is_file() for s in ('LEFT', 'RIGHT'))
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'Requires local MANO assets and runtime')
class RealManoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_sequence_contract import fixture
        cls.temporary=tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.sequence,_ = fixture(cls.temporary.name)

    def state(self, side):
        state = default_parameters('0', side, self.sequence.coordinate_system)
        state['transl'] = [0.,0.,.7]
        return state

    def test_openpose21_geometry_units_tips_and_shared_transforms(self):
        import cv2
        from unittest.mock import patch
        for side in ('left','right'):
            with self.subTest(side=side):
                model = CalibratedMano(MODEL_ROOT,self.state(side))
                with patch.object(model.layer,'forward',wraps=model.layer.forward) as forward:
                    vertices,joints = model.geometry()
                    self.assertEqual(forward.call_count,1)
                self.assertEqual(joints.shape,(21,3))
                self.assertTrue(np.isfinite(joints).all())
                tips = [745,319,444 if side=='right' else 445,556,673]
                np.testing.assert_allclose(joints[[4,8,12,16,20]],vertices[tips],atol=1e-8)
                # A nonzero shared shape ensures the shaped wrist is not assumed zero.
                model.state['betas'][0] = .8
                base_v,base_j = model.geometry()
                self.assertGreater(np.linalg.norm(base_j-joints),1e-4)
                model.state['pose'][0] = .4
                _,posed_j = model.geometry()
                self.assertGreater(np.linalg.norm(posed_j-base_j),1e-4)
                model.state['pose'][0] = 0.
                model.state['global_orient'] = [.1,-.2,.3]
                rotated_v,rotated_j = model.geometry()
                rot = cv2.Rodrigues(np.array(model.state['global_orient']))[0]
                np.testing.assert_allclose(rotated_j,(base_j-base_j[0])@rot.T+base_j[0],atol=1e-7)
                np.testing.assert_allclose(rotated_v,(base_v-base_j[0])@rot.T+base_j[0],atol=1e-7)
                model.state['transl'][0] += .025
                moved_v,moved_j = model.geometry()
                np.testing.assert_allclose(moved_j-rotated_j,np.tile([.025,0,0],(21,1)),atol=1e-8)
                np.testing.assert_allclose(moved_v-rotated_v,np.tile([.025,0,0],(778,1)),atol=1e-8)
                for cid,camera in self.sequence.cameras.items():
                    raw = self.sequence.images(0)[cid]
                    self.assertTrue(np.any(skeleton_overlay(raw,camera,moved_j) != raw))

    def test_both_models_edit_projection_and_exact_reload(self):
        images = self.sequence.images(0)
        for side in ('right', 'left'):
            with self.subTest(side=side):
                model = CalibratedMano(MODEL_ROOT, self.state(side))
                original = copy.deepcopy(model.state)
                base = model.vertices()
                self.assertEqual(base.shape, (778, 3))
                self.assertEqual(model.faces.shape, (1538, 3))
                self.assertTrue(np.isfinite(base).all())
                self.assertTrue(.1 < np.ptp(base, axis=0).max() < .3)
                for cid, camera in self.sequence.cameras.items():
                    uv, valid = camera.project(base)
                    self.assertTrue(valid.all())
                    self.assertTrue(((uv >= 0) & (uv < camera.size)).all())
                    raw = images[cid]
                    rendered = mesh_overlay(raw, camera, base, model.faces)
                    self.assertGreater(np.count_nonzero(raw != rendered), 100)
                for field, index, amount in [('pose', 0, .4), ('global_orient', 1, .3), ('betas', 0, 1.)]:
                    model.state = copy.deepcopy(original)
                    model.state[field][index] += amount
                    self.assertGreater(np.linalg.norm(model.vertices()-base), .001)
                model.state = copy.deepcopy(original)
                model.state['transl'][0] += .025
                np.testing.assert_allclose(model.vertices()-base, np.tile([.025,0,0], (778,1)), atol=1e-8)
                model.set_input_pose_i(4, .23456789)
                model.state['betas'][2] = .789123
                with tempfile.TemporaryDirectory() as directory:
                    path = save_revision(directory, original, model.state)
                    loaded_original, loaded = read_annotation(path)
                    self.assertEqual(loaded_original, original)
                    restored = CalibratedMano(MODEL_ROOT, loaded)
                    np.testing.assert_array_equal(restored.vertices(), model.vertices())

    @unittest.skipUnless(os.environ.get('DISPLAY'), 'Requires display')
    def test_actual_gui_edit_switch_save_reload(self):
        with tempfile.TemporaryDirectory() as output:
            from test_sequence_contract import gui_args
            args=gui_args(output,MODEL_ROOT)
            app = MultiViewApp(args)
            errors = []
            app.window.report_callback_exception = lambda *exc: errors.append(exc)
            try:
                app.window.update()
                app.render()
                for side in ('right', 'left'):
                    if app.side != side:
                        app.switch_side()
                    original = copy.deepcopy(app.mano_fit_tool.state)
                    app.on_trackbar_mano_finger1_ver('65')
                    app.on_trackbar_mano_rot1('55')
                    app.on_trackbar_mano_xyz_root0('48')
                    app.set_shape(1, '0.8')
                    app.window.update()
                    app.render()
                    edited = app.vertices.copy()
                    edited_joints = app.joints.copy()
                    path = app.save()
                    app.restore()
                    self.assertEqual(app.mano_fit_tool.state, original)
                    app._read(path)
                    app.sync_controls()
                    app.render()
                    np.testing.assert_array_equal(app.vertices, edited)
                    np.testing.assert_array_equal(app.joints, edited_joints)
                    for label in app.views.values():
                        self.assertIsNotNone(label.photo)
                app.open_detail('cam03')
                app.window.update()
                app.details[0]['zoom'].set(1.5)
                app.draw_detail(app.details[0])
                start = time.perf_counter()
                for _ in range(10):
                    app.render()
                print(f'Actual MANO six-view + detail render: {(time.perf_counter()-start)/10*1000:.1f} ms/update')
                self.assertFalse(errors, errors)
            finally:
                if app.pending_render is not None:
                    app.window.after_cancel(app.pending_render)
                app.window.destroy()

    @unittest.skipUnless(os.environ.get('DISPLAY'), 'Requires display')
    def test_real_tk_slider_commands_and_mouse_drag_do_not_reenter(self):
        """Exercise Tcl Scale commands and mouse bindings, not direct handler calls.

        The bounded probe makes the old bug fail at depth 8 rather than overflowing
        Python/Tcl's stack and hiding the first error in exception reporting.
        """
        import tkinter as tk
        import params
        from utils import get_mano_val_from_slider
        with tempfile.TemporaryDirectory() as output:
            from test_sequence_contract import gui_args
            args=gui_args(output,MODEL_ROOT)
            app = MultiViewApp(args)
            errors = []
            app.window.report_callback_exception = lambda *e: errors.append(str(e[1]))
            def descendants(widget):
                yield widget
                for child in widget.winfo_children():
                    yield from descendants(child)
            scales = [w for w in descendants(app.window) if isinstance(w, tk.Scale)]
            scale_for_variable = {str(w.cget('variable')): w for w in scales}
            stats = dict(depth=0, maximum=0, calls=0)
            thumb = scale_for_variable[str(app.window.slider_mano_fin0_ver_value)]
            def probe(value):
                stats['depth'] += 1
                stats['maximum'] = max(stats['maximum'], stats['depth'])
                stats['calls'] += 1
                try:
                    if stats['depth'] >= 8:
                        errors.append('Recursive Scale callback detected')
                        for scale in scales:
                            scale.configure(command='')
                        return
                    app.on_trackbar_mano_finger0_ver(value)
                finally:
                    stats['depth'] -= 1
            try:
                app.window.update()
                app.controls_notebook.select(1)
                app.choose_finger('拇指')
                app.window.update()
                thumb.configure(command=probe)
                # Programmatic Scale.set still dispatches the real Tcl command.
                thumb.set(70)
                app.window.update()
                self.assertFalse(errors, errors)
                self.assertLessEqual(stats['maximum'], 2)
                self.assertGreater(stats['calls'], 0)
                for side in ('right', 'left'):
                    if app.side != side:
                        app.switch_side()
                    app.controls_notebook.select(1)
                    for finger, name in enumerate(('拇指','食指','中指','无名指','小指')):
                        app.choose_finger(name)
                        app.window.update()
                        # Overall bend and joint-1 bend both synchronize sibling scales.
                        for suffix in ('ver', 'ver0'):
                            variable = getattr(app.window, f'slider_mano_fin{finger}_{suffix}_value')
                            scale = scale_for_variable[str(variable)]
                            before = copy.deepcopy(app.mano_fit_tool.state)
                            start = scale.coords(scale.get())
                            target = 25 if scale.get() > 50 else 75
                            end = scale.coords(target)
                            scale.event_generate('<ButtonPress-1>', x=start[0], y=start[1])
                            app.window.update()
                            scale.event_generate('<B1-Motion>', x=end[0], y=end[1], state=256)
                            app.window.update()
                            scale.event_generate('<ButtonRelease-1>', x=end[0], y=end[1])
                            app.window.update()
                            self.assertFalse(errors, (side, name, suffix, errors))
                            self.assertNotEqual(app.mano_fit_tool.state['pose'], before['pose'])
                            tick = int(scale.get())
                            index = getattr(params, f'fin{finger}_ver_idx1') - 3
                            expected = get_mano_val_from_slider(f'finger{finger}', 0, tick) + app.mano_fit_tool.mean[index]
                            self.assertAlmostEqual(app.mano_fit_tool.state['pose'][index], expected)
                            app.undo_edit()
                            self.assertEqual(app.mano_fit_tool.state, before)
                # Queued commands from import/synchronization must not quantize input.
                state = app.mano_fit_tool.state
                state['pose'] = np.linspace(-2.123456789, 2.987654321, 45).tolist()
                exact = copy.deepcopy(state)
                for _ in range(3):
                    app.sync_controls()
                    app.window.update()
                self.assertEqual(app.mano_fit_tool.state, exact)
                self.assertFalse(errors, errors)
            finally:
                if app.pending_render is not None:
                    app.window.after_cancel(app.pending_render)
                app.window.destroy()

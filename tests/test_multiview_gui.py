"""GUI integration with a synthetic triangle ONLY in tests, not a MANO substitute.

Run on a display with Tk and torch; no licensed model required. Actual MANO
asset/geometry correctness remains a separate acceptance test.
"""
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from mano_io import validate, read_annotation

try:
    import tkinter as tk
    from multiview_annotator import MultiViewApp
    from multiview_mano import CalibratedMano
    AVAILABLE = bool(os.environ.get('DISPLAY'))
except ImportError:
    AVAILABLE = False
    CalibratedMano = object

ROOT = os.environ.get('MULTIVIEW_TEST_DATA', '/home/xrt/Downloads/Perspective_Generalization_09/rgbd')


class TestGeometry(CalibratedMano):
    def __init__(self, root, state, device):
        self.state = validate(state)
        self.mean = np.zeros(45)
        self.faces = np.array([[0, 1, 2]], dtype=np.int32)
        self.snapshots = {}
        self.suspend_edits = False

    def vertices(self):
        return np.array([[0,0,0], [.03,0,0], [0,.03,0]]) + self.state['transl']

    def geometry(self):
        joints = np.zeros((21, 3))
        for finger in range(5):
            for joint in range(4):
                joints[1+finger*4+joint] = [(finger-2)*.02, (joint+1)*.02, 0]
        return self.vertices(), joints + self.state['transl']


@unittest.skipUnless(AVAILABLE, 'Needs Tk/display and torch')
class GuiTests(unittest.TestCase):
    def test_sliders_sync_save_restore_navigation(self):
        with tempfile.TemporaryDirectory() as output, patch('multiview_annotator.CalibratedMano', TestGeometry):
            from test_sequence_contract import gui_args
            from sequence_store import ManoSequence
            args=gui_args(output)
            app = MultiViewApp(args)
            errors = []
            app.window.report_callback_exception = lambda *exc: errors.append(exc)
            try:
                app.window.update()
                app.render()
                self.assertEqual(len(app.views), 6)
                # Import refresh must not round/clip parameters to old slider ranges.
                state = app.mano_fit_tool.state
                state['pose'] = np.linspace(-3.123456789, 3.987654321, 45).tolist()
                state['global_orient'] = [.123456789, -1.23456789, .321987654]
                state['transl'] = [.123456789, -.123456789, 1.987654321]
                exact = copy.deepcopy(state)
                app.sync_controls()
                app.window.update()
                self.assertEqual(state, exact)
                app.on_trackbar_mano_xyz_root0('60')
                self.assertAlmostEqual(state['transl'][0], .2)
                app.on_trackbar_mano_finger0_ver('70')
                app.window.update()
                self.assertNotEqual(state['pose'], exact['pose'])
                rotation = copy.deepcopy(state['global_orient'])
                app.button_freeze_rot_callback()
                app.on_trackbar_mano_rot0('75')
                app.button_freeze_rot_callback()
                self.assertEqual(state['global_orient'], rotation)
                app.set_shape(2, '1.25')
                app.render()
                path = app.save()
                original=copy.deepcopy(app.original)
                corrected=ManoSequence.load(path).record(0,'right')
                self.assertEqual(corrected, state)
                self.assertEqual(corrected['betas'][2], 1.25)
                app.restore()
                self.assertEqual(app.mano_fit_tool.state, original)
                app._read(path)
                app.sync_controls()
                self.assertEqual(app.mano_fit_tool.state, corrected)
                app.open_detail('cam03')
                app.window.update()
                app.details[0]['zoom'].set(1.5)
                app.details[0]['overlay'].set(False)
                app.draw_detail(app.details[0])
                app.navigate(1)
                app.navigate(-1)
                self.assertEqual(app.mano_fit_tool.state, corrected)
                app.switch_side()
                self.assertEqual(app.side, 'left')
                app.switch_side()
                self.assertEqual(app.mano_fit_tool.state, corrected)
                self.assertFalse(errors, errors)
            finally:
                if app.pending_render is not None:
                    app.window.after_cancel(app.pending_render)
                app.window.destroy()


    def test_sequence_import_navigation_and_prediction_preservation(self):
        from test_sequence_contract import gui_args
        with tempfile.TemporaryDirectory() as directory, patch('multiview_annotator.CalibratedMano',TestGeometry):
            args=gui_args(directory)
            before=Path(args.input).read_bytes()
            app=MultiViewApp(args)
            try:
                app.window.update()
                app.set_shape(0,1.25)
                app.navigate(1)
                self.assertEqual(app.sequence.frame_ids[app.index],'25')
                self.assertFalse(app.can_edit())
                app.navigate(1)
                self.assertTrue(app.can_edit())
                self.assertEqual(app.mano_fit_tool.state['betas'][0],1.25)
                app.switch_side()
                self.assertEqual(app.mano_fit_tool.state['betas'][0],0)
                self.assertEqual(Path(args.input).read_bytes(),before)
                current=copy.deepcopy(app.mano_fit_tool.state)
                with self.assertRaises((ValueError,FileNotFoundError)):
                    app._read(Path(directory)/'bad.npz')
                self.assertEqual(app.mano_fit_tool.state,current)
            finally:
                if app.pending_render is not None:app.window.after_cancel(app.pending_render)
                app.window.destroy()

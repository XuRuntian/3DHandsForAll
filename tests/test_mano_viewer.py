import json
import unittest
import os
import tempfile
from unittest.mock import patch
from urllib.request import urlopen
from urllib.error import HTTPError

from mano_viewer import ManoViewer


class ViewerTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('MANO_NATIVE_VIEWER_TEST') == '1',
                         'Opt-in real Qt desktop window')
    def test_native_window_reuse_reopen_and_close(self):
        viewer = ManoViewer()
        try:
            viewer.open()
            process = viewer.process
            self.assertIsNone(process.poll())
            viewer.open()
            self.assertIs(viewer.process, process)
            process.terminate()
            process.wait(timeout=10)
            viewer.open()
            self.assertIsNot(viewer.process, process)
            process = viewer.process
        finally:
            viewer.close()
        self.assertIsNotNone(process.poll())

    def test_read_only_state_and_local_assets(self):
        viewer = ManoViewer()
        try:
            viewer.update('157', [])
            with urlopen(viewer.url + 'state') as response:
                self.assertEqual(json.load(response), dict(frame='157', hands=[], revision=1))
            for path in ['', 'viewer.js', 'vendor/three.module.js', 'vendor/OrbitControls.js']:
                with urlopen(viewer.url + path) as response:
                    self.assertEqual(response.status, 200)
            for path in ['../sequence_store.py', 'metadata.json']:
                with self.assertRaises(HTTPError):
                    urlopen(viewer.url + path)
        finally:
            viewer.close()
        self.assertFalse(viewer.thread.is_alive())

    @unittest.skipUnless(os.environ.get('DISPLAY'), 'Needs display')
    def test_live_geometry_navigation_absence_and_shutdown(self):
        from test_sequence_contract import gui_args
        from test_multiview_gui import TestGeometry
        from multiview_annotator import MultiViewApp
        with tempfile.TemporaryDirectory() as root, \
                patch('multiview_annotator.CalibratedMano', TestGeometry), \
                patch('mano_viewer.ManoViewer.open'):
            app = MultiViewApp(gui_args(root))
            try:
                app.window.update()
                app.open_3d_view()
                before = json.loads(app.mano_viewer.payload)
                self.assertEqual(len(before['hands']), 2)
                app.adjust_global_component(3, 1)
                app.render()
                after = json.loads(app.mano_viewer.payload)
                self.assertNotEqual(before['hands'][0], after['hands'][0])
                self.assertEqual(before['hands'][1], after['hands'][1])
                app.navigate(1)
                app.render()
                self.assertEqual(json.loads(app.mano_viewer.payload)['hands'], [])
            finally:
                app.close()
            self.assertFalse(app.mano_viewer.thread.is_alive())

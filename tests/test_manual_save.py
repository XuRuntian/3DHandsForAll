import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sequence_store import ManoSequence
from test_sequence_contract import gui_args


@unittest.skipUnless(os.environ.get('DISPLAY'), 'Needs display')
class ManualSaveTests(unittest.TestCase):
    def test_undo_clears_dirty_and_close_never_creates_final(self):
        from test_multiview_gui import TestGeometry
        from multiview_annotator import MultiViewApp
        with tempfile.TemporaryDirectory() as root, patch('multiview_annotator.CalibratedMano', TestGeometry):
            app = MultiViewApp(gui_args(root))
            app.window.update()
            app.adjust_global_component(3, 1)
            self.assertTrue(app.has_unsaved_changes())
            app.undo_edit()
            self.assertFalse(app.has_unsaved_changes())
            app.adjust_global_component(3, 1)
            with patch('multiview_annotator.filedialog.askopenfilename', return_value=app.args.input), \
                    patch('multiview_annotator.messagebox.askyesno', return_value=False):
                app.load()
            self.assertTrue(app.has_unsaved_changes())
            app.close()
            self.assertFalse((Path(root)/'final').exists())

    def test_close_discards_edits_across_frames_and_hands(self):
        from test_multiview_gui import TestGeometry
        from multiview_annotator import MultiViewApp
        with tempfile.TemporaryDirectory() as root, patch('multiview_annotator.CalibratedMano', TestGeometry):
            args = gui_args(root)
            app = MultiViewApp(args)
            app.window.update()
            app.save()
            path = app.output/'mano.npz'
            before = path.read_bytes()
            app.adjust_global_component(3, 1)
            app.navigate(2)
            app.switch_side()
            self.assertTrue(app.has_unsaved_changes())
            self.assertEqual(path.read_bytes(), before)
            app.render()
            self.assertIn('未保存', app.status.get())
            app.switch_side()
            app.navigate(-2)
            self.assertAlmostEqual(app.mano_fit_tool.state['transl'][0], .001)
            with patch.object(app.document, 'save', side_effect=AssertionError('must not save')):
                app.close()
            self.assertEqual(path.read_bytes(), before)
            args.input = None
            reopened = MultiViewApp(args)
            self.assertAlmostEqual(reopened.mano_fit_tool.state['transl'][0], 0)
            reopened.close()

    def test_explicit_save_covers_sequence_and_failure_remains_dirty(self):
        from test_multiview_gui import TestGeometry
        from multiview_annotator import MultiViewApp
        with tempfile.TemporaryDirectory() as root, patch('multiview_annotator.CalibratedMano', TestGeometry):
            app = MultiViewApp(gui_args(root))
            try:
                app.window.update()
                app.adjust_global_component(3, 1)
                app.switch_side()
                app.set_shape(0, .3)
                app.navigate(2)
                with patch.object(app.document, 'save', side_effect=OSError('disk full')):
                    with self.assertRaises(OSError): app.save()
                self.assertTrue(app.has_unsaved_changes())
                app.save()
                self.assertFalse(app.has_unsaved_changes())
                saved = ManoSequence.load(app.output)
                self.assertAlmostEqual(saved.record(0, 'right')['transl'][0], .001)
                self.assertAlmostEqual(saved.record(0, 'left')['betas'][0], .3)
            finally:
                app.close()

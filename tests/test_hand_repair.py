"""Occasional-frame repair without estimator or licensed MANO assets."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sequence_store import ManoSequence
from test_sequence_contract import fixture, gui_args


class RepairStorageTests(unittest.TestCase):
    def test_repair_absence_roundtrip_and_legacy(self):
        with tempfile.TemporaryDirectory() as root:
            seq, doc = fixture(root)
            legacy = copy.deepcopy(doc.arrays)
            del legacy['repair_status']
            doc = ManoSequence(legacy, doc.meta)
            self.assertFalse(doc.arrays['repair_status'].any())
            source = doc.record(0, 'right')
            source['transl'] = [.1, .2, .7]
            source['pose'][3] = .25
            source['betas'][0] = .4
            doc.update(0, 'right', source)
            before = copy.deepcopy(doc.arrays)
            self.assertEqual(doc.reference_indices(1, 'right'), [0, 2])
            repaired = doc.repair_from(1, 'right', 0)
            self.assertEqual(repaired['frame_id'], '25')
            self.assertEqual(repaired['transl'], source['transl'])
            self.assertEqual(repaired['pose'], source['pose'])
            np.testing.assert_array_equal(doc.arrays['betas'], before['betas'])
            for key in ('valid', 'hand_pose', 'global_orient', 'transl'):
                np.testing.assert_array_equal(doc.arrays[key][[0, 2]], before[key][[0, 2]])
                np.testing.assert_array_equal(doc.arrays[key][:, 1], before[key][:, 1])
            doc.mark_absent(0, 'right')
            self.assertIsNone(doc.record(0, 'right'))
            self.assertEqual(doc.reference_indices(1, 'right'), [2])
            for reference in (0, 1, -1, 99):
                with self.assertRaises(ValueError):
                    doc.repair_from(1, 'right', reference)
            path = doc.save(Path(root)/'final')
            loaded = ManoSequence.load(path)
            np.testing.assert_array_equal(loaded.arrays['repair_status'], [[1, 0], [2, 0], [0, 0]])
            aligned = loaded.aligned(['80', '25', '10', '90'])
            np.testing.assert_array_equal(aligned.arrays['repair_status'][:, 0], [0, 2, 1, 0])

    def test_reject_inconsistent_status(self):
        with tempfile.TemporaryDirectory() as root:
            _, doc = fixture(root)
            for value, index in [(1, 0), (2, 1), (3, 0)]:
                arrays = copy.deepcopy(doc.arrays)
                arrays['repair_status'][index, 0] = value
                with self.assertRaises(ValueError):
                    ManoSequence(arrays, doc.meta)


@unittest.skipUnless(os.environ.get('DISPLAY'), 'Needs display')
class RepairGuiTests(unittest.TestCase):
    def test_repair_undo_absence_restore_autosave_and_resume(self):
        from multiview_annotator import MultiViewApp
        from test_multiview_gui import TestGeometry
        with tempfile.TemporaryDirectory() as root, patch('multiview_annotator.CalibratedMano', TestGeometry):
            args = gui_args(root)
            app = MultiViewApp(args)
            def destroy():
                if app.pending_render is not None:
                    app.window.after_cancel(app.pending_render)
                app.window.destroy()
            try:
                app.window.update()
                app.set_shape(0, .3)
                app.navigate(1)
                before = copy.deepcopy(app.document.arrays)
                app.repair_from_reference(0)
                self.assertTrue(app.can_edit())
                self.assertEqual(app.mano_fit_tool.state['betas'][0], .3)
                app.undo_edit()
                self.assertFalse(app.can_edit())
                for key in before:
                    np.testing.assert_array_equal(app.document.arrays[key], before[key])
                app.redo_edit()
                self.assertTrue(app.can_edit())
                with patch('multiview_annotator.messagebox.askyesno', return_value=True):
                    app.mark_absent()
                self.assertFalse(app.can_edit())
                app.undo_edit()
                self.assertTrue(app.can_edit())
                app.redo_edit()
                app.navigate(1)
                disk = ManoSequence.load(Path(root)/'final')
                self.assertEqual(disk.arrays['repair_status'][1, 0], 1)
                app.navigate(-1)
                app.restore()
                self.assertFalse(app.can_edit())
                self.assertEqual(app.document.arrays['repair_status'][1, 0], 0)
                app.repair_from_reference(2)
                app.save()
            finally:
                destroy()
            args.input = None
            args.frame_id = '25'
            app = MultiViewApp(args)
            try:
                self.assertTrue(app.can_edit())
                self.assertEqual(app.document.arrays['repair_status'][1, 0], 2)
                app.window.update()
                app.open_repair()
                app.window.update()
                dialogs = [w for w in app.window.winfo_children() if w.winfo_class() == 'Toplevel']
                self.assertEqual(len(dialogs), 1)
                def descendants(widget):
                    for child in widget.winfo_children():
                        yield child
                        yield from descendants(child)
                for dimensions in ('960x620', '600x420'):
                    dialogs[0].geometry(dimensions)
                    app.window.update()
                    buttons = [w for w in descendants(dialogs[0])
                               if w.winfo_class() == 'TButton']
                    self.assertEqual(len(buttons), 2)
                    for button in buttons:
                        self.assertTrue(button.winfo_ismapped(), button.cget('text'))
                        self.assertLessEqual(button.winfo_rooty() + button.winfo_height(),
                                             dialogs[0].winfo_rooty() + dialogs[0].winfo_height())
                        self.assertGreaterEqual(button.winfo_height(), button.winfo_reqheight())
                dialogs[0].destroy()
            finally:
                destroy()


if __name__ == '__main__':
    unittest.main()

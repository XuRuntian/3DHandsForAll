"""Timeline navigation uses the existing save and sparse-frame contracts."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from test_multiview_gui import AVAILABLE, TestGeometry
from test_sequence_contract import gui_args
from sequence_store import ManoSequence


@unittest.skipUnless(AVAILABLE, 'Needs Tk/display and torch')
class TimelineTests(unittest.TestCase):
    def make_app(self, single=False):
        from multiview_annotator import MultiViewApp
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        args = gui_args(temporary.name)
        if single:
            path = Path(args.project)
            manifest = json.loads(path.read_text())
            manifest['frames'] = manifest['frames'][:1]
            path.write_text(json.dumps(manifest))
            loaded = ManoSequence.load(args.input)
            doc = ManoSequence({key: value if key == 'betas' else value[:1]
                                for key, value in loaded.arrays.items()}, loaded.meta)
            doc.meta['image_manifest_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            args.input = str(doc.save(Path(temporary.name) / 'single_predictions'))
        patcher = patch('multiview_annotator.CalibratedMano', TestGeometry)
        patcher.start()
        self.addCleanup(patcher.stop)
        app = MultiViewApp(args)
        def destroy():
            app.cancel_frame_seek()
            if app.pending_render is not None:
                app.window.after_cancel(app.pending_render)
            app.window.destroy()
        self.addCleanup(destroy)
        app.window.update()
        self.errors = []
        app.window.report_callback_exception = lambda *e: self.errors.append(e)
        return app

    def pump(self, app, seconds=.15):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            app.window.update()
            time.sleep(.005)
        self.assertFalse(self.errors, self.errors)

    def test_seek_sparse_frames_save_edits_and_sync_existing_navigation(self):
        app = self.make_app()
        original = Path(app.args.input).read_bytes()
        app.mano_fit_tool.state['transl'][0] = .123
        app.update_rendered_img()
        app.timeline.set(1)
        app.flush_frame_seek()
        self.assertEqual(app.sequence.frame_ids[app.index], '25')
        self.assertFalse(app.can_edit())
        self.assertFalse(app.timeline.instate(['disabled']))
        saved = ManoSequence.load(app.output)
        self.assertAlmostEqual(saved.record(0, 'right')['transl'][0], .123)
        app.timeline.set(2)
        self.pump(app)
        self.assertEqual(app.sequence.frame_ids[app.index], '80')
        app.navigate(-1)
        self.assertEqual(app.timeline_position.get(), 1)
        self.assertEqual(app.timeline_caption.get(), '2 / 3')
        app.frame_entry.insert(0, '10')
        app.jump_to_frame()
        self.assertEqual(app.timeline_position.get(), 0)
        self.assertEqual(Path(app.args.input).read_bytes(), original)

    def test_drag_coalesces_and_button_cancels_pending_seek(self):
        app = self.make_app()
        with patch.object(app.sequence, 'images', wraps=app.sequence.images) as images:
            app.timeline.set(1)
            app.timeline.set(2)
            self.assertEqual(images.call_count, 0)
            app.flush_frame_seek()
            self.assertEqual(images.call_count, 1)
            self.assertEqual(app.index, 2)
        app.navigate(-2)
        app.timeline.set(2)
        app.navigate(1)
        self.pump(app)
        self.assertEqual(app.index, 1)
        app.timeline.set(1.2)
        app.flush_frame_seek()
        self.assertEqual(app.timeline_position.get(), 1)

    def test_load_or_save_error_keeps_frame_and_resets_slider(self):
        app = self.make_app()
        with patch('multiview_annotator.messagebox.showerror') as error:
            with patch.object(app.sequence, 'images', side_effect=ValueError('missing image')):
                app.timeline.set(2)
                app.flush_frame_seek()
            self.assertEqual(app.index, 0)
            self.assertEqual(app.timeline_position.get(), 0)
            app.mano_fit_tool.state['transl'][0] = .456
            with patch.object(app.document, 'save', side_effect=OSError('disk full')):
                app.timeline.set(1)
                app.flush_frame_seek()
            self.assertEqual(app.index, 0)
            self.assertEqual(app.timeline_position.get(), 0)
            self.assertAlmostEqual(app.mano_fit_tool.state['transl'][0], .456)
            self.assertEqual(error.call_count, 2)

    def test_single_frame_and_programmatic_sync_do_not_navigate(self):
        app = self.make_app(single=True)
        self.assertTrue(app.timeline.instate(['disabled']))
        self.assertEqual(app.timeline_caption.get(), '1 / 1')
        with patch.object(app, 'navigate', side_effect=AssertionError('unexpected navigation')):
            app.sync_timeline()
            self.pump(app)
        self.assertIsNone(app.pending_frame_seek)

    def test_mouse_drag_and_responsive_bounds(self):
        app = self.make_app()
        for geometry in ('1440x940', '1040x720'):
            app.window.geometry(geometry)
            app.window.update()
            slider = app.timeline
            self.assertGreater(slider.winfo_width(), 400)
            self.assertLessEqual(slider.winfo_rootx() + slider.winfo_width(),
                                 app.window.winfo_rootx() + app.window.winfo_width())
            app.navigate(-app.index)
            app.window.update()
            x, y = slider.coords(0)
            end_x, end_y = slider.coords(2)
            slider.event_generate('<ButtonPress-1>', x=x, y=y)
            slider.event_generate('<B1-Motion>', x=end_x, y=end_y)
            slider.event_generate('<ButtonRelease-1>', x=end_x, y=end_y)
            self.pump(app)
            self.assertEqual(app.index, 2)

    def test_continuous_drag_updates_before_release_and_clamps_outside(self):
        app = self.make_app()
        slider = app.timeline
        x, y = slider.coords(0)
        middle, _ = slider.coords(1)
        end, _ = slider.coords(2)
        slider.event_generate('<ButtonPress-1>', x=x, y=y)
        slider.event_generate('<B1-Motion>', x=middle, y=y)
        self.pump(app)
        self.assertEqual(app.index, 1)
        slider.event_generate('<B1-Motion>', x=end+100, y=y)
        slider.event_generate('<ButtonRelease-1>', x=end+100, y=y)
        self.pump(app)
        self.assertEqual(app.index, 2)
        self.assertEqual(app.timeline_position.get(), 2)

    def test_close_cancels_pending_seek(self):
        app = self.make_app()
        app.timeline.set(2)
        self.assertIsNotNone(app.pending_frame_seek)
        with patch.object(app.window, 'destroy') as destroy, patch.object(app, 'navigate') as navigate:
            app.close()
            destroy.assert_called_once()
            self.pump(app)
            navigate.assert_not_called()
        self.assertIsNone(app.pending_frame_seek)

    def test_modal_actions_cancel_pending_seek_before_editing(self):
        app = self.make_app()
        app.timeline.set(1)
        with patch('multiview_annotator.tk.Toplevel.grab_set'):
            app.open_repair()
        try:
            self.pump(app)
            self.assertEqual(app.index, 0)
            self.assertEqual(app.timeline_position.get(), 0)
        finally:
            for child in app.window.winfo_children():
                if child.winfo_class() == 'Toplevel':
                    child.destroy()
        def modal_wait(*args, **kwargs):
            self.pump(app)
            self.assertEqual(app.index, 0)
            return ''
        app.timeline.set(2)
        with patch('multiview_annotator.filedialog.askopenfilename', side_effect=modal_wait):
            app.load()
        app.timeline.set(2)
        with patch('multiview_annotator.messagebox.askyesno', side_effect=modal_wait):
            app.mark_absent()
        self.assertIsNone(app.pending_frame_seek)


if __name__ == '__main__':
    unittest.main()

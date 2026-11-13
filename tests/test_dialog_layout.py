"""Native Tk confirmation layout under broken display DPI reporting."""
import os
import unittest


@unittest.skipUnless(os.environ.get('DISPLAY'), 'Needs display')
class DialogLayoutTests(unittest.TestCase):
    def test_confirmation_has_readable_horizontal_text(self):
        import tkinter as tk
        from tkinter import font, messagebox
        from multiview_annotator import MultiViewApp

        root = tk.Tk()
        previous = float(root.tk.call('tk', 'scaling'))
        observations = {}
        try:
            root.tk.call('tk', 'scaling', .039)
            app = MultiViewApp.__new__(MultiViewApp)
            app.window = root
            app.configure_chinese_ui()
            def inspect():
                dialog = '.__tk__messagebox'
                try:
                    message = dialog + '.msg'
                    wrap = root.tk.call(message, 'cget', '-wraplength')
                    observations['wrap'] = root.winfo_fpixels(wrap)
                    observations['font_size'] = font.nametofont('TkCaptionFont', root=root).cget('size')
                    observations['height'] = int(root.tk.call('winfo', 'height', message))
                finally:
                    root.tk.call(dialog + '.no', 'invoke')
            root.after(150, inspect)
            self.assertFalse(messagebox.askyesno('标记无手', '确认帧 10 的右手不存在？', parent=root))
            self.assertGreaterEqual(observations['wrap'], 200)
            self.assertEqual(observations['font_size'], -16)
            self.assertLess(observations['height'], 60)
        finally:
            root.tk.call('tk', 'scaling', previous)
            root.destroy()

    def test_normal_high_dpi_scaling_is_preserved(self):
        import tkinter as tk
        from multiview_annotator import MultiViewApp
        root = tk.Tk()
        previous = float(root.tk.call('tk', 'scaling'))
        try:
            root.tk.call('tk', 'scaling', 2.0)
            before = float(root.tk.call('tk', 'scaling'))
            app = MultiViewApp.__new__(MultiViewApp)
            app.window = root
            app.configure_chinese_ui()
            self.assertAlmostEqual(float(root.tk.call('tk', 'scaling')), before)
        finally:
            root.tk.call('tk', 'scaling', previous)
            root.destroy()

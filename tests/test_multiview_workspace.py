"""Viewport geometry and editing interactions, without a prediction model."""
import copy
import tempfile
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from test_multiview_gui import AVAILABLE, ROOT, TestGeometry
try:
    from multiview_viewport import image_geometry
    from multiview_annotator import MultiViewApp
    VIEWPORT_AVAILABLE = True
except ImportError:
    VIEWPORT_AVAILABLE = False


@unittest.skipUnless(VIEWPORT_AVAILABLE, 'Needs Tk')
class ViewportGeometryTests(unittest.TestCase):
    def test_fit_preserves_full_image_and_cover_fills_without_stretch(self):
        for size in ((320,240),(1000,240),(900,720)):
            w,h,x,y = image_geometry((1280,720),size,'fit')
            self.assertLessEqual(w,size[0]); self.assertLessEqual(h,size[1])
            self.assertAlmostEqual(w/h,1280/720,delta=.01)
            self.assertGreaterEqual(x,0);self.assertGreaterEqual(y,0)
            w,h,x,y = image_geometry((1280,720),size,'cover')
            self.assertGreaterEqual(w,size[0]);self.assertGreaterEqual(h,size[1])
            self.assertAlmostEqual(w/h,1280/720,delta=.01)
            self.assertLessEqual(x,0);self.assertLessEqual(y,0)

    def test_pan_and_zoom_are_display_only(self):
        w,h,x,y=image_geometry((1280,720),(640,360),'fit',2,(50,-30))
        self.assertEqual((w,h,x,y),(1280,720,-270.,-210.))


@unittest.skipUnless(AVAILABLE and VIEWPORT_AVAILABLE and (Path(ROOT)/'calibration/extrinsics.json').is_file(),
                         'Needs Tk/display and real data')
class WorkspaceTests(unittest.TestCase):
    def test_resize_focus_nudge_and_undo(self):
        with tempfile.TemporaryDirectory() as output, patch('multiview_annotator.CalibratedMano',TestGeometry):
            args=SimpleNamespace(multiview=ROOT,output=output,frame_id='0',hand_side='right',input=None,mano_root='unused',device='cpu')
            app=MultiViewApp(args)
            errors=[]
            app.window.report_callback_exception=lambda *e:errors.append(e)
            try:
                app.window.update();app.render()
                original=copy.deepcopy(app.mano_fit_tool.state)
                self.assertTrue(all(w.winfo_ismapped() for w in app.global_scales))
                app.controls_notebook.select(1);app.window.update()
                self.assertFalse(any(w.winfo_ismapped() for w in app.global_scales))
                app.controls_notebook.select(0);app.window.update()
                self.assertTrue(all(w.winfo_ismapped() for w in app.global_scales))
                self.assertEqual([app.controls_notebook.tab(t,'text') for t in app.controls_notebook.tabs()],
                                 ['整体','手指','手型','精确'])
                # Each parameter has its own inline buttons; global increments
                # match the adjacent master-camera numeric input, not selected view axes.
                app.rotation_step.set('0.1')
                app.inline_buttons[0][1].invoke();app.window.update()
                self.assertAlmostEqual(app.mano_fit_tool.state['global_orient'][0],
                                       original['global_orient'][0]+np.deg2rad(.1))
                app.undo_edit();self.assertEqual(app.mano_fit_tool.state,original)
                app.nudge_step.set('0.5')
                app.inline_buttons[3][1].invoke();app.window.update()
                self.assertAlmostEqual(app.mano_fit_tool.state['transl'][0],original['transl'][0]+.0005)
                app.undo_edit();self.assertEqual(app.mano_fit_tool.state,original)
                app.controls_notebook.select(1);app.choose_finger('拇指');app.window.update()
                app.inline_buttons[6][1].invoke();app.window.update()
                self.assertNotEqual(app.mano_fit_tool.state['pose'],original['pose'])
                app.undo_edit();self.assertEqual(app.mano_fit_tool.state,original)
                app.controls_notebook.select(2);app.window.update()
                app.inline_buttons[-10][1].invoke();app.window.update()
                self.assertAlmostEqual(app.mano_fit_tool.state['betas'][0],original['betas'][0]+.01)
                app.undo_edit();self.assertEqual(app.mano_fit_tool.state,original)
                app.controls_notebook.select(0);app.window.update()
                before=app.views['cam00'].winfo_width()
                app.window.geometry('1120x780');app.window.update();app.render()
                self.assertNotEqual(app.views['cam00'].winfo_width(),before)
                for view in app.views.values():
                    self.assertEqual(view.photo.width(),view.winfo_width())
                    self.assertEqual(view.photo.height(),view.winfo_height())
                app.focus_camera('cam01');app.window.update();app.render()
                self.assertGreater(app.focus_view.winfo_height(),app.views['cam01'].winfo_height()*2)
                app.focus_view.zoom_at(2,100,100)
                app.focus_view.pan=(10.,20.);app.focus_view.draw()
                app.fit_mode.set('铺满画面');app.change_fit()
                self.assertEqual(app.focus_view.mode,'cover')
                self.assertEqual(app.focus_view.zoom,1.)
                self.assertEqual(app.mano_fit_tool.state,original)
                app.nudge_step.set('1');app.nudge(0,1)
                delta=np.asarray(app.mano_fit_tool.state['transl'])-original['transl']
                np.testing.assert_allclose(app.sequence.cameras['cam01'].R@delta,[.001,0,0],atol=1e-12)
                changed=copy.deepcopy(app.mano_fit_tool.state)
                app.undo_edit();self.assertEqual(app.mano_fit_tool.state,original)
                app.redo_edit();self.assertEqual(app.mano_fit_tool.state,changed)
                import cv2
                before_rotation = copy.deepcopy(app.mano_fit_tool.state)
                app.rotation_step.set('0.5')
                app.rotate_nudge(1,1)
                rotation = cv2.Rodrigues(np.asarray(app.mano_fit_tool.state['global_orient']))[0]
                camera = app.sequence.cameras[app.selected_camera]
                expected = camera.R.T @ cv2.Rodrigues(np.array([0.,np.deg2rad(.5),0.]))[0] @ camera.R
                np.testing.assert_allclose(rotation,expected,atol=1e-10)
                self.assertEqual(app.mano_fit_tool.state['transl'],before_rotation['transl'])
                app.undo_edit();self.assertEqual(app.mano_fit_tool.state,changed)
                app.global_entries[3][0].set('123.456')
                app.apply_global_value(3)
                self.assertAlmostEqual(app.mano_fit_tool.state['transl'][0],.123456)
                app.undo_edit();self.assertEqual(app.mano_fit_tool.state,changed)
                # Keyboard navigation must not move a hand while typing a number.
                app.global_entries[3][1].focus_force();app.window.update()
                app.keyboard_nudge(SimpleNamespace(state=0),0,1)
                self.assertEqual(app.mano_fit_tool.state,changed)
                app.focus_view.start_drag(SimpleNamespace(x=10,y=10),None);app.window.update()
                app.keyboard_nudge(SimpleNamespace(state=1),0,1)
                keyboard_delta=np.asarray(app.mano_fit_tool.state['transl'])-changed['transl']
                np.testing.assert_allclose(camera.R@keyboard_delta,[.0001,0,0],atol=1e-12)
                app.undo_edit();self.assertEqual(app.mano_fit_tool.state,changed)
                app.choose_finger('小指')
                self.assertEqual(app.finger_choice.get(),'小指')
                # Splitter can expand the editor without changing the MANO.
                app.workspace_split.sash_place(0,520,0);app.window.update()
                self.assertGreater(app.editor_panel.winfo_width(),480)
                self.assertEqual(app.mano_fit_tool.state,changed)
                app.begin_gesture()
                for v in ('.1','.2','.3'):app.set_shape(0,v)
                app.end_gesture();app.undo_edit()
                self.assertEqual(app.mano_fit_tool.state,changed)
                app.leave_focus();app.window.update();app.render()
                self.assertEqual(app.layout_mode.get(),'六视图')
                app.open_detail('cam00');app.window.update();app.draw_detail(app.details[0])
                detail=app.details[0]
                self.assertEqual(detail['canvas'].photo.width(),detail['canvas'].winfo_width())
                app.native_detail(detail);app.reset_detail(detail)
                self.assertEqual(detail['canvas'].zoom,1.)
                self.assertFalse(errors,errors)
            finally:
                if app.pending_render is not None:app.window.after_cancel(app.pending_render)
                app.window.destroy()

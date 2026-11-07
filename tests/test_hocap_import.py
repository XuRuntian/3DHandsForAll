"""Real exported bundle: geometry equivalence, projection, editing and persistence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

BUNDLE = Path(os.environ.get('HOCAP_TEST_ROOT','/home/xrt/Downloads/hocap_hand_v1'))
MODELS = Path(os.environ.get('MANO_TEST_ROOT','/home/xrt/models/mano_v1_2/models'))
try:
    from hocap_adapter import HoCapSequence, convert_predictions, export_project
    from sequence_store import ManoSequence
    from mano_io import parse_sequence, read_annotation
    from multiview_annotator import MultiViewApp
    AVAILABLE = BUNDLE.is_dir() and (MODELS/'MANO_LEFT.pkl').is_file()
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE,'Requires real HO-Cap bundle, MANO assets and runtime')
class HoCapImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sequence = HoCapSequence(BUNDLE)
        cls.source_hash = hashlib.sha256(cls.sequence.prediction_file.read_bytes()).hexdigest()
        cls.payload = convert_predictions(cls.sequence,MODELS)

    def test_all_valid_meshes_and_rectified_projections(self):
        seq = self.sequence
        records = parse_sequence(self.payload,seq.coordinate_system,seq.frame_ids)
        with np.load(seq.prediction_file) as data:
            self.assertEqual(len(records),data['valid'].sum())
            self.assertLess(self.payload['source']['max_vertex_error_m'],1e-6)
            for ci,serial in enumerate(data['serials']):
                cam = next(c for c in seq.cameras.values() if c.serial==serial)
                self.assertEqual(cam.model,'none')
                self.assertFalse(cam.D.any())
                points = data['joints_world'].reshape(-1,3)
                pixels, valid = cam.project(points)
                expected = data['joints_2d'][ci].reshape(-1,2)
                # Verify exported extrinsic direction and rectified K independently
                # against the producer's projections, for every hand/frame/joint.
                np.testing.assert_allclose(pixels[valid],expected[valid],atol=.002,rtol=1e-5)
        for index in (0,len(seq.frames)//2,len(seq.frames)-1):
            self.assertEqual(set(seq.images(index)),set(seq.cameras))

    def test_refuse_wrong_conventions_and_model(self):
        import hocap_adapter
        original = hocap_adapter.read_yaml
        def altered(path):
            value = original(path)
            if Path(path).name=='effective_config.yaml':
                value['mano']['flat_hand_mean']=True
            return value
        with patch('hocap_adapter.read_yaml',side_effect=altered):
            with self.assertRaisesRegex(ValueError,'Unsupported HO-Cap MANO'):
                convert_predictions(self.sequence,MODELS)
        with self.assertRaisesRegex(ValueError,'belonging to'):
            convert_predictions(self.sequence,MODELS,'/tmp/unrelated.npz')
        import manopth.manolayer
        original_layer = manopth.manolayer.ManoLayer
        def wrong_model(*args,**kwargs):
            layer = original_layer(*args,**kwargs)
            if kwargs.get('use_pca'):
                layer.th_hands_mean.add_(.1)
            return layer
        with patch('manopth.manolayer.ManoLayer',side_effect=wrong_model):
            with self.assertRaisesRegex(ValueError,'mesh mismatch'):
                convert_predictions(self.sequence,MODELS)

    @unittest.skipUnless(os.environ.get('DISPLAY'),'Requires display')
    def test_actual_import_edit_save_reload_and_switch(self):
        with tempfile.TemporaryDirectory() as output:
            project = Path(output)/'project'
            manifest = export_project(BUNDLE,MODELS,project)
            args = SimpleNamespace(project=str(manifest),multiview=None,mano_root=str(MODELS),
                output=str(project/'final'),frame_id='0',hand_side='right',
                input=str(project/'predictions/hocap_v1/mano.npz'),device='cpu')
            app = MultiViewApp(args)
            errors=[]
            app.window.report_callback_exception=lambda *e:errors.append(str(e[1]))
            try:
                app.window.update();app.render()
                original=copy.deepcopy(app.original)
                self.assertEqual(app.mano_fit_tool.state,original)
                self.assertEqual(app.document.arrays['valid'].sum(),518)
                for side in ('right','left'):
                    if app.side!=side:app.switch_side()
                    before=copy.deepcopy(app.mano_fit_tool.state)
                    app.inline_buttons[3][1].invoke();app.window.update()
                    self.assertNotEqual(app.mano_fit_tool.state['transl'],before['transl'])
                    app.undo_edit();self.assertEqual(app.mano_fit_tool.state,before)
                    app.controls_notebook.select(1);app.choose_finger('食指');app.window.update()
                    # Invoke the actual inline command adjacent to the index finger.
                    button=next(pair[1] for pair in app.inline_buttons[6:] if pair[1].winfo_ismapped())
                    button.invoke();app.window.update()
                    self.assertNotEqual(app.mano_fit_tool.state['pose'],before['pose'])
                    corrected=copy.deepcopy(app.mano_fit_tool.state)
                    path=app.save()
                    saved = ManoSequence.load(path)
                    self.assertEqual(saved.record(0,side),corrected)
                    app.restore();app._read(path);app.sync_controls();app.render()
                    self.assertEqual(app.mano_fit_tool.state,corrected)
                    app.navigate(1);app.navigate(-1)
                    self.assertEqual(app.mano_fit_tool.state,corrected)
                self.assertFalse(errors,errors)
                self.assertEqual(hashlib.sha256(self.sequence.prediction_file.read_bytes()).hexdigest(),self.source_hash)
            finally:
                if app.pending_render is not None:app.window.after_cancel(app.pending_render)
                app.window.destroy()

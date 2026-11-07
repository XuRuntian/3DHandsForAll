"""Unified I/O and editor tests; synthetic images/geometry are test fixtures only."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from image_manifest import ImageSequence
from multiview_data import CONVENTION
from sequence_store import ManoSequence, metadata, revision
from mano_io import default_parameters


def fixture(root):
    root=Path(root);inputs=root/'inputs';inputs.mkdir()
    cv2.imwrite(str(inputs/'rgb.png'),np.zeros((80,100,3),dtype=np.uint8))
    cameras={f'cam{i:02d}':dict(serial=str(i),intrinsics=dict(width=100,height=80,fx=90,fy=90,ppx=50,ppy=40,model='none',coeffs=[0]*5),T_master_from_camera=np.eye(4).tolist()) for i in range(6)}
    manifest=dict(schema='multiview_images_v1',units='meter',extrinsic_convention=CONVENTION,master_serial='0',cameras=cameras,
        frames=[dict(frame_id=fid,images={c:'rgb.png' for c in cameras}) for fid in ['10','25','80']])
    manifest['frames'][1]['images']['cam01']=None
    path=inputs/'manifest.json';path.write_text(json.dumps(manifest))
    seq=ImageSequence(path)
    records={}
    for fid in ['10','80']:
        for side in ['right','left']:
            records[fid,side]=default_parameters(fid,side,seq.coordinate_system)
    doc=ManoSequence.from_records(records,seq.frame_ids,metadata(seq.coordinate_system,seq.manifest_sha256))
    doc.save(root/'predictions/model_v1')
    return seq,doc


def gui_args(root, model_root='unused'):
    root=Path(root)
    seq,doc=fixture(root)
    return SimpleNamespace(project=str(seq.manifest_path),multiview=None,output=str(root/'final'),
        frame_id='10',hand_side='right',input=str(root/'predictions/model_v1/mano.npz'),
        mano_root=str(model_root),device='cpu')


class SequenceContractTests(unittest.TestCase):
    def test_sparse_valid_shared_shape_and_atomic_save(self):
        with tempfile.TemporaryDirectory() as root:
            seq,doc=fixture(root)
            self.assertIsNone(doc.record(1,'right'))
            state=doc.record(0,'right');state['betas'][0]=.2;state['transl'][0]=.11
            doc.update(0,'right',state)
            self.assertEqual(doc.record(2,'right')['betas'][0],.2)
            self.assertEqual(doc.record(2,'left')['betas'][0],0)
            path=doc.save(Path(root)/'final');before=path.read_bytes()
            state['transl'][0]=.22;doc.update(0,'right',state)
            with patch('sequence_store.os.replace',side_effect=OSError('simulated interruption')):
                with self.assertRaises(OSError):doc.save(path.parent)
            self.assertEqual(path.read_bytes(),before)
            doc.save(path.parent)
            self.assertEqual(sorted(p.name for p in path.parent.iterdir()),['mano.npz','metadata.json'])
            loaded=ManoSequence.load(path,seq.coordinate_system,seq.manifest_sha256)
            np.testing.assert_array_equal(loaded.arrays['valid'],[[True,True],[False,False],[True,True]])
            self.assertEqual(loaded.record(0,'right')['transl'][0],.22)
            subset={k:v[[2,0]] if k not in ('betas',) else v for k,v in loaded.arrays.items()}
            aligned=ManoSequence(subset,loaded.meta).aligned(seq.frame_ids)
            for key in loaded.arrays:np.testing.assert_array_equal(aligned.arrays[key],loaded.arrays[key])

    def test_stale_writer_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            _,doc=fixture(root)
            directory=Path(root)/'final'
            doc.save(directory,expected_revision='missing')
            old=revision(directory)
            state=doc.record(0,'right');state['transl'][0]=1;doc.update(0,'right',state)
            doc.save(directory,expected_revision=old)
            latest=revision(directory)
            with self.assertRaisesRegex(ValueError,'另一个窗口'):
                doc.save(directory,expected_revision=old)
            self.assertEqual(revision(directory),latest)

    def test_reject_incompatible_and_missing_parameters(self):
        with tempfile.TemporaryDirectory() as root:
            seq,doc=fixture(root)
            for key,value in [('stage','monocular'),('units','millimeter'),('rotation','pca')]:
                meta=copy.deepcopy(doc.meta);meta[key]=value
                with self.assertRaises(ValueError):ManoSequence(doc.arrays,meta)
            for field in ['hand_pose','transl','global_orient']:
                arrays=copy.deepcopy(doc.arrays);del arrays[field]
                with self.assertRaises(ValueError):ManoSequence(arrays,doc.meta)
            arrays=copy.deepcopy(doc.arrays);arrays['transl'][0,0,0]=np.nan
            with self.assertRaises(ValueError):ManoSequence(arrays,doc.meta)
            with self.assertRaisesRegex(ValueError,'different image manifest'):
                ManoSequence.load(Path(root)/'predictions/model_v1','master_camera:0','a'*64)
            records={('10','right'):doc.record(0,'right'),('80','right'):doc.record(2,'right')}
            records['80','right']['betas'][0]=1
            with self.assertRaisesRegex(ValueError,'Per-frame shape'):ManoSequence.from_records(records,seq.frame_ids,doc.meta)

    def test_manifest_missing_view_is_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            seq,_=fixture(root)
            self.assertFalse(seq.has_image(1,'cam01'))
            self.assertEqual(len(seq.images(1)),6)
            content=json.loads(seq.manifest_path.read_text())
            content['frames'][0]['images']['cam00']='does-not-exist.jpg'
            seq.manifest_path.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError,'Missing image'):ImageSequence(seq.manifest_path)

    @unittest.skipUnless(os.environ.get('DISPLAY'),'Needs display')
    def test_gui_save_resume_without_predictions_shared_shape_undo_invalid(self):
        from multiview_annotator import MultiViewApp
        from test_multiview_gui import TestGeometry
        with tempfile.TemporaryDirectory() as root,patch('multiview_annotator.CalibratedMano',TestGeometry):
            seq,doc=fixture(root)
            args=SimpleNamespace(project=str(seq.manifest_path),multiview=None,output=str(Path(root)/'final'),input=str(Path(root)/'predictions/model_v1/mano.npz'),frame_id='10',hand_side='right',mano_root='unused',device='cpu')
            app=MultiViewApp(args);errors=[]
            app.window.report_callback_exception=lambda *e:errors.append(str(e[1]))
            def close():
                if app.pending_render is not None:app.window.after_cancel(app.pending_render)
                app.window.destroy()
            try:
                app.window.update()
                app.set_shape(0,.3);app.window.update()
                self.assertEqual(app.document.record(2,'right')['betas'][0],.3)
                app.undo_edit();self.assertEqual(app.document.record(2,'right')['betas'][0],0)
                app.redo_edit();self.assertEqual(app.document.record(2,'right')['betas'][0],.3)
                app.adjust_global_component(3,1);app.window.update()
                expected=copy.deepcopy(app.document.arrays)
                app.save();app.save()
                app.navigate(1);app.window.update();app.render()
                self.assertFalse(app.can_edit());self.assertIsNone(app.vertices)
                app.nudge(0,1);app.set_shape(0,2);app.window.update()
                for key in expected:np.testing.assert_array_equal(app.document.arrays[key],expected[key])
                app.navigate(1);app.window.update()
                self.assertEqual(app.mano_fit_tool.state['betas'][0],.3)
                self.assertFalse(errors,errors)
            finally:close()
            shutil.rmtree(Path(root)/'predictions')
            args.input=None
            # Renaming the entire project preserves relative paths and metadata binding.
            moved=Path(root)/'moved';moved.mkdir()
            shutil.move(str(Path(root)/'inputs'),moved/'inputs');shutil.move(str(Path(root)/'final'),moved/'final')
            args.project=str(moved/'inputs/manifest.json');args.output=str(moved/'final')
            with patch.dict('sys.modules',{'hocap_adapter':None}):
                app=MultiViewApp(args)
                try:
                    app.window.update();app.render()
                    for key in expected:np.testing.assert_array_equal(app.document.arrays[key],expected[key])
                    app.adjust_global_component(3,1);app.save()
                finally:close()

"""Read-only adapter for sync_lab's rectified HO-Cap hand bundle.

This deliberately requires conversion metadata and verifies fitted meshes;
unknown HO-Cap exports are not interpreted by guessing their conventions.
"""
import json
from pathlib import Path
import numpy as np
import yaml
from multiview_data import Camera, MultiViewSequence, CONVENTION
from mano_io import SEQUENCE_SCHEMA, parse_sequence


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text())


class HoCapSequence(MultiViewSequence):
    def __init__(self, root):
        self.root = Path(root).resolve()
        report = json.loads((self.root / 'conversion.json').read_text())
        if (report.get('status') != 'complete' or report.get('rectified') is not True
                or report.get('world_is_master') is not True
                or not np.allclose(report['world_from_master'], np.eye(4), atol=1e-8)):
            raise ValueError('Only complete rectified HO-Cap bundles in master-camera coordinates are supported')
        self.sequence_dir = (self.root / report['sequence']).resolve()
        if not self.sequence_dir.is_relative_to(self.root):
            raise ValueError('Sequence path escapes bundle')
        meta = read_yaml(self.sequence_dir / 'meta.yaml')
        extr = report['source_extrinsics']
        if extr.get('convention') != CONVENTION or extr.get('translation_unit') != 'meter':
            raise ValueError('Unknown HO-Cap extrinsic direction or units')
        self.master_serial = report['master_serial']
        exported = read_yaml(self.root / 'calibration/extrinsics/extrinsics.yaml')
        if exported['rs_master'] != self.master_serial or extr['master_serial'] != self.master_serial:
            raise ValueError('Inconsistent master camera')
        ids = report['source_manifest']['camera_ids']
        serials = meta['realsense']['serials']
        if len(serials) != 6 or set(serials) != set(ids) or len(set(ids.values())) != 6:
            raise ValueError('Expected six distinct calibrated RGB cameras')
        self.cameras = {}
        for serial in sorted(serials, key=lambda s: ids[s]):
            transform = np.asarray(extr['T_master_from_camera'][serial])
            if not np.allclose(np.asarray(exported['extrinsics'][serial]).reshape(3,4), transform[:3], atol=1e-8):
                raise ValueError('Exported camera transform disagrees with conversion provenance')
            intr = read_yaml(self.root / f'calibration/intrinsics/{serial}.yaml')['color']
            if intr['model'] != 'distortion.none' or np.any(intr['coeffs']):
                raise ValueError('Rectified RGB requires distortion-free exported intrinsics')
            camera = Camera(ids[serial], dict(serial=serial, original_calibration={'intrinsics': {'color': intr}}), transform)
            self.cameras[camera.id] = camera
        self.master_id = ids[self.master_serial]
        if not np.allclose(self.cameras[self.master_id].master_from_camera,np.eye(4),atol=1e-5):
            raise ValueError('Master transform is not identity')
        self.coordinate_system = f'master_camera:{self.master_serial}'
        rows = [json.loads(line) for line in (self.sequence_dir/'frames.jsonl').read_text().splitlines() if line.strip()]
        # The converter enumerates bundle frames; source frame IDs remain provenance.
        self.frame_ids = [str(r['frame_id']) for r in rows]
        if self.frame_ids != list(map(str,range(meta['num_frames']))) or len(rows) != report['num_frames']:
            raise ValueError('Missing, duplicate or reordered HO-Cap frames')
        self.frames = []
        for row in rows:
            frame = dict(frame_id=row['frame_id'], cameras={})
            if set(row['source']['cameras']) != set(self.cameras):
                raise ValueError('Incomplete camera group')
            for cid, cam in self.cameras.items():
                if row['source']['cameras'][cid]['serial'] != cam.serial:
                    raise ValueError('Frame camera serial mismatch')
                relative = str((self.sequence_dir/f'{cam.serial}/color_{row["frame_id"]:06d}.jpg').relative_to(self.root))
                self.image_path(relative)
                frame['cameras'][cid] = dict(serial=cam.serial,color_path=relative)
            self.frames.append(frame)
        self.prediction_file = self.sequence_dir/'processed/05_coordinates/mano_world.npz'


def convert_predictions(sequence, model_root, path=None):
    """Expand PCA to absolute axis angles; require full valid-mesh equivalence."""
    import torch
    from manopth.manolayer import ManoLayer
    path = Path(path or sequence.prediction_file).resolve()
    if path != sequence.prediction_file.resolve():
        raise ValueError('Use mano_world.npz belonging to the displayed HO-Cap bundle')
    for stage in ('04_mano_fit','05_coordinates'):
        status = json.loads((sequence.sequence_dir/f'processed/{stage}/status.json').read_text())
        if status.get('status') != 'complete':
            raise ValueError(f'Incomplete HO-Cap stage: {stage}')
    config = read_yaml(sequence.sequence_dir/'processed/04_mano_fit/effective_config.yaml')['mano']
    if config.get('use_pca') is not True or config.get('num_pca_comps') != 45 or config.get('flat_hand_mean') is not False:
        raise ValueError('Unsupported HO-Cap MANO configuration (requires PCA45 + hand mean)')
    with np.load(path,allow_pickle=False) as archive:
        data = {k:archive[k] for k in archive.files}
    hands = data['hands'].tolist()
    count = len(sequence.frames)
    if (not hands or len(set(hands)) != len(hands) or not set(hands) <= {'left','right'}
            or data['poses'].shape != (len(hands),count,51)
            or data['valid'].shape != (len(hands),count) or data['valid'].dtype != np.bool_
            or data['vertices_world'].shape != (len(hands),count,778,3)
            or data['betas'].shape != (10,) or not np.isfinite(data['betas']).all()):
        raise ValueError('Invalid HO-Cap array dimensions or hand identities')
    if str(data['coordinate_unit']) != 'meter' or str(data['world_frame']) != 'master RGB camera':
        raise ValueError('Unknown fitted mesh coordinates or units')
    if data['serials'].tolist() != sorted(c.serial for c in sequence.cameras.values()):
        raise ValueError('Prediction cameras disagree with RGB cameras')
    frames = [dict(frame_id=fid,hands={}) for fid in sequence.frame_ids]
    max_error = 0.
    for hi, side in enumerate(hands):
        indices = np.flatnonzero(data['valid'][hi])
        if not len(indices):
            continue
        poses = data['poses'][hi,indices]
        reference = data['vertices_world'][hi,indices]
        if not np.isfinite(poses).all() or not np.isfinite(reference).all():
            raise ValueError('Non-finite valid MANO prediction')
        layer = ManoLayer(mano_root=str(model_root),side=side,use_pca=True,ncomps=45,
                          flat_hand_mean=False,center_idx=None)
        absolute_layer = ManoLayer(mano_root=str(model_root),side=side,use_pca=False,ncomps=45,
                                   flat_hand_mean=True,center_idx=None)
        # CPU float32 matches the producer. Check the actual converted representation,
        # not just the PCA reconstruction; this detects model/mean/origin mismatches.
        with torch.no_grad():
            p = torch.tensor(poses,dtype=torch.float32)
            local = p[:,3:48] @ layer.th_selected_comps + layer.th_hands_mean
            absolute = torch.cat((p[:,:3],local),dim=1)
            vertices = []
            for start in range(0,len(p),32):
                batch = absolute[start:start+32]
                v,_ = absolute_layer(batch,torch.tensor(data['betas'],dtype=torch.float32).expand(len(batch),-1))
                vertices.append(v.numpy()/1000.+poses[start:start+32,None,48:])
            error = np.linalg.norm(np.concatenate(vertices)-reference,axis=-1).max()
            if not np.isfinite(error) or error > 1e-5:
                raise ValueError(f'MANO conversion mesh mismatch: {error} meters; refusing import')
            max_error = max(max_error,float(error))
        for j,index in enumerate(indices):
            frames[index]['hands'][side] = dict(pose=local[j].tolist(),betas=data['betas'].tolist(),
                global_orient=poses[j,:3].tolist(),transl=poses[j,48:].tolist())
    payload = dict(schema=SEQUENCE_SCHEMA,coordinate_system=sequence.coordinate_system,units='meter',
                   source=dict(estimator='HO-Cap',file=str(path),stage='05_coordinates',
                               conversion='PCA45 + mean -> absolute axis angle',max_vertex_error_m=max_error),
                   frames=[f for f in frames if f['hands']])
    parse_sequence(payload,sequence.coordinate_system,sequence.frame_ids)
    return payload


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Validate and export a rectified HO-Cap hand bundle on CPU')
    parser.add_argument('bundle')
    parser.add_argument('--mano-root',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    sequence = HoCapSequence(args.bundle)
    output = Path(args.output).resolve()
    if output.is_relative_to(sequence.root):
        raise ValueError('Export must be outside the source bundle')
    payload = convert_predictions(sequence,args.mano_root)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:
        json.dump(payload,f,indent=2,allow_nan=False)
    print(json.dumps(payload['source'],indent=2))
    print(f'Exported {len(payload["frames"])} frames to {output}')

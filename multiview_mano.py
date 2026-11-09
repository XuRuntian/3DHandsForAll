"""Reuse manopth and the legacy slider interface, without prediction or fitting."""
from pathlib import Path
import numpy as np
import torch
import params
from mano_io import validate


class CalibratedMano:
    def __init__(self, model_root, state, device='cpu'):
        self.state = validate(state)
        path = Path(model_root) / f'MANO_{state["hand_side"].upper()}.pkl'
        if not path.is_file():
            raise FileNotFoundError(f'Missing licensed MANO model: {path}. Supply --mano-root; no substitute mesh is used.')
        from manopth.manolayer import ManoLayer
        self.device = torch.device(device)
        # Native MANO origin: compatible with standard SMPL/MANO transl semantics.
        self.layer = ManoLayer(mano_root=str(model_root), side=state['hand_side'],
                               use_pca=False, ncomps=45, flat_hand_mean=True,
                               center_idx=None, root_rot_mode='axisang',
                               joint_rot_mode='axisang').to(self.device)
        # Legacy sliders describe offsets from the MANO mean. Keep that UI mapping
        # while the interchange format always stores absolute local axis angles.
        import pickle
        with path.open('rb') as f:
            model = pickle.load(f, encoding='latin1')
        self.mean = np.asarray(model['hands_mean']).reshape(45)
        self.faces = self.layer.th_faces.detach().cpu().numpy().astype(np.int32)
        self.snapshots = {}
        self.suspend_edits = False

    def geometry(self):
        """Vertices and OpenPose21 joints in master-camera meters, from one pass.

        manopth already appends side-specific fingertips and orders the joints
        as wrist, thumb, index, middle, ring, pinky (four joints per finger).
        Do not apply the native MANO-to-OpenPose permutation a second time.
        """
        with torch.no_grad():
            pose = torch.tensor([self.state['global_orient'] + self.state['pose']],
                                dtype=torch.float32, device=self.device)
            shape = torch.tensor([self.state['betas']], dtype=torch.float32, device=self.device)
            vertices, joints = self.layer(pose, shape)
        # manopth emits millimeters; do not use legacy key-bone normalization.
        translation = np.asarray(self.state['transl'])
        return (vertices[0].cpu().numpy() / 1000. + translation,
                joints[0].cpu().numpy() / 1000. + translation)

    def vertices(self):
        return self.geometry()[0]

    def set_input_pose_i(self, i, value):
        if not self.suspend_edits:
            self.state['pose'][i] = float(value + self.mean[i])

    def get_input_pose_i(self, i):
        return self.state['pose'][i] - self.mean[i]

    def set_input_rot_i(self, i, value):
        if not self.suspend_edits:
            self.state['global_orient'][i] = float(value)

    def get_input_rot_i(self, i):
        return self.state['global_orient'][i]

    def set_xyz_root_i(self, i, value):
        if not self.suspend_edits:
            self.state['transl'][i] = float(value) / 100.  # legacy UI centimeters

    def get_xyz_root_i(self, i):
        return self.state['transl'][i] * 100.

    def _snapshot(self, key, field, indices):
        if key in self.snapshots:
            for i, v in zip(indices, self.snapshots.pop(key)):
                self.state[field][i] = v
            return False
        self.snapshots[key] = [self.state[field][i] for i in indices]
        return True

    def toggle_rot_freeze_state(self):
        return self._snapshot('rotation', 'global_orient', range(3))

    def toggle_finger_freeze_state(self, i):
        start, end = params.freeze_fin_idx_range_tuple_list[i]
        return self._snapshot(i, 'pose', range(start, end))

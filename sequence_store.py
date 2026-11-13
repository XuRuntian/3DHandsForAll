"""Model-independent, whole-sequence MANO storage. No GUI or estimator imports."""
import copy
import hashlib
import io
import fcntl
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from mano_io import default_parameters, validate

SIDES = ('right', 'left')
FIELDS = {'hand_pose':45, 'global_orient':3, 'transl':3}


def metadata(coordinate_system, image_manifest_sha256):
    return dict(schema='mano_sequence_v1', stage='multiview_fitted', model='MANO_v1.2',
                hand_order=list(SIDES), rotation='axis_angle_radians',
                hand_pose_convention='absolute_flat_hand_mean',
                translation_convention='native_mano_origin', units='meter',
                axes='x_right_y_down_z_forward', coordinate_system=coordinate_system,
                image_manifest_sha256=image_manifest_sha256)


def revision(directory):
    path=Path(directory)/'mano.npz'
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else 'missing'


class ManoSequence:
    def __init__(self, arrays, meta):
        self.arrays = {k:np.array(v,copy=True) for k,v in arrays.items()}
        self.meta = copy.deepcopy(meta)
        self.validate()

    def validate(self):
        expected = metadata(self.meta.get('coordinate_system'),self.meta.get('image_manifest_sha256'))
        if self.meta != expected or not isinstance(expected['coordinate_system'],str):
            raise ValueError('Unsupported MANO metadata; requires complete multiview fitted parameters')
        validate(default_parameters('0','right',expected['coordinate_system']))
        digest = expected['image_manifest_sha256']
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Missing or invalid image manifest fingerprint')
        a=self.arrays
        if set(a)-{'repair_status'}!={'frame_ids','betas','valid',*FIELDS}:
            raise ValueError('Unexpected or missing MANO arrays')
        ids=a['frame_ids']
        if ids.ndim!=1 or ids.dtype.kind not in 'US' or not len(ids) or len(set(ids.tolist()))!=len(ids):
            raise ValueError('frame_ids must be unique strings')
        for fid in ids:
            validate(default_parameters(str(fid),'right',expected['coordinate_system']))
        n=len(ids)
        if a['valid'].shape!=(n,2) or a['valid'].dtype!=np.bool_:
            raise ValueError('valid must be bool [T,2]')
        if 'repair_status' not in a:
            a['repair_status'] = np.zeros((n, 2), dtype=np.uint8)
        status = a['repair_status']
        if status.shape != (n, 2) or status.dtype != np.uint8 or not np.isin(status, [0, 1, 2]).all():
            raise ValueError('repair_status must be uint8 [T,2]: 0 original, 1 absent, 2 repaired')
        if np.any((status == 1) & a['valid']) or np.any((status == 2) & ~a['valid']):
            raise ValueError('repair_status disagrees with valid')
        for key,shape in [('betas',(2,10))]+[(k,(n,2,d)) for k,d in FIELDS.items()]:
            v=a[key]
            if v.shape!=shape or v.dtype.kind!='f' or not np.isfinite(v).all():
                raise ValueError(f'{key} must be finite floating point {shape}')

    @classmethod
    def from_records(cls,records,frame_ids,meta):
        a=dict(frame_ids=np.asarray(frame_ids,dtype=str),betas=np.zeros((2,10)),
               valid=np.zeros((len(frame_ids),2),dtype=bool))
        a.update({k:np.zeros((len(frame_ids),2,d)) for k,d in FIELDS.items()})
        seen={}
        for (fid,side),state in records.items():
            state=validate(state,meta['coordinate_system'],fid)
            if state['hand_side']!=side or fid not in frame_ids:
                raise ValueError('Record does not match image frame/hand')
            i,j=frame_ids.index(fid),SIDES.index(side)
            if side in seen and not np.allclose(seen[side],state['betas'],rtol=0,atol=1e-8):
                raise ValueError('Per-frame shape is unsupported; supply one shared shape per hand')
            seen[side]=state['betas']
            a['betas'][j]=state['betas'];a['valid'][i,j]=True
            for k in FIELDS:a[k][i,j]=state['pose' if k=='hand_pose' else k]
        return cls(a,meta)

    def record(self,index,side,include_invalid=False):
        j=SIDES.index(side)
        if not include_invalid and not self.arrays['valid'][index,j]:return None
        state=default_parameters(str(self.arrays['frame_ids'][index]),side,self.meta['coordinate_system'])
        state['betas']=self.arrays['betas'][j].tolist()
        for k in FIELDS:state['pose' if k=='hand_pose' else k]=self.arrays[k][index,j].tolist()
        return state

    def update(self,index,side,state):
        state=validate(state,self.meta['coordinate_system'],str(self.arrays['frame_ids'][index]))
        j=SIDES.index(side)
        if state['hand_side']!=side or not self.arrays['valid'][index,j]:
            raise ValueError('Cannot edit an invalid hand instance')
        self.arrays['betas'][j]=state['betas']
        for k in FIELDS:self.arrays[k][index,j]=state['pose' if k=='hand_pose' else k]

    def aligned(self,frame_ids):
        """Input may be sparse/reordered; output follows the complete image index."""
        if not set(self.arrays['frame_ids']).issubset(frame_ids):
            raise ValueError('MANO contains frames absent from image manifest')
        a=dict(frame_ids=np.asarray(frame_ids,dtype=str),betas=self.arrays['betas'].copy(),
               valid=np.zeros((len(frame_ids),2),dtype=bool),
               repair_status=np.zeros((len(frame_ids),2),dtype=np.uint8))
        a.update({k:np.zeros((len(frame_ids),2,d)) for k,d in FIELDS.items()})
        for old,fid in enumerate(self.arrays['frame_ids']):
            new=frame_ids.index(str(fid))
            for k in ['valid','repair_status',*FIELDS]:a[k][new]=self.arrays[k][old]
        return ManoSequence(a,self.meta)

    def reference_indices(self, index, side):
        j = SIDES.index(side)
        return sorted((i for i in range(len(self.arrays['frame_ids']))
                       if i != index and self.arrays['valid'][i, j]),
                      key=lambda i: (abs(i-index), i))

    def mark_absent(self, index, side):
        j = SIDES.index(side)
        self.arrays['valid'][index, j] = False
        self.arrays['repair_status'][index, j] = 1

    def repair_from(self, index, side, reference):
        if reference not in self.reference_indices(index, side):
            raise ValueError('Reference must be another valid frame of the same hand')
        j = SIDES.index(side)
        for key in FIELDS:
            self.arrays[key][index, j] = self.arrays[key][reference, j]
        self.arrays['valid'][index, j] = True
        self.arrays['repair_status'][index, j] = 2
        return self.record(index, side)

    @classmethod
    def load(cls,path,coordinate_system=None,manifest_sha256=None):
        path=Path(path)
        if path.is_dir():path=path/'mano.npz'
        meta=json.loads((path.parent/'metadata.json').read_text())
        raw=path.read_bytes()
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:doc=cls({k:z[k] for k in z.files},meta)
        doc.disk_revision=hashlib.sha256(raw).hexdigest()
        if coordinate_system is not None and meta['coordinate_system']!=coordinate_system:
            raise ValueError('MANO and image coordinate systems differ')
        if manifest_sha256 is not None and meta['image_manifest_sha256']!=manifest_sha256:
            raise ValueError('MANO belongs to a different image manifest')
        return doc

    def save(self,directory,expected_revision=None):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        # A directory lock needs no extra persistent lock file (Linux desktop app).
        fd=os.open(directory,os.O_RDONLY)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX)
            if expected_revision is not None and revision(directory)!=expected_revision:
                raise ValueError('结果已被另一个窗口修改，请先另存到新目录或重新加载，避免覆盖。')
            return self._save_locked(directory)
        finally:
            fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)

    def _save_locked(self,directory):
        self.validate()
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        meta_path=directory/'metadata.json'
        # Metadata is immutable within a result directory. Subsequent saves replace
        # only the NPZ atomically, so there is no two-file revision consistency race.
        if meta_path.exists():
            if json.loads(meta_path.read_text())!=self.meta:
                raise ValueError('Output directory belongs to another sequence or convention')
        else:
            if (directory/'mano.npz').exists():raise ValueError('Output has NPZ without metadata')
            with meta_path.open('x') as f:
                json.dump(self.meta,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        name=None
        try:
            with tempfile.NamedTemporaryFile(dir=directory,suffix='.npz',delete=False) as f:
                name=f.name;np.savez_compressed(f,**self.arrays);f.flush();os.fsync(f.fileno())
            os.replace(name,directory/'mano.npz')
            fd=os.open(directory,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
        finally:
            if name and os.path.exists(name):os.unlink(name)
        return directory/'mano.npz'

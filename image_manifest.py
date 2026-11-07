"""Read-only image/calibration contract, independent of capture and estimation tools."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from multiview_data import Camera, MultiViewSequence, CONVENTION


class ImageSequence(MultiViewSequence):
    def __init__(self,path):
        self.manifest_path=Path(path).resolve()
        self.root=self.manifest_path.parent
        raw=self.manifest_path.read_bytes()
        self.manifest_sha256=hashlib.sha256(raw).hexdigest()
        m=json.loads(raw)
        if m.get('schema')!='multiview_images_v1' or m.get('extrinsic_convention')!=CONVENTION or m.get('units')!='meter':
            raise ValueError('Unknown image manifest schema, extrinsics or units')
        self.master_serial=m['master_serial']
        self.cameras={}
        for cid,value in m['cameras'].items():
            camera=Camera(cid,dict(serial=value['serial'],original_calibration={'intrinsics':{'color':value['intrinsics']}}),value['T_master_from_camera'])
            if any(not isinstance(n,int) or n<=0 for n in camera.size):raise ValueError('Invalid image dimensions')
            if camera.model=='none' and camera.D.any():raise ValueError('Distortion-free images require zero distortion coefficients')
            self.cameras[cid]=camera
        if len(self.cameras)!=6 or len({c.serial for c in self.cameras.values()})!=6:
            raise ValueError('This UI requires six distinct calibrated cameras')
        masters=[c for c in self.cameras.values() if c.serial==self.master_serial]
        if len(masters)!=1 or not np.allclose(masters[0].master_from_camera,np.eye(4),atol=1e-5):
            raise ValueError('Invalid master camera')
        self.master_id=masters[0].id
        self.coordinate_system=f'master_camera:{self.master_serial}'
        self.frames=m['frames'];self.frame_ids=[str(f['frame_id']) for f in self.frames]
        if not self.frames or len(set(self.frame_ids))!=len(self.frames):raise ValueError('Empty/duplicate frame IDs')
        for frame in self.frames:
            if not set(frame['images']).issubset(self.cameras):raise ValueError('Unknown camera in frame')
            for relative in frame['images'].values():
                if relative is not None:self.image_path(relative)

    def image_path(self,relative):
        # Relative ../ paths deliberately support referencing existing project RGB.
        if not isinstance(relative,str) or Path(relative).is_absolute():raise ValueError('Image paths must be relative')
        path=(self.root/relative).resolve()
        if not path.is_file():raise ValueError(f'Missing image: {relative}; use null for explicitly absent views')
        return path

    def images(self,index):
        result={}
        for cid,camera in self.cameras.items():
            relative=self.frames[index]['images'].get(cid)
            if relative is None:
                image=np.zeros((camera.size[1],camera.size[0],3),dtype=np.uint8)
                cv2.putText(image,'Missing image: '+cid,(30,60),cv2.FONT_HERSHEY_SIMPLEX,1,(200,200,200),2)
            else:
                image=cv2.imread(str(self.image_path(relative)))
                if image is None or image.shape[1::-1]!=camera.size:raise ValueError('Image size disagrees with calibration')
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            result[cid]=image
        return result

    def has_image(self,index,cid):
        return self.frames[index]['images'].get(cid) is not None

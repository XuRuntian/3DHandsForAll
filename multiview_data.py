"""Read-only sync_lab RGB adapter; geometry in meters, column-vector extrinsics."""
import json
from pathlib import Path
import cv2
import numpy as np

CONVENTION = 'p_master = T_master_from_camera @ p_camera (homogeneous column vectors)'


def distort(xy, d, model):
    """Match multical/camera_realsense.py and librealsense rs.cpp deprojection.

    Inverse Brown uses radially adjusted tangential coordinates, unlike OpenCV.
    """
    if model == 'none':
        return xy.copy()
    x, y = xy.T
    k1, k2, p1, p2, k3 = d
    r2 = x*x + y*y
    radial = 1 + r2*(k1 + r2*(k2 + r2*k3))
    xq, yq = x*radial, y*radial
    tx, ty = (xq, yq) if model == 'inverse_brown_conrady' else (x, y)
    return np.column_stack((xq + 2*p1*tx*ty + p2*(r2+2*tx*tx),
                            yq + 2*p2*tx*ty + p1*(r2+2*ty*ty)))


class Camera:
    def __init__(self, camera_id, metadata, transform):
        self.id, self.serial = camera_id, metadata['serial']
        intr = metadata['original_calibration']['intrinsics']['color']
        self.size = (intr['width'], intr['height'])
        self.K = np.array([[intr['fx'], 0, intr['ppx']],
                           [0, intr['fy'], intr['ppy']], [0, 0, 1]], dtype=float)
        self.D = np.asarray(intr['coeffs'], dtype=float)
        self.model = intr['model'].removeprefix('distortion.')
        if self.model not in ('none', 'brown_conrady', 'inverse_brown_conrady'):
            raise ValueError(f'Unsupported distortion: {self.model}')
        t = np.asarray(transform, dtype=float)
        if (t.shape != (4, 4) or not np.isfinite(t).all()
                or not np.allclose(t[3], [0, 0, 0, 1])
                or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5)):
            raise ValueError(f'Invalid rigid transform: {camera_id}')
        if (self.D.shape != (5,) or not np.isfinite(self.D).all()
                or not np.isfinite(self.K).all() or min(intr['fx'], intr['fy']) <= 0):
            raise ValueError(f'Invalid intrinsics: {camera_id}')
        self.master_from_camera = t
        self.camera_from_master = np.linalg.inv(t)
        self.R = self.camera_from_master[:3, :3]
        self.t = self.camera_from_master[:3, 3]

    def project(self, vertices):
        xyz = np.asarray(vertices) @ self.R.T + self.t
        valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 1e-4)
        pixels = np.full((len(xyz), 2), np.nan)
        with np.errstate(over='ignore', invalid='ignore'):
            xy = xyz[valid, :2] / xyz[valid, 2:3]
            pixels[valid] = distort(xy, self.D, self.model) * self.K.diagonal()[:2] + self.K[:2, 2]
        valid &= np.isfinite(pixels).all(axis=1)
        return pixels, valid


class MultiViewSequence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        extr = json.loads((self.root / 'calibration/extrinsics.json').read_text())
        if extr.get('convention') != CONVENTION or extr.get('translation_unit') != 'meter':
            raise ValueError('Unknown extrinsic direction or units; refusing to guess')
        self.master_serial = extr['master_serial']
        self.cameras = {}
        for path in sorted(self.root.glob('cam*/calibration.json')):
            meta = json.loads(path.read_text())
            cam = Camera(path.parent.name, meta, extr['T_master_from_camera'][meta['serial']])
            self.cameras[cam.id] = cam
        if len(self.cameras) != 6 or len({c.serial for c in self.cameras.values()}) != 6:
            raise ValueError('Expected six distinct calibrated RGB cameras')
        self.master_id = next(c.id for c in self.cameras.values() if c.serial == self.master_serial)
        if not np.allclose(self.cameras[self.master_id].master_from_camera, np.eye(4), atol=1e-5):
            raise ValueError('Master transform is not identity')
        self.coordinate_system = f'master_camera:{self.master_serial}'
        self.frames = [json.loads(line) for line in (self.root / 'frames.jsonl').read_text().splitlines() if line.strip()]
        self.frame_ids = [str(f['frame_id']) for f in self.frames]
        if not self.frames or len(set(self.frame_ids)) != len(self.frames):
            raise ValueError('Empty or duplicate frame IDs')
        for frame in self.frames:
            if set(frame['cameras']) != set(self.cameras):
                raise ValueError(f'Incomplete synchronized frame: {frame["frame_id"]}')
            for cid, cam in self.cameras.items():
                item = frame['cameras'][cid]
                if item['serial'] != cam.serial:
                    raise ValueError(f'Camera serial mismatch: {cid}')
                self.image_path(item['color_path'])

    def initial_translation(self):
        """Place the default hand near the rig viewing-axis intersection.

        This uses calibration only, not image evidence or hand estimation.
        """
        matrices, targets = [], []
        for camera in self.cameras.values():
            transform = camera.master_from_camera
            direction = transform[:3, 2]
            projector = np.eye(3) - np.outer(direction, direction)
            matrices.append(projector)
            targets.append(projector @ transform[:3, 3])
        point = np.linalg.lstsq(sum(matrices), sum(targets), rcond=None)[0]
        for camera in self.cameras.values():
            uv, valid = camera.project([point])
            if not valid[0] or not ((uv[0] >= 0).all() and (uv[0] < camera.size).all()):
                raise ValueError('Rig viewing axes have no visible default center; supply --input with explicit transl')
        return point.tolist()

    def image_path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError(f'Missing/invalid RGB image: {relative}')
        return path

    def images(self, index):
        images = {}
        for cid, cam in self.cameras.items():
            path = self.image_path(self.frames[index]['cameras'][cid]['color_path'])
            image = cv2.imread(str(path))
            if image is None or image.shape[1::-1] != cam.size:
                raise ValueError(f'Image dimensions disagree with calibration: {path}')
            images[cid] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return images


def open_sequence(root):
    if (Path(root) / 'conversion.json').is_file():
        raise ValueError('HO-Cap 原始包请先使用 hocap_adapter.py 转换，再使用 --project 加载统一输入。')
    return MultiViewSequence(root)


def mesh_overlay(image, camera, vertices, faces, size=None, alpha=.75):
    """Distorted wire mesh on unrectified RGB. No hidden-surface removal."""
    if not 0 <= alpha <= 1:
        raise ValueError('Overlay alpha must be between 0 and 1')
    size = size or camera.size
    out = cv2.resize(image, size)
    if alpha == 0:
        return out
    xy, valid = camera.project(vertices)
    xy *= np.array(size) / np.array(camera.size)
    valid &= (np.abs(xy) < 1e6).all(axis=1)
    triangles = xy[faces[valid[faces].all(axis=1)]]
    if len(triangles):
        layer = out.copy()
        cv2.polylines(layer, np.rint(triangles).astype(np.int32), True, (70, 255, 140), 1, cv2.LINE_AA)
        out = cv2.addWeighted(out, 1-alpha, layer, alpha, 0)
    return out


# OpenPose21 topology, also used by EgoHandKit's parquet_mesh renderer.
HAND_CHAINS = tuple((0, *range(start, start+4)) for start in (1, 5, 9, 13, 17))
HAND_COLORS = ((255, 135, 80), (85, 200, 255), (120, 225, 110),
               (235, 160, 255), (255, 215, 70))  # RGB, one color per finger


def skeleton_overlay(image, camera, joints, size=None, alpha=.75):
    """Project 21 MANO joints and 20 bones onto RGB, without modifying the image."""
    if not 0 <= alpha <= 1:
        raise ValueError('Overlay alpha must be between 0 and 1')
    joints = np.asarray(joints)
    if joints.shape != (21, 3):
        raise ValueError('Expected OpenPose21 joints with shape (21, 3)')
    size = size or camera.size
    out = cv2.resize(image, size)
    if alpha == 0:
        return out
    xy, valid = camera.project(joints)
    xy *= np.array(size) / np.array(camera.size)
    valid &= (np.abs(xy) < 1e6).all(axis=1)
    points = np.zeros((21, 2), dtype=np.int32)
    points[valid] = np.rint(xy[valid]).astype(np.int32)
    layer = out.copy()
    radius = max(2, round(min(size) / 240))
    thickness = max(1, round(min(size) / 360))
    for chain, color in zip(HAND_CHAINS, HAND_COLORS):
        for a, b in zip(chain, chain[1:]):
            if valid[a] and valid[b]:
                cv2.line(layer, tuple(points[a]), tuple(points[b]), color, thickness, cv2.LINE_AA)
    for i in range(21):
        if valid[i]:
            color = (245, 245, 245) if i == 0 else HAND_COLORS[(i-1)//4]
            center = tuple(points[i])
            cv2.circle(layer, center, radius+1, (25, 25, 25), -1, cv2.LINE_AA)
            cv2.circle(layer, center, radius, color, -1, cv2.LINE_AA)
    return cv2.addWeighted(out, 1-alpha, layer, alpha, 0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Read-only six-camera dataset validation (no MANO needed)')
    parser.add_argument('rgbd_root')
    parser.add_argument('--decode-all', action='store_true')
    args = parser.parse_args()
    sequence = MultiViewSequence(args.rgbd_root)
    print(f'{len(sequence.frames)} frames; master {sequence.master_id} / {sequence.master_serial}')
    for camera in sequence.cameras.values():
        print(camera.id, camera.serial, camera.size, camera.model)
    for index in range(len(sequence.frames)) if args.decode_all else [0]:
        sequence.images(index)
    print('RGB paths, serials, calibration and image sizes validated.')

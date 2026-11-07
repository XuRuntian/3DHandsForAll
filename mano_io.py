"""Versioned interchange: absolute MANO axis-angle radians, native origin, meters."""
import copy
import json
from pathlib import Path
import re
import numpy as np


def default_parameters(frame_id, side, coordinate_system):
    return dict(schema_version=1, frame_id=str(frame_id), hand_side=side,
                pose=[0.] * 45, betas=[0.] * 10, global_orient=[0.] * 3,
                transl=[0., 0., .5], coordinate_system=coordinate_system,
                units='meter', mano_model_type='MANO',
                rotation_representation='axis_angle_radians',
                pose_convention='absolute_flat_hand_mean',
                translation_convention='native_mano_origin')


def validate(data, coordinate_system=None, frame_id=None):
    if not isinstance(data, dict):
        raise ValueError('MANO record must be a JSON object')
    data = copy.deepcopy(data)
    if data.get('schema', 'mano_frame_v1') != 'mano_frame_v1':
        raise ValueError('Unsupported single-frame schema')
    required = default_parameters('', 'right', '')
    # Fixed v1 conventions may be omitted by producers. Explicit incompatible
    # values are still rejected; never guess units, PCA or camera transforms.
    conventions = ('schema_version', 'mano_model_type',
                   'rotation_representation', 'pose_convention', 'translation_convention')
    for key in conventions:
        data.setdefault(key, required[key])
    for key in required:
        if key not in data:
            raise ValueError(f'Missing MANO field: {key}')
    for key in ('schema_version', 'units', 'mano_model_type', 'rotation_representation',
                'pose_convention', 'translation_convention'):
        if data[key] != required[key]:
            raise ValueError(f'Unsupported {key}: {data[key]}')
    if data['hand_side'] not in ('left', 'right'):
        raise ValueError('hand_side must be left or right')
    data['frame_id'] = str(data['frame_id'])
    if not re.fullmatch(r'[A-Za-z0-9_-]+', data['frame_id']):
        raise ValueError('Invalid frame_id')
    if not isinstance(data['coordinate_system'], str) or not re.fullmatch(r'master_camera:[A-Za-z0-9_-]+', data['coordinate_system']):
        raise ValueError('Expected master_camera:<serial> coordinate_system')
    if coordinate_system is not None and data['coordinate_system'] != coordinate_system:
        raise ValueError('Input coordinate system differs from the rig master; convert before import')
    if frame_id is not None and data['frame_id'] != str(frame_id):
        raise ValueError('Input frame_id differs from the displayed synchronized frame')
    for key, n in [('pose', 45), ('betas', 10), ('global_orient', 3), ('transl', 3)]:
        value = np.asarray(data[key], dtype=float)
        if key == 'pose' and value.shape == (15, 3):
            value = value.reshape(45)
        if value.shape != (n,) or not np.isfinite(value).all():
            raise ValueError(f'{key} must have shape ({n},) and finite values')
        data[key] = value.tolist()
    return data


def read_annotation(path, coordinate_system=None, frame_id=None):
    payload = json.loads(Path(path).read_text())
    if payload.get('schema') == '3dhands.multiview.annotation.v1':
        original = validate(payload['original'], coordinate_system, frame_id)
        corrected = validate(payload['corrected'], coordinate_system, frame_id)
        if (original['frame_id'], original['hand_side']) != (corrected['frame_id'], corrected['hand_side']):
            raise ValueError('Original and corrected refer to different hands/frames')
        return original, corrected
    state = validate(payload, coordinate_system, frame_id)
    return copy.deepcopy(state), state


def save_revision(directory, original, corrected, source=None):
    original = validate(original)
    corrected = validate(corrected, original['coordinate_system'], original['frame_id'])
    if original['hand_side'] != corrected['hand_side']:
        raise ValueError('Cannot change hand side within an annotation')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(schema='3dhands.multiview.annotation.v1', source=source,
                   original=original, corrected=corrected)
    stem = f'{corrected["frame_id"]}_{corrected["hand_side"]}'
    # Exclusive creation: never replace predictions or earlier corrections.
    for version in range(1, 1000000):
        path = directory / f'{stem}.v{version:04d}.json'
        try:
            with path.open('x') as f:
                json.dump(payload, f, indent=2, allow_nan=False)
                f.write('\n')
            return path
        except FileExistsError:
            continue
    raise RuntimeError('Revision limit reached')


SEQUENCE_SCHEMA = 'mano_multiview_v1'


def parse_sequence(payload, coordinate_system=None, frame_ids=None):
    """Normalize a model-independent sequence into (frame_id, side) records.

    Missing hands are omitted, never filled with zero poses or sentinel values.
    A sequence may contain only selected frames; the UI skips unprovided frames.
    """
    if not isinstance(payload, dict) or payload.get('schema') != SEQUENCE_SCHEMA:
        raise ValueError('Expected schema: mano_multiview_v1')
    frames = payload.get('frames')
    if not isinstance(frames, list) or not frames:
        raise ValueError('frames must be a nonempty list')
    if payload.get('units') != 'meter':
        raise ValueError('Sequence units must explicitly be meter')
    coord = payload.get('coordinate_system')
    allowed_frames = None if frame_ids is None else set(map(str, frame_ids))
    records, seen = {}, set()
    for frame in frames:
        if not isinstance(frame, dict) or 'frame_id' not in frame:
            raise ValueError('Each frame requires frame_id and hands')
        fid = str(frame['frame_id'])
        if fid in seen:
            raise ValueError(f'Duplicate frame_id: {fid}; put both hands in one frame')
        seen.add(fid)
        if allowed_frames is not None and fid not in allowed_frames:
            raise ValueError(f'Frame {fid} is not in the RGB sequence')
        hands = frame.get('hands')
        if not isinstance(hands, dict) or not hands:
            raise ValueError(f'Frame {fid} must contain at least one hand; omit frames with no prediction')
        for side, values in hands.items():
            if side not in ('left', 'right') or not isinstance(values, dict):
                raise ValueError(f'Invalid hand in frame {fid}: {side}')
            for reserved in ('frame_id', 'hand_side', 'coordinate_system', 'units'):
                if reserved in values:
                    raise ValueError(f'{reserved} belongs to frame/sequence metadata, not hands')
            state = dict(values, frame_id=fid, hand_side=side,
                         coordinate_system=coord, units=payload.get('units', 'meter'))
            if 'source' in payload:
                state['prediction_source'] = copy.deepcopy(payload['source'])
            records[(fid, side)] = validate(state, coordinate_system, fid)
    return records


def read_sequence(path, coordinate_system=None, frame_ids=None):
    return parse_sequence(json.loads(Path(path).read_text()), coordinate_system, frame_ids)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Validate unified MANO JSON; no model or GPU required')
    parser.add_argument('input')
    parser.add_argument('--coordinate-system', help='Expected master_camera:<serial>')
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text())
    if payload.get('schema') == SEQUENCE_SCHEMA:
        records = parse_sequence(payload, args.coordinate_system)
        print(f'Valid: {len(records)} hands in {len({k[0] for k in records})} frames')
    else:
        read_annotation(args.input, args.coordinate_system)
        print('Valid: single MANO record / correction')

"""Estimator-independent final shared 3D MANO interchange tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from mano_io import validate, parse_sequence, read_annotation, SEQUENCE_SCHEMA


def hand(x=0.):
    return dict(pose=[[0., 0., 0.] for _ in range(15)], betas=[0.] * 10,
                global_orient=[0., 0., 0.], transl=[x, 0., .8])


def sequence():
    return dict(schema=SEQUENCE_SCHEMA, coordinate_system='master_camera:338522300613', units='meter',
                source='test fixture, not estimated data', frames=[
                    dict(frame_id='0', hands=dict(right=hand(.01), left=hand(-.01))),
                    dict(frame_id='2', hands=dict(right=hand(.02)))])


class PredictionIOTests(unittest.TestCase):
    def test_minimal_single_record_and_legacy_roundtrip(self):
        minimal = dict(hand(), frame_id='0', hand_side='right', units='meter',
                       coordinate_system='master_camera:338522300613')
        normalized = validate(minimal)
        self.assertEqual(len(normalized['pose']), 45)
        self.assertEqual(normalized['pose_convention'], 'absolute_flat_hand_mean')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'input.json'
            p.write_text(json.dumps(minimal))
            self.assertEqual(read_annotation(p), (normalized, normalized))
        self.assertEqual(validate(normalized), normalized)

    def test_two_hands_sparse_frames_and_source_preserved(self):
        payload = sequence()
        snapshot = copy.deepcopy(payload)
        records = parse_sequence(payload, payload['coordinate_system'], ['0','1','2'])
        self.assertEqual(set(records), {('0','right'), ('0','left'), ('2','right')})
        self.assertEqual(records[('2','right')]['transl'][0], .02)
        self.assertEqual(records[('0','left')]['prediction_source'], payload['source'])
        records[('0','left')]['pose'][0] = 2.
        self.assertEqual(payload, snapshot)

    def test_reject_wrong_stage_missing_units_and_inconsistent_ids(self):
        for change in [dict(schema='per_camera_2d_v1'), dict(units='mm'), dict(units=None),
                       dict(coordinate_system='world'), dict(frames=[])]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_sequence(dict(sequence(), **change))
        payload = sequence()
        with self.assertRaises(ValueError):
            parse_sequence(payload, 'master_camera:another')
        with self.assertRaises(ValueError):
            parse_sequence(payload, frame_ids=['0','1'])
        payload['frames'].append(copy.deepcopy(payload['frames'][0]))
        with self.assertRaises(ValueError):
            parse_sequence(payload)

    def test_reject_pca_sentinel_and_invalid_rotations(self):
        for change in [dict(pose=[0.] * 6), dict(pose_convention='pca'),
                       dict(global_orient=[[1,0,0],[0,1,0],[0,0,1]]),
                       dict(transl=[0,0,float('inf')]), dict(hand_side='left')]:
            payload = sequence()
            payload['frames'][0]['hands']['right'].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_sequence(payload)
        record = dict(hand(), frame_id='0', hand_side='right', coordinate_system='master_camera:123')
        with self.assertRaises(ValueError):
            validate(record)  # no implicit unit conversion

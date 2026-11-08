import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from annotation_cli import parse_args


class DatasetArgumentsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.annotations = self.root / "hand_annotations"
        self.manifest = self.create("inputs/manifest.json")
        self.prediction = self.create("predictions/mano.npz")

    def create(self, relative):
        path = self.annotations / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def test_first_open_infers_paths(self):
        args = parse_args([str(self.root)])
        self.assertEqual(args.project, str(self.manifest))
        self.assertEqual(args.input, str(self.prediction))
        self.assertEqual(args.output, str(self.annotations / "final"))
        self.assertEqual(args.mano_root, str(Path.home() / "models/mano"))
        self.assertEqual(args.device, "cpu")

    def test_resume_prefers_final(self):
        final = self.create("final/mano.npz")
        self.prediction.unlink()
        self.assertEqual(parse_args([str(self.root)]).input, str(final))

    def test_explicit_input_overrides_resume(self):
        self.create("final/mano.npz")
        args = parse_args([str(self.root), "--input", str(self.prediction)])
        self.assertEqual(args.input, str(self.prediction))

    def test_output_override_resumes_its_own_results(self):
        final = self.create("custom/mano.npz")
        args = parse_args([str(self.root), "--output", str(final.parent)])
        self.assertEqual(args.input, str(final))

    def test_explicit_paths_remain_supported(self):
        args = parse_args(["--project", str(self.manifest), "--input", str(self.prediction),
                           "--mano-root", str(self.root / "models")])
        self.assertEqual(args.input, str(self.prediction))
        self.assertIsNone(args.output)
        self.assertEqual(args.mano_root, str(self.root / "models"))

    def test_missing_input_is_clear_error(self):
        self.prediction.unlink()
        output = io.StringIO()
        with contextlib.redirect_stderr(output), self.assertRaises(SystemExit):
            parse_args([str(self.root)])
        self.assertIn("MANO input does not exist", output.getvalue())

    def test_missing_manifest_is_clear_error(self):
        self.manifest.unlink()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args([str(self.root)])


if __name__ == "__main__":
    unittest.main()

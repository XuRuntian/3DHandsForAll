"""Resolve the fixed dataset layout without importing the GUI or ML stack."""

import argparse
from pathlib import Path


def parse_args(argv=None, legacy_mano_root=None):
    parser = argparse.ArgumentParser(description="Open a hand annotation dataset.")
    parser.add_argument("data_root", nargs="?", type=Path,
                        help="Dataset root containing hand_annotations/")
    parser.add_argument("--project", help="Override image manifest JSON")
    parser.add_argument("--multiview", metavar="MANIFEST", help="Legacy alias for --project")
    parser.add_argument("--mano-root")
    parser.add_argument("--hand-side", choices=("left", "right"), default="right")
    parser.add_argument("--output", default=None)
    parser.add_argument("--input", help="Explicit initial mano.npz; overrides automatic resume")
    parser.add_argument("--frame-id", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.project and args.multiview:
        parser.error("Use only one of --project and --multiview")
    if args.data_root:
        root = args.data_root.expanduser().resolve()
        if not root.is_dir():
            parser.error(f"Dataset directory does not exist: {root}")
        annotations = root / "hand_annotations"
        if not args.project and not args.multiview:
            args.project = str(annotations / "inputs/manifest.json")
        if not args.output:
            args.output = str(annotations / "final")
        if not args.input:
            saved = Path(args.output).expanduser() / "mano.npz"
            args.input = str(saved if saved.is_file() else annotations / "predictions/mano.npz")
    project_mode = bool(args.project or args.multiview)
    if not args.mano_root:
        args.mano_root = str(Path.home() / "models/mano") if project_mode else legacy_mano_root
    for field in ("project", "multiview", "input", "output", "mano_root"):
        value = getattr(args, field)
        if value:
            setattr(args, field, str(Path(value).expanduser().resolve()))
    if project_mode:
        manifest = Path(args.project or args.multiview)
        if not manifest.is_file():
            parser.error(f"Image manifest does not exist: {manifest}")
        if args.input and not Path(args.input).is_file():
            parser.error(f"MANO input does not exist: {args.input}; finish prediction export or pass --input")
    return args

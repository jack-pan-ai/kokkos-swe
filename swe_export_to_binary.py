#!/usr/bin/env python3
"""Convert SWE HDF5 inputs into binary blobs consumed by the Kokkos example."""

import argparse
from pathlib import Path

import h5py
import numpy as np


def _write_array(path: Path, array, dtype: np.dtype) -> int:
    data = np.asarray(array, dtype=dtype).reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as sink:
        sink.write(np.array([data.size], dtype=np.uint64).tobytes())
        sink.write(data.tobytes(order="C"))
    return int(data.size)


def export(mesh_path: Path, state_path: Path, output_dir: Path, quiet: bool = False) -> None:
    mesh_path = mesh_path.expanduser().resolve()
    state_path = state_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"State file not found: {state_path}")

    datasets = {
        "src": ("src.int64.bin", np.int64, mesh_path),
        "dst": ("dst.int64.bin", np.int64, mesh_path),
        "bcells": ("bcells.int64.bin", np.int64, mesh_path),
        "alpha": ("alpha.float64.bin", np.float64, state_path),
        "area": ("area.float64.bin", np.float64, state_path),
        "sx": ("sx.float64.bin", np.float64, state_path),
        "sy": ("sy.float64.bin", np.float64, state_path),
        "bsx": ("bsx.float64.bin", np.float64, state_path),
        "bsy": ("bsy.float64.bin", np.float64, state_path),
        "h": ("h.float64.bin", np.float64, state_path),
        "x": ("x.float64.bin", np.float64, state_path),
        "y": ("y.float64.bin", np.float64, state_path),
    }

    with h5py.File(mesh_path, "r") as mesh_file, h5py.File(state_path, "r") as state_file:
        for key, (filename, dtype, source) in datasets.items():
            group = mesh_file if source == mesh_path else state_file
            if key not in group:
                if dtype == np.float64 and key in {"bsx", "bsy"}:
                    # Optional boundary buffers: create empty files when absent.
                    _write_array(output_dir / filename, np.empty((0,), dtype=dtype), dtype)
                    if not quiet:
                        print(f"[optional] {key}: 0 entries (absent in HDF5)")
                    continue
                raise KeyError(f"Dataset '{key}' missing from {source}")

            count = _write_array(output_dir / filename, group[key][()], dtype)
            if not quiet:
                print(f"[ok] {key:6s}: {count} entries -> {output_dir / filename}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path, help="triangular mesh HDF5 file")
    parser.add_argument("state", type=Path, help="shallow water state HDF5 file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/swe_binary"),
        help="destination directory (default: %(default)s)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-array logs")
    args = parser.parse_args()

    export(args.mesh, args.state, args.output_dir, quiet=args.quiet)


if __name__ == "__main__":
    main()


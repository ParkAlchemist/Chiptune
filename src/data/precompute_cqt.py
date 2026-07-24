from pathlib import Path
import argparse
import traceback

import torch
from tqdm import tqdm

from src.config import config
from src.data.audio_utils import (
    list_audio_files,
    stable_file_id,
    load_audio,
    pad_or_crop,
    audio_to_cqt_tensor,
    write_jsonl,
)


def precompute_domain(
    domain_name: str,
    source_dir: Path,
    cache_dir: Path,
    rebuild: bool = False,
) -> list[dict]:
    cqt_cfg = config.cqt
    data_cfg = config.data

    cache_dir.mkdir(parents=True, exist_ok=True)

    audio_files = list_audio_files(source_dir, data_cfg.audio)
    records: list[dict] = []

    target_samples = int(cqt_cfg.sample_rate * cqt_cfg.duration)

    for audio_path in tqdm(audio_files, desc=f"Precomputing {domain_name}"):
        file_id = stable_file_id(audio_path, source_dir)
        out_path = cache_dir / f"{file_id}.pt"

        record = {
            "id": file_id,
            "domain": domain_name,
            "source_path": str(audio_path),
            "cache_path": str(out_path),
            "sample_rate": cqt_cfg.sample_rate,
            "duration": cqt_cfg.duration,
            "hop_length": cqt_cfg.hop_length,
            "n_bins": cqt_cfg.n_bins,
            "bins_per_octave": cqt_cfg.bins_per_octave,
        }

        if out_path.exists() and not rebuild:
            records.append(record)
            continue

        try:
            y = load_audio(
                audio_path,
                sample_rate=cqt_cfg.sample_rate,
                mono=cqt_cfg.mono,
                normalize_peak=cqt_cfg.normalize_peak,
            )

            y = pad_or_crop(
                y,
                target_samples=target_samples,
                random_crop=False,
            )

            cqt = audio_to_cqt_tensor(
                y,
                sample_rate=cqt_cfg.sample_rate,
                hop_length=cqt_cfg.hop_length,
                n_bins=cqt_cfg.n_bins,
                bins_per_octave=cqt_cfg.bins_per_octave,
                db_min=cqt_cfg.db_min,
                db_max=cqt_cfg.db_max,
                fmin=cqt_cfg.fmin,
            )

            if data_cfg.cache_dtype == "float16":
                cqt = cqt.half()
            elif data_cfg.cache_dtype == "float32":
                cqt = cqt.float()
            else:
                raise ValueError(f"Unsupported cache_dtype: {data_cfg.cache_dtype}")

            torch.save(
                {
                    "cqt": cqt,
                    "metadata": record,
                },
                out_path,
            )

            records.append(record)

        except Exception as exc:
            record["error"] = repr(exc)
            record["traceback"] = traceback.format_exc()
            records.append(record)

    index_path = cache_dir / "index.jsonl"
    write_jsonl(index_path, records)

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=config.data.dataset_root)
    parser.add_argument("--cache-root", type=Path, default=config.data.cache_root)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    poly_dir = args.dataset_root / config.data.poly_subdir
    chip_dir = args.dataset_root / config.data.chip_subdir

    poly_cache = args.cache_root / config.data.poly_subdir
    chip_cache = args.cache_root / config.data.chip_subdir

    precompute_domain(
        domain_name="poly",
        source_dir=poly_dir,
        cache_dir=poly_cache,
        rebuild=args.rebuild,
    )

    precompute_domain(
        domain_name="chip",
        source_dir=chip_dir,
        cache_dir=chip_cache,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()

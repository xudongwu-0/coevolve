"""Extract official ALFWorld CPU games, retaining source archive identities."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from .driver import ASSETS, UPSTREAM, cpu_env, digest, prepare_runtime


def main():
    data = ASSETS / "data"
    data.mkdir(parents=True, exist_ok=True)
    source_split = UPSTREAM / "data/alfworld/test_set.json"
    destination_split = data / "test_set.json"
    if destination_split.exists() and digest(destination_split) != digest(source_split):
        raise RuntimeError("prepared ALFWorld test split has changed")
    if not destination_split.exists():
        destination_split.write_bytes(source_split.read_bytes())
    with tempfile.TemporaryDirectory(dir=ASSETS) as temporary:
        runtime_root = Path(temporary)
        os.environ.update(cpu_env(runtime_root))
        control_root = runtime_root / "control"
        cohort = prepare_runtime(32, None, control_root)
        shutil.copyfile(control_root / "cohort32.json", ASSETS / "cohort32.json")
        shutil.copyfile(control_root / "task_ids32.json", ASSETS / "task_ids32.json")
        official_paths = json.loads(destination_split.read_text())["mixed"]
        directories = {str(Path(p).parent) for p in official_paths}
        sources = {}
        for name in (
            "json_2.1.1_json.zip",
            "json_2.1.1_pddl.zip",
            "json_2.1.1_tw-pddl.zip",
        ):
            archive = ASSETS / "downloads" / name
            if not archive.is_file():
                raise FileNotFoundError(archive)
            sources[name] = digest(archive)
            with zipfile.ZipFile(archive) as zipped:
                for entry in zipped.infolist():
                    destination = (data / entry.filename).resolve()
                    if not destination.is_relative_to(data.resolve()):
                        raise ValueError(f"unsafe archive path {entry.filename}")
                    if str(Path(entry.filename).parent) in directories:
                        zipped.extract(entry, data)
        from alfworld.info import ALFRED_PDDL_PATH, ALFRED_TWL2_PATH
        logic = data / "logic"
        logic.mkdir(exist_ok=True)
        for src, dst in (
            (ALFRED_PDDL_PATH, "alfred.pddl"),
            (ALFRED_TWL2_PATH, "alfred.twl2"),
        ):
            shutil.copyfile(src, logic / dst)
        missing = [p for p in cohort["task_paths"] if not (data / p).is_file()]
        if missing:
            raise RuntimeError(f"official games missing: {missing[:3]}")
        sources["cohort32_games"] = {
            p: digest(data / p) for p in cohort["task_paths"]
        }
        (ASSETS / "asset_manifest.json").write_text(
            json.dumps(sources, indent=2) + "\n")
        print(
            f"Official CPU games ready: {len(cohort['task_paths'])}; "
            "no THOR/detector download."
        )


if __name__ == "__main__":
    main()

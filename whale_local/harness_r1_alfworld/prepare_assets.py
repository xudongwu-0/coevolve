"""Extract official ALFWorld CPU games, retaining source archive identities."""
import json
import os
from pathlib import Path
import shutil
import zipfile

from .driver import ASSETS, cpu_env, digest, prepare_runtime


def main():
    os.environ.update(cpu_env())
    prepare_runtime()
    official_paths = json.loads((ASSETS / "data/test_set.json").read_text())["mixed"]
    directories = {str(Path(p).parent) for p in official_paths}
    sources = {}
    for name in ("json_2.1.1_json.zip", "json_2.1.1_pddl.zip", "json_2.1.1_tw-pddl.zip"):
        archive = ASSETS / "downloads" / name
        if not archive.is_file():
            raise FileNotFoundError(archive)
        sources[name] = digest(archive)
        with zipfile.ZipFile(archive) as zipped:
            for entry in zipped.infolist():
                destination = (ASSETS / "data" / entry.filename).resolve()
                if not destination.is_relative_to((ASSETS / "data").resolve()):
                    raise ValueError(f"unsafe archive path {entry.filename}")
                if str(Path(entry.filename).parent) in directories:
                    zipped.extract(entry, ASSETS / "data")
    from alfworld.info import ALFRED_PDDL_PATH, ALFRED_TWL2_PATH
    logic = ASSETS / "data/logic"
    logic.mkdir(exist_ok=True)
    for src, dst in ((ALFRED_PDDL_PATH, "alfred.pddl"), (ALFRED_TWL2_PATH, "alfred.twl2")):
        shutil.copyfile(src, logic / dst)
    cohort = json.loads((ASSETS / "cohort32.json").read_text())
    missing = [p for p in cohort["task_paths"] if not (ASSETS / "data" / p).is_file()]
    if missing:
        raise RuntimeError(f"official games missing: {missing[:3]}")
    sources["cohort32_games"] = {p: digest(ASSETS / "data" / p) for p in cohort["task_paths"]}
    (ASSETS / "asset_manifest.json").write_text(json.dumps(sources, indent=2) + "\n")
    print(f"Official CPU games ready: {len(cohort['task_paths'])}; no THOR/detector download.")


if __name__ == "__main__":
    main()

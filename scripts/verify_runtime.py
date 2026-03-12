import importlib
import json
import os
import platform
import subprocess
import sys


REQUIRED_MODULES = [
    "torch",
    "deepspeed",
    "accelerate",
    "transformers",
    "datasets",
    "evaluate",
]


def run(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def get_module_version(name):
    module = importlib.import_module(name)
    return getattr(module, "__version__", "unknown")


def main():
    print("=== Runtime Check ===")
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("Conda env:", os.environ.get("CONDA_DEFAULT_ENV", "N/A"))

    missing = []
    versions = {}
    for mod in REQUIRED_MODULES:
        try:
            versions[mod] = get_module_version(mod)
        except Exception:
            missing.append(mod)

    if missing:
        print("[ERROR] Missing modules:", ", ".join(missing))
        sys.exit(1)

    print("Versions:", json.dumps(versions, ensure_ascii=False))

    import torch  # noqa: E402

    print("Torch CUDA available:", torch.cuda.is_available())
    print("Torch CUDA version:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            print(
                "GPU {}: {} | VRAM {:.1f} GB | CC {}.{}".format(
                    idx,
                    props.name,
                    props.total_memory / (1024**3),
                    props.major,
                    props.minor,
                )
            )

    rc, out, _ = run("git lfs version")
    if rc == 0:
        print("Git LFS:", out)
    else:
        print("Git LFS: not found in current env")

    print("[OK] runtime dependencies look good.")


if __name__ == "__main__":
    main()

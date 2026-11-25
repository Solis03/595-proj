"""Retrieve the pinned upstream inputs used by the experiment."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
CALM = "f6248e9f9a93c20c2059bbb46ce3096ca414ad19"
GSM = "3101c7d5072418e28b9008a6636bde82a006892c"
SCIENCE = "2cbf8318e07b9ece895bb2ae605e71e38d623264"
MATH = "21a5633873b6a120296cce3e2df9d5550074f4a3"
ACL = "d5adc823ff0f80f98c80405ca0ab66c68e684409"
MODELS = {
    "Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
}
SUBJECTS = (
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
)


def url_request(url):
    return Request(url, headers={"User-Agent": "cse595-style-judges/1.0"})


def fetch(url, path, digest=None, size=None):
    if path.is_file() and path.stat().st_size > 0:
        if not size or path.stat().st_size == size:
            return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    for attempt in range(4):
        try:
            with urlopen(url_request(url), timeout=90) as response, partial.open("wb") as stream:
                expected = size or int(response.headers.get("Content-Length", 0))
                shutil.copyfileobj(response, stream, length=4 * 1024 * 1024)
            if expected and partial.stat().st_size != expected:
                raise OSError(f"Incomplete download for {path.name}")
            if digest:
                with partial.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                        raise OSError(f"Checksum mismatch for {path.name}")
            partial.replace(path)
            print(f"Fetched {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)", flush=True)
            return path
        except OSError as error:
            print(f"Download attempt {attempt + 1} for {path.name}: {error}", flush=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def fetch_group(entries):
    entries = list(entries)
    small = [entry for entry in entries if len(entry) < 4 or (entry[3] or 0) <= 128 * 1024**2]
    large = [entry for entry in entries if len(entry) == 4 and (entry[3] or 0) > 128 * 1024**2]
    with ThreadPoolExecutor(max_workers=6) as pool:
        tasks = [pool.submit(fetch, *entry) for entry in small]
        paths = [task.result() for task in tasks]
    for entry in large:
        paths.append(fetch_ranges(*entry))
    return paths


def fetch_ranges(url, path, digest, size):
    if path.is_file() and path.stat().st_size == size:
        return path
    part_size = 32 * 1024**2
    parts = path.with_name(path.name + ".ranges")
    parts.mkdir(parents=True, exist_ok=True)
    bounds = [(start, min(size, start + part_size) - 1) for start in range(0, size, part_size)]
    print(f"Downloading {path.name} in {len(bounds)} verified ranges", flush=True)

    def download_part(bound):
        start, stop = bound
        length = stop - start + 1
        piece = parts / str(start)
        if piece.is_file() and piece.stat().st_size == length:
            return piece
        pending = piece.with_suffix(".part")
        for attempt in range(7):
            try:
                separator = "&" if "?" in url else "?"
                request_url = url if attempt < 2 else f"{url}{separator}range_start={start}"
                req = url_request(request_url)
                req.add_header("Range", f"bytes={start}-{stop}")
                with urlopen(req, timeout=45) as response, pending.open("wb") as stream:
                    if response.status != 206 or response.headers["Content-Range"] != f"bytes {start}-{stop}/{size}":
                        raise OSError(f"Unexpected HTTP response for {start}-{stop}: {response.status}")
                    while block := response.read(256 * 1024):
                        stream.write(block)
                if pending.stat().st_size != length:
                    raise OSError(f"Short HTTP range at offset {start}")
                pending.replace(piece)
                return piece
            except OSError as error:
                if attempt == 6:
                    raise
                if attempt == 2:
                    print(f"Retrying {path.name} at offset {start}: {error}", flush=True)
                time.sleep(min(8, 2 ** attempt))

    pieces = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        for count, piece in enumerate(pool.map(download_part, bounds), 1):
            pieces.append(piece)
            if count % 12 == 0 or count == len(bounds):
                print(f"{path.name} | {count}/{len(bounds)} ranges", flush=True)
    combining = path.with_name(path.name + ".assembling")
    checksum = hashlib.sha256()
    with combining.open("wb") as output:
        for piece in pieces:
            with piece.open("rb") as source:
                while block := source.read(4 * 1024**2):
                    output.write(block)
                    checksum.update(block)
    if combining.stat().st_size != size or (digest and checksum.hexdigest() != digest):
        raise OSError(f"Checksum or file size mismatch for {path.name}")
    combining.replace(path)
    for piece in pieces:
        piece.unlink()
    parts.rmdir()
    print(f"Verified {path.relative_to(ROOT)} ({size:,} bytes)", flush=True)
    return path


def torch_wheel():
    """Download the Windows CUDA wheel separately when a full transfer stalls."""
    index = "https://download.pytorch.org/whl/cu128/torch/"
    name = "torch-2.10.0+cu128-cp312-cp312-win_amd64.whl"
    wheel_url = f"https://download.pytorch.org/whl/cu128/{name.replace('+', '%2B')}"
    with urlopen(url_request(index), timeout=30) as response:
        page = response.read().decode()
    filename = re.escape(name.replace("+", "%2B"))
    match = re.search(rf"{filename}#sha256=([0-9a-f]{{64}})", page, re.I)
    if match is None:
        raise ValueError("Could not verify the CUDA wheel's SHA256 from the PyTorch index")
    request = url_request(wheel_url)
    request.add_header("Range", "bytes=0-0")
    with urlopen(request, timeout=30) as response:
        size = int(response.headers["Content-Range"].split("/")[-1])
    fetch_ranges(wheel_url, ASSETS / "wheels" / name, match[1], size)


def data_entries():
    github = "https://raw.githubusercontent.com"
    yield (f"{github}/Y0oMu/LLM-Judge-Bias-Dataset/{CALM}/base_datasets/fact_related_dataset.json",
           ASSETS / "data" / "calm_fact.json")
    yield (f"{github}/Y0oMu/LLM-Judge-Bias-Dataset/{CALM}/bias_datasets/verbosity_bias_standardized.json",
           ASSETS / "data" / "calm_verbosity.json")
    yield (f"{github}/lupantech/ScienceQA/{SCIENCE}/data/scienceqa/problems.json",
           ASSETS / "data" / "scienceqa.json")
    for split in ("train", "test"):
        yield (f"{github}/openai/grade-school-math/{GSM}/grade_school_math/data/{split}.jsonl",
               ASSETS / "data" / f"gsm_{split}.jsonl")
        for subject in SUBJECTS:
            file = f"{subject}/{split}-00000-of-00001.parquet"
            yield (f"https://huggingface.co/datasets/EleutherAI/hendrycks_math/resolve/{MATH}/{file}?download=true",
                   ASSETS / "data" / "math" / file)


def model_entries(weights):
    allowed = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "vocab.json", "merges.txt", "chat_template.jinja", "special_tokens_map.json"}
    for model, revision in MODELS.items():
        url = f"https://huggingface.co/api/models/Qwen/{model}/revision/{revision}?blobs=true"
        with urlopen(url_request(url), timeout=30) as response:
            metadata = json.load(response)
        if metadata["sha"] != revision:
            raise ValueError(f"Unexpected revision for {model}")
        for item in metadata["siblings"]:
            name = item["rfilename"]
            is_weight = name.endswith(".safetensors") or name == "model.safetensors.index.json"
            if name in allowed or (weights and is_weight):
                yield (f"https://huggingface.co/Qwen/{model}/resolve/{revision}/{name}?download=true",
                       ASSETS / "models" / model / name, item.get("lfs", {}).get("sha256"),
                       item.get("size"))


def report_entries():
    for name in ("acl.sty", "acl_natbib.bst"):
        yield (f"https://raw.githubusercontent.com/acl-org/acl-style-files/{ACL}/{name}",
               ROOT / "work" / "latex" / name)
    version = "0.17.0"
    name = f"tectonic-{version}-x86_64-pc-windows-msvc.zip"
    yield (f"https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40{version}/{name}",
           ROOT / "work" / "latex" / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=("data", "models", "tokenizers", "report", "torch-wheel"))
    group = parser.parse_args().group
    if group == "torch-wheel":
        torch_wheel()
        return
    if group == "data":
        entries = data_entries()
    elif group == "report":
        entries = report_entries()
    else:
        entries = model_entries(weights=group == "models")
    fetch_group(entries)


if __name__ == "__main__":
    main()

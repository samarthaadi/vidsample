"""File listings and download URLs for Hugging Face and ModelScope dataset repos."""
import json
import os
import urllib.parse

from .remote import session

INDEX_DIR = os.environ.get("VIDSAMPLE_INDEX", os.path.join(os.getcwd(), "index"))


def _cached(name, fetch):
    path = os.path.join(INDEX_DIR, "_hosts", name.replace("/", "__") + ".json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    data = fetch()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh)
    return data


def hf_files(repo, revision="main"):
    """{path: size} for every file in a HF dataset repo."""
    def fetch():
        out, url = {}, f"https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true"
        while url:
            r = session().get(url, timeout=60)
            r.raise_for_status()
            for f in r.json():
                if f["type"] == "file":
                    out[f["path"]] = f["size"]
            url = r.links.get("next", {}).get("url")
        return out
    return _cached(f"hf_{repo}_{revision}", fetch)


def hf_url(repo, path, revision="main"):
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(path)}"


def ms_files(repo, revision="master"):
    def fetch():
        out, page = {}, 1
        while True:
            r = session().get(f"https://modelscope.cn/api/v1/datasets/{repo}/repo/tree",
                              params={"Revision": revision, "Root": "/", "Recursive": "true",
                                      "PageNumber": page, "PageSize": 500}, timeout=60)
            r.raise_for_status()
            files = r.json()["Data"]["Files"]
            for f in files:
                if f["Type"] == "blob":
                    out[f["Path"]] = f["Size"]
            if len(files) < 500:
                return out
            page += 1
    return _cached(f"ms_{repo}_{revision}", fetch)


def ms_url(repo, path, revision="master"):
    return (f"https://modelscope.cn/api/v1/datasets/{repo}/repo?"
            + urllib.parse.urlencode({"Revision": revision, "FilePath": path}))

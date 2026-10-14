import os
import json
import urllib.request
import urllib.error
from pathlib import Path

# Load Kaggle credentials from env or ~/.kaggle/kaggle.json
TOKEN = os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_TOKEN", "")
USERNAME = os.environ.get("KAGGLE_USERNAME", "anilanna4678")

if not TOKEN:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.is_file():
        try:
            cfg = json.loads(kaggle_json.read_text(encoding="utf-8"))
            TOKEN = cfg.get("key", "")
            USERNAME = cfg.get("username", USERNAME)
        except Exception:
            pass

if not TOKEN:
    TOKEN = "KGAT_648743b1f9a35012576e291f7f8465ad"

SLUG = "train-rf-detr-agar"

with open("notebooks/train_rfdetr_agar_kaggle.ipynb", "r", encoding="utf-8") as f:
    nb_content = f.read()

payload = {
    "slug": f"{USERNAME}/{SLUG}",
    "newTitle": "Train RF-DETR AGAR",
    "text": nb_content,
    "language": "python",
    "kernelType": "notebook",
    "isPrivate": True,
    "enableGpu": True,
    "enableInternet": True,
    "datasetDataSources": [f"{USERNAME}/agar-dataset"]
}

req = urllib.request.Request(
    "https://www.kaggle.com/api/v1/kernels/push",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    },
    method="POST"
)

try:
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        print("Success! Push response:")
        print(json.dumps(result, indent=2))
except urllib.error.HTTPError as e:
    print(f"HTTPError: {e.code} {e.reason}")
    print(e.read().decode("utf-8"))
except Exception as e:
    print(f"Error: {e}")

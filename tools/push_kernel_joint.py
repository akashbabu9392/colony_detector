import os
import json
import urllib.request
import urllib.error
from pathlib import Path

# Load Kaggle credentials
TOKEN = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY", "")
USERNAME = os.environ.get("KAGGLE_USERNAME", "anilanna4678")

if not TOKEN:
    token_file = Path.home() / ".kaggle" / "access_token"
    if token_file.is_file():
        TOKEN = token_file.read_text(encoding="utf-8").strip()

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

SLUG = "train-rf-detr-joint"

nb_path = Path(__file__).parent.parent / "notebooks" / "train_rfdetr_joint_kaggle.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb_content = f.read()

payload = {
    "slug": f"{USERNAME}/{SLUG}",
    "newTitle": "Train RF-DETR Universal (AGAR + 22022540)",
    "text": nb_content,
    "language": "python",
    "kernelType": "notebook",
    "isPrivate": True,
    "enableGpu": True,
    "enableInternet": True,
    "datasetDataSources": [
        f"{USERNAME}/agar-dataset",
        f"{USERNAME}/colony-dataset-22022540"
    ]
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
        print("Success! Unified joint training kernel pushed to Kaggle:")
        print(json.dumps(result, indent=2))
        print(f"\nKernel URL: https://www.kaggle.com/code/{USERNAME}/{SLUG}")
except urllib.error.HTTPError as e:
    print(f"HTTPError: {e.code} {e.reason}")
    print(e.read().decode("utf-8"))
except Exception as e:
    print(f"Error: {e}")

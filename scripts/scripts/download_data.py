"""Download the IBM Telco Customer Churn sample data into data/telco_churn.csv.gz.

The file is not bundled in this repo because its redistribution terms are not
stated by IBM. See "Data and license" in the README.
"""
import gzip
import hashlib
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
OUT = Path(__file__).resolve().parents[1] / "data" / "telco_churn.csv.gz"


def main() -> None:
    raw = urllib.request.urlopen(URL, timeout=60).read()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise SystemExit("Downloaded file does not match the expected checksum.")
    OUT.parent.mkdir(exist_ok=True)
    with gzip.open(OUT, "wb") as f:
        f.write(raw)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()

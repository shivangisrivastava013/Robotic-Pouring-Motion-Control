import hashlib
import os

CHECKPOINTS = {
    "ppo_pouring.zip": {
        "url": "https://github.com/shivangisrivastava013/Robotic-Pouring-Motion-Control/raw/main/checkpoints/ppo_pouring.zip",
        "sha256": "904e93478c8a579ef25f92ec2cf442efa834b7576acd9ae7a4c7575068fe08cb",
    },
    "sac_pouring.zip": {
        "url": "https://github.com/shivangisrivastava013/Robotic-Pouring-Motion-Control/raw/main/checkpoints/sac_pouring.zip",
        "sha256": "1a9453d8886050f48bab7a4c338eddb4cc1867c676e55dcb110399e82398665f",
    },
}


def verify_sha256(filepath: str, expected_hash: str) -> bool:
    if not os.path.exists(filepath):
        return False
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower() == expected_hash.lower()


def main():
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
    os.makedirs(target_dir, exist_ok=True)

    print("[+] Verifying local model checkpoints...")

    all_ok = True
    for fname, meta in CHECKPOINTS.items():
        fpath = os.path.join(target_dir, fname)
        if verify_sha256(fpath, meta["sha256"]):
            print(f"  [OK] '{fname}' present and verified (SHA-256 match).")
        else:
            print(f"  [!] '{fname}' missing or hash mismatch. Attempting download from repository...")
            try:
                import urllib.request

                urllib.request.urlretrieve(meta["url"], fpath)
                if verify_sha256(fpath, meta["sha256"]):
                    print(f"  [OK] Downloaded and verified '{fname}'.")
                else:
                    print(f"  [!] Downloaded '{fname}' but hash validation failed.")
                    all_ok = False
            except (urllib.error.URLError, OSError) as e:
                print(f"  [FAIL] Failed to download '{fname}': {e}")
                all_ok = False

    if all_ok:
        print("[+] All checkpoints verified successfully.")
    else:
        print("[!] Some checkpoints could not be automatically downloaded/verified.")


if __name__ == "__main__":
    main()

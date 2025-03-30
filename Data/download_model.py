# download_model.py
import gdown
import os

def download_model(file_id, output_path="factor_second.pkl"):
    url = f"https://drive.google.com/uc?id={file_id}"

    if os.path.exists(output_path):
        print(f"[INFO] '{output_path}' already exists. Skipping download.")
        return

    print(f"[INFO] Downloading model from Google Drive to '{output_path}'...")
    gdown.download(url, output_path, quiet=False)
    print(f"[INFO] Download complete.")

if __name__ == "__main__":
    # Replace this with your actual file ID
    FILE_ID = "1808931LWG7T3R1xV3fSoJAhw-81v7o5n"
    download_model(FILE_ID)

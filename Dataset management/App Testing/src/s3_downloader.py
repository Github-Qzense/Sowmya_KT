"""
s3_downloader.py

Utilities to download input/output images from S3 based on a results DataFrame.
"""

from __future__ import annotations

import os
from typing import Optional

from urllib.parse import urlparse

import boto3
from tqdm import tqdm


def configure_aws_from_env(
    access_key_id_env: str = "AWS_ACCESS_KEY_ID",
    secret_access_key_env: str = "AWS_SECRET_ACCESS_KEY",
) -> None:
    """
    Ensure AWS credentials are available via environment variables.

    You can set these in the notebook environment or via
    os.environ before calling this function.
    """
    if access_key_id_env not in os.environ or secret_access_key_env not in os.environ:
        raise RuntimeError(
            f"Missing AWS env vars: {access_key_id_env} / {secret_access_key_env}"
        )


def download_object_from_s3(
    url: str,
    destination_folder: str,
    bucket_name: Optional[str] = None,
) -> None:
    """
    Download a single object from S3 given its full HTTPS URL.

    Parameters
    ----------
    url : str
        Full S3 object HTTPS URL.
    destination_folder : str
        Local folder where the file will be saved.
    bucket_name : str, optional
        Bucket name. If None, uses the bucket in the URL.
    """
    parsed_url = urlparse(url)
    # If bucket_name is not passed, infer from URL host
    if bucket_name is None:
        # host example: mobile-api-results-v2.s3.ap-south-1.amazonaws.com
        bucket_name = parsed_url.netloc.split(".")[0]

    object_key = parsed_url.path.lstrip("/") 
    local_key = object_key.replace(":", "_")  # safer for Windows

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder, exist_ok=True)

    filename = os.path.join(destination_folder, os.path.basename(local_key))

    if os.path.exists(filename):
        return

    s3 = boto3.client("s3")

    try:
        s3.download_file(Bucket=bucket_name, Key=object_key, Filename=filename)
    except Exception as e:
        print("\nError downloading:", url, "->", str(e))


def download_images_to_folder(
    df,
    destination_folder: str = "/kaggle/working/",
    bucket_name: str = "mobile-api-results-v2",
) -> None:
    """
    Download input and output images for each row in the results DataFrame.

    Expected DataFrame columns:
    - column 1: input_image_url
    - column 2: output_image_url
    - column 3: fish1_actual ('G' / 'B')
    - column -3: Species
    - column -4: Date

    Images are stored under:
    - {destination_folder}/input/{Date}/{Species}/{Good|Bad}/
    - {destination_folder}/output/{Date}/{Species}/
    """
    if not os.path.exists(destination_folder):
        print(f"Creating destination folder: {destination_folder}")
        os.makedirs(destination_folder, exist_ok=True)

    label_map = {"G": "Good", "B": "Bad"}

    for _, row in tqdm(df.iterrows(), total=len(df)):
        input_url = row[1]
        output_url = row[2]
        species = row[-3]
        date = row[-4]

        try:
            label = label_map[row[3]]
        except Exception:
            # Skip rows where fish1_actual is not G/B
            continue

        input_folder = os.path.join(destination_folder, "input", date, species, label)
        output_folder = os.path.join(destination_folder, "output", date, species)

        download_object_from_s3(input_url, input_folder, bucket_name)
        download_object_from_s3(output_url, output_folder, bucket_name)
    print("Images downloaded successfully.")

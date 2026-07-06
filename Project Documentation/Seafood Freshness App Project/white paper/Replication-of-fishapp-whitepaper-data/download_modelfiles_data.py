'''
download_modelfiles_data.py

This script downloads the model files and input data for testing purposes from GitHub releases from the Sowmya_KT repository.
we can run this file by using the command: python download_modelfiles_data.py
'''

import os

links = [
    "https://github.com/Github-Qzense/Sowmya_KT/releases/download/v1/mobileapp.models.zip",
    "https://github.com/Github-Qzense/Sowmya_KT/releases/download/v1/Mackerel.Recent.App.Testing.Input.Data.zip",
    "https://github.com/Github-Qzense/Sowmya_KT/releases/download/v1/Sardine.Recent.App.Testing.Input.Data.zip",
]

for link in links:
    filename = link.split("/")[-1]  # Extract file name from URL
    
    print(f"Downloading {filename}...")
    os.system(f"wget -L {link} -O {filename}")
    
    print(f"Unzipping {filename}...")
    os.system(f"unzip -o {filename}")
    
    print(f"Deleting {filename}...")
    os.system(f"rm {filename}")
    
print("All files processed successfully.")
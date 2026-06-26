import os
import shutil
import boto3
from tqdm import tqdm
from datetime import datetime, timedelta

# Set the environment variables
# use your AWS credentials insted of these
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""


# filepath to misclassified.txt file
filepath = r"/content/drive/MyDrive/Sowmya /qZense Dataset/Misclassified_data.txt"
with open(filepath, "r") as f:
    misclassified = [line[:-1] for line in f]


print(len(misclassified))


class S3DataDownloader:
    def __init__(self, bucket_name, download_path):
        """
        Initializes the S3DataDownloader instance.

        Args:
            bucket_name (str): The name of the S3 bucket.
            download_path (str): The local directory to download files to.
        """
        self.bucket_name = bucket_name
        self.download_path = download_path
        # Create an S3 client
        self.s3_client = boto3.client("s3")
        self.grouped_objects = self.group_objects_by_date()

    def list_s3_objects(self, bucket_name=None):
        """
        Lists all objects in the specified S3 bucket.

        Args:
            bucket_name (str): The name of the S3 bucket.

        Returns:
            list: A list of objects in the S3 bucket.
        """
        if bucket_name is None:
            bucket_name = self.bucket_name
        object_list = []
        # Use a paginator to iterate through all the objects in the bucket
        paginator = self.s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket_name)

        for page in page_iterator:
            # Get the list of objects in the current page
            objects = page.get("Contents", [])
            object_list.extend(objects)
        print("\nlist_s3_objects done")
        return object_list

    def group_objects_by_date(self):
        """
        Lists all objects in the S3 bucket & groups them by their last modified date.

        Returns:
            dict: A dictionary where keys are dates and values are lists of object keys.
        """
        objects = self.list_s3_objects()
        grouped_objects = {}
        for obj in objects:
            key = obj["Key"]

            # extracting date from key
            date = (key.split("/")[-1])[:8]
            date = date[:4] + "-" + date[4:6] + "-" + date[6:8]

            if date in grouped_objects:
                grouped_objects[date].append(key)
            else:
                grouped_objects[date] = [key]

        # Sort the dictionary based on date values in the keys
        grouped_objects = dict(
            sorted(grouped_objects.items(), key=lambda item: item[0])
        )
        print("group_objects_by_date done\n")
        return grouped_objects

    def replace_misspelled_folder_names(self, species_name):
        misspelled_folders = {
            "Are": ["Ar", "Are"],
            "Basa": ["Basa", "Basaa"],
            "Barracuda": ["Barcoda", "Barkoda", "Barracoda", "Barracuda"],
            "Bolo": ["Bolo", "Bulo"],
            "Catla": ["Katala", "Katalaa", "Katla"],
            "Croaker": ["Kokor", "Croaker", "Silver croaker"],
            "Chara pona": ["Chara pana"],
            "Demo": ["Demo", "Demo2", "Test", "Trial"],
            "Emperor": [
                "Comprel",
                "Emperor",
                "Emporwel",
                "Empowel",
                "M perl",
                "M preal",
            ],
            "Hilsa": ["Hilsa", "Hilis", "Hilisa"],
            "Lady": ["Lady", "Ledi"],
            "Malabar trevally": [
                "Mabar tavili",
                "Malbhot",
                "Travely",
                "Travaily",
                "Trvili",
                "Trevally",
                "Travelly",
                "Giant trevally",
            ],
            "Needle": ["Needale", "Nidal", "Nidil"],
            "Parsi": ["Parci"],
            "Pearl spot": [
                "(bloch,",
                "Bloch,",
                "Bloch",
                "Pearl spot",
                "Pearls spot",
                "Hols spot",
                "Green chromide",
                "Green chormide",
                "Hals spot",
            ],
            "Sea bass": ["C boss", "C boos", "Siba"],
            "Shol": ["Sholo"],
            "Snapper": ["Sinper", "Sniper"],
            "White snapar": ["White snapper"],
        }

        for key, misspellings in misspelled_folders.items():
            if species_name in misspellings:
                return key
        return species_name

    def download_data(self, date, keys):
        # Create a directory for the date if it doesn't exist
        date_directory = os.path.join(self.download_path, date)
        os.makedirs(date_directory, exist_ok=True)

        # Download each object in the group
        for key in tqdm(keys, desc=f"Downloading {date} data"):
            # Extract fish name and fresh type from the image name
            image_name = key.split("/")[-1]
            try:
                fish_name, fresh_type = image_name.split("_")[-2:]
            except:
                continue
            fresh_type = fresh_type.split(".")[0]
            fish_name = fish_name.capitalize()
            fresh_type = fresh_type.capitalize()

            if fish_name.endswith(" "):
                fish_name = fish_name[:-1]
            if fresh_type.endswith(" "):
                fresh_type = fresh_type[:-1]

            old_fish_name = fish_name

            fish_name = self.replace_misspelled_folder_names(fish_name)

            # if fish_name == "Demo":
            #     continue

            if fish_name not in ["Sardine", "Mackerel", "White prawns"]:
                continue

            if old_fish_name != fish_name:
                duplicate_folder = os.path.join(date_directory, old_fish_name)
                if os.path.exists(duplicate_folder):
                    shutil.rmtree(duplicate_folder)

            # Create a directory structure:
            # date_folder/fish_name_folder/fresh_type_folder
            fish_directory = os.path.join(date_directory, fish_name)
            fresh_directory = os.path.join(fish_directory, fresh_type)
            Misclassified_directory = os.path.join(fresh_directory, "Misclassified")

            if image_name in misclassified:
                os.makedirs(Misclassified_directory, exist_ok=True)
                file_name = os.path.join(Misclassified_directory, image_name)

            else:
                os.makedirs(fresh_directory, exist_ok=True)
                file_name = os.path.join(fresh_directory, image_name)

            # Download the object if it doesn't already exist locally
            if not os.path.exists(file_name):
                self.s3_client.download_file(self.bucket_name, key, file_name)
            else:
                pass
                # print(f"Skipped (already exists): {file_name}")

    def download_daily_data(self):
        for date, keys in self.grouped_objects.items():
            self.download_data(date, keys)

    def download_specific_date_data(self, specific_date):
        valid_dates = [date for date in self.grouped_objects.keys()]
        print(valid_dates)
        if specific_date not in valid_dates:
            print(f"Data is not collected on : {specific_date}")
            return
        keys = self.grouped_objects[specific_date]
        self.download_data(specific_date, keys)

    def download_weekly_data(self, start_date, end_date):
        for date, keys in self.grouped_objects.items():
            if start_date <= date <= end_date:
                self.download_data(date, keys)


# Example usage
bucket_name = "fish-data-collection-v2"
# download_path is the destination folder where we want to save the data
download_path = r"/content/drive/MyDrive/Sowmya /qZense Dataset/S3 Data/Daily Data"

data_downloader = S3DataDownloader(bucket_name, download_path)

# Download all data daily
data_downloader.download_daily_data()

# Date should be in YYYY-MM-DD format

# # Download data for a specific date
# specific_date = '2023-05-23'
# data_downloader.download_specific_date_data(specific_date)

# Download data weekly between two dates
# start_date = '2024-09-30'
# end_date = '2024-10-08'
# data_downloader.download_weekly_data(start_date, end_date)

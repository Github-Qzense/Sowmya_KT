import os
from app_testing_results_analysis import (
    DBConfig,
    load_feedback_table,
    enrich_feedback_dataframe,
    filter_by_species_and_dates,
)
from s3_downloader import configure_aws_from_env, download_images_to_folder

def main():
    # 1. Configure AWS credentials (set env vars first in env or code)
    os.environ["AWS_ACCESS_KEY_ID"] = ""
    os.environ["AWS_SECRET_ACCESS_KEY"] = ""
    configure_aws_from_env()

    # 2. Load and prepare data
    config = DBConfig()
    df_raw = load_feedback_table(config)
    df = enrich_feedback_dataframe(df_raw)

    # 3. Filter the data you want to download
    result_df = filter_by_species_and_dates(
        df,
        species="sardine",
        date_range=("2024-12-02", "2025-01-31"),
    )

    # 4. Download images
    download_images_to_folder(result_df, destination_folder="../raw_data")
    
if __name__ == "__main__":
    main()
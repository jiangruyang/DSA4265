import polars as pl
import urllib.request
import os
from typing import Generator

def get_ACRA_filenames() -> Generator[str, None, None]:
    ''' 
    Get the filenames of the ACRA files.
    
    Returns:
        Generator[str, None, None]: The filenames of the ACRA files.
    '''
    for filename in [
        f"ACRA Information on Corporate Entities ('{chr(i)}').csv"
        for i in range(65, 91) 
    ] + ["ACRA Information on Corporate Entities ('Others').csv"]:
        yield filename

def error_if_ACRA_file_missing(file_dir: str) -> None:
    ''' 
    Raise an error if the ACRA files are missing.
    
    Args:
        file_dir (str): The directory where the ACRA files are located.
    '''
    for filename in get_ACRA_filenames():
        if not os.path.exists(file_dir + filename):
            raise FileNotFoundError(f"File {filename} is missing. Get it from https://data.gov.sg/collections/2/view")

def download_SSIC_definition_file(file_dir: str) -> None:
    ''' 
    Download the SSIC definition file if it doesn't exist.
    
    Args:
        file_dir (str): The directory to save the SSIC definition file.
    '''
    if not os.path.exists(file_dir + "ssic2020-detailed-definitions.xlsx"):
        urllib.request.urlretrieve(
            'https://www.singstat.gov.sg/-/media/files/standards_and_classifications/'
            + 'industrial_classification/ssic2020-detailed-definitions.ashx',
            file_dir + 'ssic2020-detailed-definitions.xlsx'
        )
        print("SSIC definition file downloaded")
    else:
        print("SSIC definition file already exists, using it")

def join_SSIC_on_ACRA(acra_df: pl.DataFrame, ssic_df: pl.DataFrame) -> pl.DataFrame:
    ''' 
    Join the SSIC definition file on the ACRA data.
    
    Args:
        acra_df (pl.DataFrame): The ACRA dataframe.
        ssic_df (pl.DataFrame): The SSIC dataframe.

    Returns:
        pl.DataFrame: The ACRA data with SSIC description.
    '''
    # Convert primary_ssic_code to string
    df = acra_df.with_columns(
        pl.col("primary_ssic_code").cast(pl.Utf8).alias("retrieved_ssic_code")
    )
    
    # Join on retrieved_ssic_code
    df = df.join(
        ssic_df.select([
            pl.col("Singapore Standard Industrial Classification 2020 – Detailed Definitions").alias("retrieved_ssic_code"),
            pl.col("__UNNAMED__1").alias("retrieved_ssic_description"),
        ]),
        on="retrieved_ssic_code",
        how="left"
    )
    
    return df

def fetch_ACRA(file_dir: str) -> pl.DataFrame:
    ''' 
    Fetch the ACRA data and join it with the SSIC definition file.
    
    Args:
        file_dir (str): The directory to save the concatenated ACRA and SSIC data, also where
        the raw ACRA files need to be found.

    Returns:
        pl.DataFrame: The concatenated ACRA data with SSIC description.
    '''
    error_if_ACRA_file_missing(file_dir)
    download_SSIC_definition_file(file_dir)
    
    # Columns to force to string
    schema_overrides = {
        "postal_code": pl.Utf8,
        "level_no": pl.Utf8,
        "unit_no": pl.Utf8,
        "secondary_ssic_code": pl.Utf8
    }

    # Read all files
    df = pl.concat([
        pl.scan_csv(file_dir + file, null_values="na", schema_overrides=schema_overrides)
        for file in get_ACRA_filenames()
    ]).collect()
    
    ssic_df = pl.read_excel(file_dir + "ssic2020-detailed-definitions.xlsx")[3:]
    
    df = join_SSIC_on_ACRA(df, ssic_df)
    
    df.write_csv(file_dir + "ACRA.csv")
    print(f"ACRA + SSIC data written to {file_dir}ACRA.csv")

    return df


if __name__ == "__main__":
    # Define the directory where ACRA files are located and where output will be saved
    # Adjust this path as needed based on your project structure
    file_dir = "data/categorization/"
    
    # Ensure directory exists
    os.makedirs(file_dir, exist_ok=True)
    
    # Fetch and process ACRA data
    acra_df = fetch_ACRA(file_dir)
    
    # Print sample of the data
    print(f"Loaded {acra_df.shape[0]} ACRA records")
    print("Sample data:")
    print(acra_df.head(5))
    
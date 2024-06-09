import tarfile
import zipfile
import sys
import os
import wget
import requests
import pandas as pd
import pickle

os.makedirs("data/", exist_ok=True)

if len(sys.argv) < 2:
    print("Usage: python script_name.py [physio|pm25|national_illness]")
    sys.exit(1)

if sys.argv[1] == "physio":
    url = "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download"
    wget.download(url, out="data")
    with tarfile.open("data/set-a.tar.gz", "r:gz") as t:
        t.extractall(path="data/physio")

elif sys.argv[1] == "pm25":
    url = "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/STMVL-Release.zip"
    urlData = requests.get(url).content
    filename = "data/STMVL-Release.zip"
    with open(filename, mode="wb") as f:
        f.write(urlData)
    with zipfile.ZipFile(filename) as z:
        z.extractall("data/pm25")
        
    def create_normalizer_pm25():
        df = pd.read_csv(
            "./data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
            index_col="datetime",
            parse_dates=True,
        )
        test_month = [3, 6, 9, 12]
        for i in test_month:
            df = df[df.index.month != i]
        mean = df.describe().loc["mean"].values
        std = df.describe().loc["std"].values
        path = "./data/pm25/pm25_meanstd.pk"
        with open(path, "wb") as f:
            pickle.dump([mean, std], f)
    create_normalizer_pm25()

elif sys.argv[1] == "national_illness":
    # Path to the local dataset
    local_dataset_path = "/content/drive/myDrive/data/national_illness.csv"

    if not os.path.exists(local_dataset_path):
        print(f"Dataset not found at {local_dataset_path}")
        sys.exit(1)

    # Copy the local dataset to the working directory
    destination_path = "data/national_illness.csv"
    os.system(f"cp {local_dataset_path} {destination_path}")

    # Load the dataset to verify
    data = pd.read_csv(destination_path)
    print("Dataset loaded successfully:")
    print(data.head())

    # Define a function to create a normalizer for the national illness dataset if needed
    def create_normalizer_national_illness():
        df = pd.read_csv(destination_path, index_col="date", parse_dates=True)
        mean = df.describe().loc["mean"].values
        std = df.describe().loc["std"].values
        path = "./data/national_illness_meanstd.pk"
        with open(path, "wb") as f:
            pickle.dump([mean, std], f)

    create_normalizer_national_illness()
    print("Normalizer created successfully.")
else:
    print("Invalid argument. Please use 'physio', 'pm25', or 'national_illness'.")
    sys.exit(1)

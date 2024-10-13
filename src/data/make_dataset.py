import pandas as pd
from glob import glob

def read_data_from_files(files, data_path):
    """
    Read raw data from CSV files and convert it into two DataFrames: one for Accelerometer and one for Gyroscope.
    """
    acs_df = pd.DataFrame()
    gyro_df = pd.DataFrame()

    acs_set = 1
    gyro_set = 1

    for f in files:
        fs = f.split("-")
        participants = fs[0].replace(data_path, "")
        label = fs[1]
        category = fs[2].rstrip("123").rstrip("_MetaWear_2019")
        
        df = pd.read_csv(f)

        df["participants"] = participants
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in f:
            df["set"] = acs_set
            acs_set += 1
            acs_df = pd.concat([acs_df, df])

        if "Gyroscope" in f:
            df["set"] = gyro_set
            gyro_set += 1
            gyro_df = pd.concat([gyro_df, df])

    return acs_df, gyro_df

def main(input_data_path,output_data_path):
    """
    Convert raw data to interim processed data.
    """
    files = glob(input_data_path + "*.csv")
    acs_df, gyro_df = read_data_from_files(files, input_data_path)

    # Convert epoch (ms) to datetime index
    acs_df.index = pd.to_datetime(acs_df["epoch (ms)"], unit="ms")
    gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")

    # Drop unnecessary columns
    acs_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True)
    gyro_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True)

    # Merge Accelerometer and Gyroscope data on the index
    data_merged = pd.concat([acs_df.iloc[:, :3], gyro_df], axis=1)

    data_merged.columns = [
        "Accelerometer_x",
        "Accelerometer_y",
        "Accelerometer_z",
        "Gyroscope_x",
        "Gyroscope_y",
        "Gyroscope_z",
        "Participants",
        "Label",
        "Category",
        "Set"
    ]

    # Resampling settings
    sampling = {
        "Accelerometer_x": "mean",
        "Accelerometer_y": "mean",
        "Accelerometer_z": "mean",
        "Gyroscope_x": "mean",
        "Gyroscope_y": "mean",
        "Gyroscope_z": "mean",
        "Participants": "last",
        "Label": "last",
        "Category": "last",
        "Set": "last"
    }

    # Resample data in chunks by day, then apply sampling
    days = [g for _, g in data_merged.groupby(pd.Grouper(freq="D"))]
    data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

    # Ensure 'Set' is an integer
    data_resampled["Set"] = data_resampled["Set"].astype(int)

    # Save the processed data
    data_resampled.to_pickle(output_data_path)
    print("Done -- make_dataset.py")

if __name__ == '__main__':
    input_data_path = "././data/raw/MetaMotion/"
    output_data_path="././data/interim/01_Data_Processed.pkl"
    main(input_data_path,output_data_path)

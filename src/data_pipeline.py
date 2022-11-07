from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import os
import copy
import util as util

def read_raw_data(config: dict) -> pd.DataFrame:
    # Create variable to store raw dataset
    raw_dataset = pd.DataFrame()

    # Raw dataset dir
    raw_dataset_dir = config["raw_dataset_dir"]

    # Look and load add CSV files
    for i in tqdm(os.listdir(raw_dataset_dir)):
        raw_dataset = pd.concat([pd.read_csv(raw_dataset_dir + i), raw_dataset])
    
    # Return raw dataset
    return raw_dataset

def check_data(input_data, params, api = False):
    input_data = copy.deepcopy(input_data)
    params = copy.deepcopy(params)

    if not api:
        # Check data types
        assert input_data.select_dtypes("datetime").columns.to_list() == \
            params["datetime_columns"], "an error occurs in datetime column(s)."
        assert input_data.select_dtypes("object").columns.to_list() == \
            params["object_columns"], "an error occurs in object column(s)."
        assert input_data.select_dtypes("int").columns.to_list() == \
            params["int32_columns"], "an error occurs in int32 column(s)."
    else:
        # In case checking data from api
        # Predictor that has object dtype only stasiun
        object_columns = params["object_columns"]
        del object_columns[1:]

        # Max column not used as predictor
        int_columns = params["int32_columns"]
        del int_columns[-1]

        # Check data types
        assert input_data.select_dtypes("object").columns.to_list() == \
            object_columns, "an error occurs in object column(s)."
        assert input_data.select_dtypes("int").columns.to_list() == \
            int_columns, "an error occurs in int32 column(s)."

    assert set(input_data.stasiun).issubset(set(params["range_stasiun"])), \
        "an error occurs in stasiun range."
    assert input_data.pm10.between(params["range_pm10"][0], params["range_pm10"][1]).sum() == \
        len(input_data), "an error occurs in pm10 range."
    assert input_data.pm25.between(params["range_pm25"][0], params["range_pm25"][1]).sum() == \
        len(input_data), "an error occurs in pm25 range."
    assert input_data.so2.between(params["range_so2"][0], params["range_so2"][1]).sum() == \
        len(input_data), "an error occurs in so2 range."
    assert input_data.co.between(params["range_co"][0], params["range_co"][1]).sum() == \
        len(input_data), "an error occurs in co range."
    assert input_data.o3.between(params["range_o3"][0], params["range_o3"][1]).sum() == \
        len(input_data), "an error occurs in o3 range."
    assert input_data.no2.between(params["range_no2"][0], params["range_no2"][1]).sum() == \
        len(input_data), "an error occurs in no2 range."

if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()

    # 2. Read all raw dataset
    raw_dataset = read_raw_data(config_data)

    # 3. Reset index
    raw_dataset.reset_index(
        inplace = True,
        drop = True
    )

    # 4. Save raw dataset
    util.pickle_dump(
        raw_dataset,
        config_data["raw_dataset_path"]
    )

    # 5. Handling variable tanggal
    raw_dataset.tanggal = pd.to_datetime(raw_dataset.tanggal)

    # 6. Handling variable pm10
    raw_dataset.pm10 = raw_dataset.pm10.replace(
        "---",
        -1
    ).astype(int)

    # 7. Handling variable pm25
    raw_dataset.pm25.fillna(
        -1,
        inplace = True
    )
    raw_dataset.pm25 = raw_dataset.pm25.replace(
        "---",
        -1
    ).astype(int)

    # 6. Handling variable so2
    raw_dataset.so2 = raw_dataset.so2.replace(
        "---",
        -1
    ).astype(int)

    # 7. Handling variable co
    raw_dataset.co = raw_dataset.co.replace(
        "---",
        -1
    ).astype(int)

    # 8. Handling variable o3
    raw_dataset.o3 = raw_dataset.o3.replace(
        "---",
        -1
    ).astype(int)

    # 9. Handling variable no2
    raw_dataset.no2 = raw_dataset.no2.replace(
        "---",
        -1
    ).astype(int)

    # 10. Handling variable max
    max_index_trouble = raw_dataset[raw_dataset["max"] == "PM25"].index[0]
    raw_dataset.loc[max_index_trouble, "max"] = 49
    raw_dataset.loc[max_index_trouble, "critical"] = "PM10"
    raw_dataset.loc[max_index_trouble, "categori"] = "BAIK"
    raw_dataset["max"] = raw_dataset["max"].astype(int)

    # 11. Handling variable categori
    raw_dataset.drop(
        index = raw_dataset[raw_dataset.categori == "TIDAK ADA DATA"].index,
        inplace = True
    )
    util.pickle_dump(
        raw_dataset,
        config_data["cleaned_raw_dataset_path"]
    )

    # 12. Check data definition
    check_data(raw_dataset, config_data)

    # 13. Splitting input output
    x = raw_dataset[config_data["predictors"]].copy()
    y = raw_dataset.categori.copy()

    # 14. Splitting train test
    x_train, x_test, \
    y_train, y_test = train_test_split(
        x, y,
        test_size = 0.3,
        random_state = 42,
        stratify = y
    )

    # 15. Splitting test valid
    x_valid, x_test, \
    y_valid, y_test = train_test_split(
        x_test, y_test,
        test_size = 0.5,
        random_state = 42,
        stratify = y_test
    )

    # 16. Save train, valid and test set
    util.pickle_dump(x_train, config_data["train_set_path"][0])
    util.pickle_dump(y_train, config_data["train_set_path"][1])

    util.pickle_dump(x_valid, config_data["valid_set_path"][0])
    util.pickle_dump(y_valid, config_data["valid_set_path"][1])

    util.pickle_dump(x_test, config_data["test_set_path"][0])
    util.pickle_dump(y_test, config_data["test_set_path"][1])
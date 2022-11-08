import pandas as pd
import numpy as np
import util as util
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def load_dataset(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    x_train = util.pickle_load(config_data["train_set_path"][0])
    y_train = util.pickle_load(config_data["train_set_path"][1])

    x_valid = util.pickle_load(config_data["valid_set_path"][0])
    y_valid = util.pickle_load(config_data["valid_set_path"][1])

    x_test = util.pickle_load(config_data["test_set_path"][0])
    y_test = util.pickle_load(config_data["test_set_path"][1])

    # Concatenate x and y each set
    train_set = pd.concat(
        [x_train, y_train],
        axis = 1
    )
    valid_set = pd.concat(
        [x_valid, y_valid],
        axis = 1
    )
    test_set = pd.concat(
        [x_test, y_test],
        axis = 1
    )

    # Return 3 set of data
    return train_set, valid_set, test_set

def join_label_categori(set_data, config_data):
    # Check if label not found in set data
    if config_data["label"] in set_data.columns.to_list():
        # Create copy of set data
        set_data = set_data.copy()

        # Rename sedang to tidak sehat
        set_data.categori.replace(
            config_data["label_categories"][1],
            config_data["label_categories"][2], inplace = True
        )

        # Renam tidak sehat to tidak baik
        set_data.categori.replace(
            config_data["label_categories"][2],
            config_data["label_categories_new"][1], inplace = True
        )

        # Return renamed set data
        return set_data
    else:
        raise RuntimeError("Kolom label tidak terdeteksi pada set data yang diberikan!")

def nan_detector(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Replace -1 with NaN
    set_data.replace(
        -1, np.nan,
        inplace = True
    )

    # Return replaced set data
    return set_data

def ohe_fit(data_tobe_fitted: dict, ohe_path: str) -> OneHotEncoder:
    # Create ohe object
    ohe_statiun = OneHotEncoder(sparse = False)

    # Fit ohe
    ohe_statiun.fit(np.array(data_tobe_fitted).reshape(-1, 1))

    # Save ohe object
    util.pickle_dump(
        ohe_statiun,
        ohe_path
    )

    # Return trained ohe
    return ohe_statiun

def ohe_transform(set_data: pd.DataFrame, tranformed_column: str, ohe_statiun: OneHotEncoder) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Transform variable stasiun of set data, resulting array
    stasiun_features = ohe_statiun.transform(np.array(set_data[tranformed_column].to_list()).reshape(-1, 1))

    # Convert to dataframe
    stasiun_features = pd.DataFrame(
        stasiun_features,
        columns = list(ohe_statiun.categories_[0])
    )

    # Set index by original set data index
    stasiun_features.set_index(
        set_data.index,
        inplace = True
    )

    # Concatenate new features with original set data
    set_data = pd.concat(
        [stasiun_features, set_data],
        axis = 1
    )

    # Drop stasiun column
    set_data.drop(
        columns = "stasiun",
        inplace = True
    )

    # Convert columns type to string
    new_col = [str(col_name) for col_name in set_data.columns.to_list()]
    set_data.columns = new_col

    # Return new feature engineered set data
    return set_data

def rus_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Create sampling object
    rus = RandomUnderSampler(random_state = 26)

    # Balancing set data
    x_rus, y_rus = rus.fit_resample(
        set_data.drop("categori", axis = 1),
        set_data.categori
    )

    # Concatenate balanced data
    set_data_rus = pd.concat(
        [x_rus, y_rus],
        axis = 1
    )

    # Return balanced data
    return set_data_rus

def ros_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Create sampling object
    ros = RandomOverSampler(random_state = 11)

    # Balancing set data
    x_ros, y_ros = ros.fit_resample(
        set_data.drop("categori", axis = 1),
        set_data.categori
    )

    # Concatenate balanced data
    set_data_ros = pd.concat(
        [x_ros, y_ros],
        axis = 1
    )

    # Return balanced data
    return set_data_ros

def sm_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Create sampling object
    sm = SMOTE(random_state = 112)

    # Balancing set data
    x_sm, y_sm = sm.fit_resample(
        set_data.drop("categori", axis = 1),
        set_data.categori
    )

    # Concatenate balanced data
    set_data_sm = pd.concat(
        [x_sm, y_sm],
        axis = 1
    )

    # Return balanced data
    return set_data_sm

def le_fit(data_tobe_fitted: dict, le_path: str) -> LabelEncoder:
    # Create le object
    le_encoder = LabelEncoder()

    # Fit le
    le_encoder.fit(data_tobe_fitted)

    # Save le object
    util.pickle_dump(
        le_encoder,
        le_path
    )

    # Return trained le
    return le_encoder

def le_transform(label_data: pd.Series, config_data: dict) -> pd.Series:
    # Create copy of label_data
    label_data = label_data.copy()

    # Load le encoder
    le_encoder = util.pickle_load(config_data["le_encoder_path"])

    # If categories both label data and trained le matched
    if len(set(label_data.unique()) - set(le_encoder.classes_) | set(le_encoder.classes_) - set(label_data.unique())) == 0:
        # Transform label data
        label_data = le_encoder.transform(label_data)
    else:
        raise RuntimeError("Check category in label data and label encoder.")
    
    # Return transformed label data
    return label_data

if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()

    # 2. Load dataset
    train_set, valid_set, test_set = load_dataset(config_data)

    # 3. Join label categories
    train_set = join_label_categori(
        train_set,
        config_data
    )
    valid_set = join_label_categori(
        valid_set,
        config_data
    )
    test_set = join_label_categori(
        test_set,
        config_data
    )

    # 4. Converting -1 to NaN
    train_set = nan_detector(train_set)
    valid_set = nan_detector(valid_set)
    test_set = nan_detector(test_set)

    # 5. Handilng NaN pm10
    # 5.1. Train set
    train_set.loc[train_set[(train_set.categori == "BAIK") & \
    (train_set.pm10.isnull() == True)].index, "pm10"] = \
    config_data["missing_value_pm10"]["BAIK"]

    train_set.loc[train_set[(train_set.categori == "TIDAK BAIK") & \
    (train_set.pm10.isnull() == True)].index, "pm10"] = \
    config_data["missing_value_pm10"]["TIDAK BAIK"]

    # 5.2. Validation set
    valid_set.loc[valid_set[(valid_set.categori == "BAIK") & \
    (valid_set.pm10.isnull() == True)].index, "pm10"] = \
    config_data["missing_value_pm10"]["BAIK"]

    valid_set.loc[valid_set[(valid_set.categori == "TIDAK BAIK") & \
    (valid_set.pm10.isnull() == True)].index, "pm10"] = \
    config_data["missing_value_pm10"]["TIDAK BAIK"]

    # 5.3. Test set
    test_set.loc[test_set[(test_set.categori == "BAIK") & \
    (test_set.pm10.isnull() == True)].index, "pm10"] = \
    config_data["missing_value_pm10"]["BAIK"]

    test_set.loc[test_set[(test_set.categori == "TIDAK BAIK") & \
    (test_set.pm10.isnull() == True)].index, "pm10"] = \
    config_data["missing_value_pm10"]["TIDAK BAIK"]

    # 6. Handling NaN pm25
    # 6.1. Train set
    train_set.loc[train_set[(train_set.categori == "BAIK") & \
    (train_set.pm25.isnull() == True)].index, "pm25"] = \
    config_data["missing_value_pm25"]["BAIK"]

    train_set.loc[train_set[(train_set.categori == "TIDAK BAIK") & \
    (train_set.pm25.isnull() == True)].index, "pm25"] = \
    config_data["missing_value_pm25"]["TIDAK BAIK"]

    # 6.2. Validation set
    valid_set.loc[valid_set[(valid_set.categori == "BAIK") & \
    (valid_set.pm25.isnull() == True)].index, "pm25"] = \
    config_data["missing_value_pm25"]["BAIK"]

    valid_set.loc[valid_set[(valid_set.categori == "TIDAK BAIK") & \
    (valid_set.pm25.isnull() == True)].index, "pm25"] = \
    config_data["missing_value_pm25"]["TIDAK BAIK"]

    # 6.3. Test set
    test_set.loc[test_set[(test_set.categori == "BAIK") & \
    (test_set.pm25.isnull() == True)].index, "pm25"] = \
    config_data["missing_value_pm25"]["BAIK"]

    test_set.loc[test_set[(test_set.categori == "TIDAK BAIK") & \
    (test_set.pm25.isnull() == True)].index, "pm25"] = \
    config_data["missing_value_pm25"]["TIDAK BAIK"]

    # 7. Handling Nan so2, co, o3, and no2
    impute_values = {
        "so2" : config_data["missing_value_so2"],
        "co" : config_data["missing_value_co"],
        "o3" : config_data["missing_value_o3"],
        "no2" : config_data["missing_value_no2"]
    }

    train_set.fillna(
        value = impute_values,
        inplace = True
    )
    valid_set.fillna(
        value = impute_values,
        inplace = True
    )
    test_set.fillna(
        value = impute_values,
        inplace = True
    )

    # 8. Fit ohe with predefined stasiun data
    ohe_stasiun = ohe_fit(
        config_data["range_stasiun"],
        config_data["ohe_stasiun_path"]
    )

    # 9. Transform stasiun on train, valid, and test set
    train_set = ohe_transform(
        train_set,
        "stasiun",
        ohe_stasiun
    )

    valid_set = ohe_transform(
        valid_set,
        "stasiun",
        ohe_stasiun
    )

    test_set = ohe_transform(
        test_set,
        "stasiun",
        ohe_stasiun
    )

    # 10. Undersampling dataset
    train_set_rus = rus_fit_resample(train_set)

    # 11. Oversampling dataset
    train_set_ros = ros_fit_resample(train_set)

    # 12. SMOTE dataset
    train_set_sm = sm_fit_resample(train_set)

    # 13. Fit label encoder
    le_encoder = le_fit(
        config_data["label_categories_new"],
        config_data["le_encoder_path"]
    )

    # 14. Label encoding undersampling set
    train_set_rus.categori = le_transform(
        train_set_rus.categori, 
        config_data
    )

    # 15. Label encoding overrsampling set
    train_set_ros.categori = le_transform(
        train_set_ros.categori,
        config_data
    )

    # 16. Label encoding smote set
    train_set_sm.categori = le_transform(
        train_set_sm.categori,
        config_data
    )

    # 17. Label encoding validation set
    valid_set.categori = le_transform(
        valid_set.categori,
        config_data
    )

    # 18. Label encoding test set
    test_set.categori = le_transform(
        test_set.categori,
        config_data
    )

    # 19. Dumping dataset
    x_train = {
        "Undersampling" : train_set_rus.drop(columns = "categori"),
        "Oversampling" : train_set_ros.drop(columns = "categori"),
        "SMOTE" : train_set_sm.drop(columns = "categori")
    }

    y_train = {
        "Undersampling" : train_set_rus.categori,
        "Oversampling" : train_set_ros.categori,
        "SMOTE" : train_set_sm.categori
    }

    util.pickle_dump(
        x_train,
        "data/processed/x_train_feng.pkl"
    )
    util.pickle_dump(
        y_train,
        "data/processed/y_train_feng.pkl"
    )

    util.pickle_dump(
        valid_set.drop(columns = "categori"),
        "data/processed/x_valid_feng.pkl"
    )
    util.pickle_dump(
        valid_set.categori,
        "data/processed/y_valid_feng.pkl"
    )

    util.pickle_dump(
        test_set.drop(columns = "categori"),
        "data/processed/x_test_feng.pkl"
    )
    util.pickle_dump(
        test_set.categori,
        "data/processed/y_test_feng.pkl"
    )
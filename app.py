import pandas as pd
from sklearn.impute import KNNImputer

# loading in data
base_path = './data/'
ufc_event_data = pd.read_csv(base_path + 'ufc_event_data.csv')
ufc_fight_data = pd.read_csv(base_path + 'ufc_fight_data.csv')
ufc_fight_stat_data = pd.read_csv(base_path + 'ufc_fight_stat_data.csv')
ufc_fighter_data = pd.read_csv(base_path + 'ufc_fighter_data.csv')


def clean_ufc_fighter_data(filepath):
    # Load data
    ufc_fighter_data = pd.read_csv(filepath)

    # Display initial missing data counts
    print("Initial missing values in UFC Fighter Data:")
    print(ufc_fighter_data.isnull().sum())

    # Handle missing values
    # Dropping rows with missing last names
    ufc_fighter_data.dropna(subset=['fighter_l_name'], inplace=True)

    # Filling nicknames with 'No Nickname'
    ufc_fighter_data['fighter_nickname'].fillna('No Nickname', inplace=True)

    # Filling missing height and weight with the mean
    average_height = ufc_fighter_data['fighter_height_cm'].mean()
    average_weight = ufc_fighter_data['fighter_weight_lbs'].mean()
    ufc_fighter_data['fighter_height_cm'].fillna(average_height, inplace=True)
    ufc_fighter_data['fighter_weight_lbs'].fillna(average_weight, inplace=True)

    # Imputing missing reach using KNN based on height and weight
    imputer = KNNImputer(n_neighbors=5)
    ufc_fighter_data[['fighter_height_cm', 'fighter_weight_lbs', 'fighter_reach_cm']] = imputer.fit_transform(
        ufc_fighter_data[['fighter_height_cm', 'fighter_weight_lbs', 'fighter_reach_cm']]
    )

    # Filling missing stance with the mode
    common_stance = ufc_fighter_data['fighter_stance'].mode()[0]
    ufc_fighter_data['fighter_stance'].fillna(common_stance, inplace=True)

    # Dropping rows with missing date of birth
    ufc_fighter_data.dropna(subset=['fighter_dob'], inplace=True)

    # Setting missing NC/DQ to 0
    ufc_fighter_data['fighter_nc_dq'].fillna(0, inplace=True)

    # Remove duplicates
    ufc_fighter_data.drop_duplicates(inplace=True)

    # Verify data cleaning
    print("\nAfter cleaning, missing values in UFC Fighter Data:")
    print(ufc_fighter_data.isnull().sum())

    # Display cleaned data summary
    print("\nCleaned UFC Fighter Data sample:")
    print(ufc_fighter_data.head())

    return ufc_fighter_data

# Usage example
cleaned_ufc_fighter_data = clean_ufc_fighter_data(base_path + 'ufc_fighter_data.csv')

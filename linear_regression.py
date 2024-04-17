import pandas as pd
import statsmodels.api as sm
from datetime import datetime

def load_data(base_path):
    """ Load data from CSV files. """
    fighter_data = pd.read_csv(base_path + 'ufc_fighter_data.csv')
    fight_data = pd.read_csv(base_path + 'ufc_fight_data.csv')
    event_data = pd.read_csv(base_path + 'ufc_event_data.csv')
    return fighter_data, fight_data, event_data
def prepare_data(fighter_data, fight_data, event_data):
    """ Prepare and merge data for analysis. """
    # Convert event dates to datetime
    event_data['event_date'] = pd.to_datetime(event_data['event_date'])
    
    # Merge fight data with event dates
    fight_data = pd.merge(fight_data, event_data[['event_id', 'event_date']], on='event_id', how='left')
    
    # Calculate first and last fight dates
    fight_dates = fight_data.groupby('f_1')['event_date'].agg(first_fight='min', last_fight='max').reset_index()
    fight_dates.rename(columns={'f_1': 'fighter_id'}, inplace=True)
    
    # Merge with fighter data
    fighter_data = pd.merge(fighter_data, fight_dates, on='fighter_id', how='left')
    
    # Calculate career length in years
    fighter_data['career_length_years'] = (fighter_data['last_fight'] - fighter_data['first_fight']).dt.days / 365.25
    
    # Dummy encoding for categorical variables
    if 'fighter_stance' in fighter_data.columns:
        fighter_data = pd.get_dummies(fighter_data, columns=['fighter_stance'], drop_first=True)
    else:
        print("Warning: 'fighter_stance' column not found in fighter_data.")
    
    return fighter_data


def run_regression_analysis(fighter_data):
    """ Run linear regression to analyze factors affecting career length. """
    # Select features and target
    X = fighter_data[['fighter_height_cm', 'fighter_reach_cm'] + [col for col in fighter_data.columns if 'fighter_stance' in col or 'fighter_weight' in col]]
    y = fighter_data['career_length_years']
    
    # Add a constant to the model (intercept)
    X = sm.add_constant(X)
    
    # Handle missing values
    X.fillna(X.mean(), inplace=True)
    y.fillna(y.mean(), inplace=True)
    
    # Fit the linear regression model
    model = sm.OLS(y, X).fit()
    
    # Print the summary of the regression
    return model.summary()

def main():
    base_path = './data/'
    fighter_data, fight_data, event_data = load_data(base_path)
    fighter_data = prepare_data(fighter_data, fight_data, event_data)
    result = run_regression_analysis(fighter_data)
    print(result)
    

if __name__ == "__main__":
    main()

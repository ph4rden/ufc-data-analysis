import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime

# Load and prepare data
def load_data(base_path):
    event_data = pd.read_csv(base_path + 'ufc_event_data.csv')
    fight_data = pd.read_csv(base_path + 'ufc_fight_data.csv')
    fighter_data = pd.read_csv(base_path + 'ufc_fighter_data.csv')
    fight_stat_data = pd.read_csv(base_path + 'ufc_event_data.csv')
    return event_data, fight_data, fighter_data, fight_stat_data

def preprocess_event_data(event_data):
    event_data['event_date'] = pd.to_datetime(event_data['event_date'])
    return event_data

def calculate_fighter_age_and_career_length(fight_data, event_data, fighter_data):
    fight_data = pd.merge(fight_data, event_data[['event_id', 'event_date']], on='event_id', how='left')
    fights_f1 = fight_data[['f_1', 'event_date']].rename(columns={'f_1': 'fighter_id'})
    fights_f2 = fight_data[['f_2', 'event_date']].rename(columns={'f_2': 'fighter_id'})
    all_fights = pd.concat([fights_f1, fights_f2])

    last_fights = all_fights.groupby('fighter_id')['event_date'].max().reset_index().rename(columns={'event_date': 'event_date_last'})
    first_fights = all_fights.groupby('fighter_id')['event_date'].min().reset_index().rename(columns={'event_date': 'event_date_first'})

    fighter_data = pd.merge(fighter_data, last_fights, on='fighter_id', how='left')
    fighter_data = pd.merge(fighter_data, first_fights, on='fighter_id', how='left')

    fighter_data['fighter_dob'] = pd.to_datetime(fighter_data['fighter_dob'])
    fighter_data['age_at_last_fight'] = (fighter_data['event_date_last'].dt.year - fighter_data['fighter_dob'].dt.year)
    fighter_data['career_length_years'] = (fighter_data['event_date_last'] - fighter_data['event_date_first']).dt.days / 365.25
    fighter_data['age_at_debut'] = (fighter_data['event_date_first'] - fighter_data['fighter_dob']).dt.days / 365.25

    return fighter_data

# Regression analysis
def run_regression(data):
    X = data[['age_at_debut']]  # Predictor variable
    y = data['career_length_years']  # Dependent variable
    X = sm.add_constant(X)  # Adding a constant (intercept)
    model = sm.OLS(y, X, missing='drop').fit()  # Fit the model
    return model.summary()

# Visualization
def plot_histogram(data, column, bins=20, kde=False, title="", xlabel="", ylabel=""):
    sns.histplot(data=data, x=column, bins=bins, kde=kde)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Main function
def main():
    base_path = './data/'
    event_data, fight_data, fighter_data, fight_stat_data = load_data(base_path)
    event_data = preprocess_event_data(event_data)
    fighter_data = calculate_fighter_age_and_career_length(fight_data, event_data, fighter_data)

    print("\nMissing or Incorrect Data in Final Dataset:")
    print(fighter_data[['fighter_id', 'event_date_last', 'age_at_last_fight', 'career_length_years', 'age_at_debut']].isnull().sum())
    print("\nSample Data with Ages and Career Lengths at Last Fight:")
    print(fighter_data[['fighter_id', 'event_date_last', 'age_at_last_fight', 'career_length_years', 'age_at_debut']].head())

    # Plotting histograms
    plot_histogram(fighter_data, 'age_at_last_fight', bins=20, kde=True, title='Distribution of Fighter Ages at Last Fight', xlabel='Age at Last Fight', ylabel='Frequency')
    plot_histogram(fighter_data, 'career_length_years', bins=20, kde=True, title='Distribution of UFC Career Lengths', xlabel='Career Length (years)', ylabel='Frequency')

    # Perform regression analysis on age at debut
    results = run_regression(fighter_data)
    print("\nRegression Analysis Results:")
    print(results)

if __name__ == "__main__":
    main()
    """
        Dependent Variable: career_length_years
        Independent Variable: age_at_debut

        The regression analysis reveals a significant negative relationship between the age at UFC debut and 
        the career length of fighters, indicating that fighters who begin their UFC careers at younger ages 
        tend to have longer careers. Specifically, for each additional year in age at debut, the expected career 
        length decreases by approximately 0.233 years (about 2.8 months). Despite this statistically significant finding, 
        the model explains only 6.5% of the variance in career lengths (R-squared = 0.065), suggesting that other 
        unmodeled factors also play crucial roles in influencing a fighter’s career duration. 
        This analysis underscores the importance of considering multiple aspects when assessing career longevity in 
        professional athletes, particularly in combat sports like UFC, where age at debut impacts, 
        but does not fully determine, the length of an athlete's career.
        
    """

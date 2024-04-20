import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime

# Load and prepare data
def load_data(base_path):
    event_data = pd.read_csv(base_path + 'ufc_event_data.csv')
    fight_data = pd.read_csv(base_path + 'ufc_fight_data.csv')
    fighter_data = pd.read_csv(base_path + 'ufc_fighter_data.csv')
    fight_stat_data = pd.read_csv(base_path + 'ufc_fight_stat_data.csv')
    return event_data, fight_data, fighter_data, fight_stat_data

def preprocess_event_data(event_data):
    event_data['event_date'] = pd.to_datetime(event_data['event_date'])
    return event_data

# Calculate fight frequency for each fighter
def calculate_fight_frequency(fighter_data):
    fight_frequency = fighter_data.groupby('fighter_id').size().reset_index(name='fight_count')
    return fight_frequency

# Analyze injury history
def analyze_injury_history(fight_stat_data):
    # Assuming injury history is represented by columns like knockdowns, reversals, etc.
    injury_history = fight_stat_data[['fighter_id', 'knockdowns', 'reversals']]
    # Aggregate injury metrics for each fighter
    injury_summary = injury_history.groupby('fighter_id').sum().reset_index()
    return injury_summary

# Calculate fighter longevity based on total number of fights
def calculate_fighter_longevity(fighter_data):
    print(fighter_data.columns)
    fighter_longevity = fighter_data.groupby('fighter_id').size().reset_index(name='fighter_longevity')
    return fighter_longevity

# Explore relationship between fight frequency, injury history, and fighter longevity
def analyze_longevity(fight_frequency, injury_summary, fighter_longevity, fighter_data):
    # Merge fight frequency, injury summary, and fighter longevity with fighter data
    fighter_data = pd.merge(fighter_data, fight_frequency, on='fighter_id', how='left')
    fighter_data = pd.merge(fighter_data, injury_summary, on='fighter_id', how='left')
    fighter_data = pd.merge(fighter_data, fighter_longevity, on='fighter_id', how='left')
    
    # Drop rows with missing values
    fighter_data.dropna(inplace=True)
    
    # Display descriptive statistics of relevant columns
    print("Descriptive Statistics:")
    print(fighter_data[['fight_count', 'knockdowns', 'fighter_longevity']].describe())
    
    # Visualize the relationship using scatter plots or other appropriate plots
    sns.pairplot(fighter_data[['fight_count', 'knockdowns', 'fighter_longevity']])
    plt.savefig('pairplot.png')
    plt.close()
    
    # Calculate correlation matrix
    correlation_matrix = fighter_data[['fight_count', 'knockdowns', 'fighter_longevity']].corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

def check_data_quality(fight_frequency, injury_summary, fighter_longevity, fighter_data):
    # Check for missing values
    print("Missing Values:")
    print(fighter_data.isnull().sum())
    
    # Check for constant columns
    print("\nConstant Columns:")
    constant_cols = fighter_data.columns[fighter_data.nunique() == 1]
    print(constant_cols)

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

    # Drop rows with missing values
    fighter_data.dropna(subset=['event_date_last', 'event_date_first', 'age_at_last_fight', 'career_length_years', 'age_at_debut'], inplace=True)

    return fighter_data

# Main function
def main():
    base_path = ',/data/' 
    event_data, fight_data, fighter_data, fight_stat_data = load_data(base_path)
    event_data = preprocess_event_data(event_data)
    fight_frequency = calculate_fight_frequency(fighter_data)
    injury_summary = analyze_injury_history(fight_stat_data)
    fighter_longevity = calculate_fighter_longevity(fighter_data)
    
    # Calculate fighter age and career length
    fighter_data = calculate_fighter_age_and_career_length(fight_data, event_data, fighter_data)
    
    check_data_quality(fight_frequency, injury_summary, fighter_longevity, fighter_data)
    analyze_longevity(fight_frequency, injury_summary, fighter_longevity, fighter_data)

if __name__ == "__main__":
    """
    From the descriptive statistics and correlation matrix, we can draw the following 
    observations regarding the potential effects of fight frequency and injury on 
    fighter longevity:

    Descriptive Statistics:
    The mean and standard deviation for all three variables (fight_count, knockdowns, f
    ighter_longevity) are relatively consistent, with fight_count having a standard 
    deviation of 0, indicating little variability in this variable.
    The minimum and maximum values suggest that the range of values for 
    knockdowns is broader than that for fight_count and fighter_longevity.
    Correlation Matrix:
    The correlation matrix shows the correlation coefficients between the variables. 
    However, since the fight_count variable is constant (as indicated by a standard 
    deviation of 0), the correlation between fight_count and other variables is 
    undefined (NaN).
    The correlation coefficient between knockdowns and fighter_longevity is also 
    undefined (NaN), likely due to the presence of NaN values or constant values.
    Overall, the analysis provides limited insights into the relationship between 
    fight frequency, injury (represented by knockdowns), and fighter longevity. 
    The undefined correlation coefficients suggest that further investigation or 
    refinement of the analysis may be necessary to understand these relationships 
    better. Additionally, the presence of missing values, especially in the knockdowns 
    variable, may also affect the reliability of the analysis.
    """
    main()
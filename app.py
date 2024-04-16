import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_data(base_path):
    event_data = pd.read_csv(base_path + 'ufc_event_data.csv')
    fight_data = pd.read_csv(base_path + 'ufc_fight_data.csv')
    fighter_data = pd.read_csv(base_path + 'ufc_fighter_data.csv')
    fight_stat_data = pd.read_csv(base_path + 'ufc_event_data.csv')
    return event_data, fight_data, fighter_data, fight_stat_data

def preprocess_event_data(event_data):
    """Ensure event_date is in datetime format."""
    event_data['event_date'] = pd.to_datetime(event_data['event_date'])
    return event_data

def calculate_fighter_age_and_career_length(fight_data, event_data, fighter_data):
    """Calculate age at last fight and career length by merging fight, event, and fighter data."""
    # Merge event data with fight data to get the event dates
    fight_data = pd.merge(fight_data, event_data[['event_id', 'event_date']], on='event_id', how='left')
    
    # Reshape fight data for analysis
    fights_f1 = fight_data[['f_1', 'event_date']].rename(columns={'f_1': 'fighter_id'})
    fights_f2 = fight_data[['f_2', 'event_date']].rename(columns={'f_2': 'fighter_id'})
    all_fights = pd.concat([fights_f1, fights_f2])
    
    # Find the first and last fight dates for each fighter
    last_fights = all_fights.groupby('fighter_id')['event_date'].max().reset_index().rename(columns={'event_date': 'event_date_last'})
    first_fights = all_fights.groupby('fighter_id')['event_date'].min().reset_index().rename(columns={'event_date': 'event_date_first'})
    
    # Merge first and last fight dates with fighter data
    fighter_data = pd.merge(fighter_data, last_fights, on='fighter_id', how='left')
    fighter_data = pd.merge(fighter_data, first_fights, on='fighter_id', how='left')
    
    # Calculate the age at last fight and career length in years
    fighter_data['fighter_dob'] = pd.to_datetime(fighter_data['fighter_dob'])
    fighter_data['age_at_last_fight'] = (fighter_data['event_date_last'].dt.year - fighter_data['fighter_dob'].dt.year)
    fighter_data['career_length_years'] = (fighter_data['event_date_last'] - fighter_data['event_date_first']).dt.days / 365.25
    
    return fighter_data


def plot_histogram(data, column, bins=20, kde=False, title="", xlabel="", ylabel=""):
    """Plot histogram for the specified column."""
    sns.histplot(data=data, x=column, bins=bins, kde=kde)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def main():
    base_path = './data/'
    event_data, fight_data, fighter_data, fight_stat_data = load_data(base_path)
    event_data = preprocess_event_data(event_data)
    fighter_data = calculate_fighter_age_and_career_length(fight_data, event_data, fighter_data)
    
  # Display data and plot histogram
    print("\nMissing or Incorrect Data in Final Dataset:")
    print(fighter_data[['fighter_id', 'event_date_last', 'age_at_last_fight', 'career_length_years']].isnull().sum())
    print("\nSample Data with Ages and Career Lengths at Last Fight:")
    print(fighter_data[['fighter_id', 'event_date_last', 'age_at_last_fight', 'career_length_years']].head())
    
    # Plotting the histogram of ages and career lengths at last fight
    plot_histogram(fighter_data, 'age_at_last_fight', bins=20, kde=True, title='Distribution of Fighter Ages at Last Fight', xlabel='Age at Last Fight', ylabel='Frequency')
    plot_histogram(fighter_data, 'career_length_years', bins=20, kde=True, title='Distribution of UFC Career Lengths', xlabel='Career Length (years)', ylabel='Frequency')

if __name__ == "__main__":
    main()

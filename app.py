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
    """Preprocess event data to ensure datetime format."""
    event_data['event_date'] = pd.to_datetime(event_data['event_date'])
    return event_data

def calculate_age_at_last_fight(fight_data, event_data, fighter_data):
    """Calculate the age of fighters at their last fight by merging fight, event, and fighter data."""
    # Merge event data with fight data
    fight_data = pd.merge(fight_data, event_data[['event_id', 'event_date']], on='event_id', how='left')

    # Reshape fight data
    fights_f1 = fight_data[['f_1', 'event_date']].rename(columns={'f_1': 'fighter_id'})
    fights_f2 = fight_data[['f_2', 'event_date']].rename(columns={'f_2': 'fighter_id'})
    all_fights = pd.concat([fights_f1, fights_f2])

    # Find the last fight date
    last_fights = all_fights.groupby('fighter_id')['event_date'].max().reset_index()

    # Merge last fight date with fighter data
    fighter_data = pd.merge(fighter_data, last_fights, on='fighter_id', how='left')

    # Calculate the age at the last fight
    fighter_data['fighter_dob'] = pd.to_datetime(fighter_data['fighter_dob'])
    fighter_data['age_at_last_fight'] = (fighter_data['event_date'].dt.year - fighter_data['fighter_dob'].dt.year)
    
    return fighter_data

def plot_histogram(data, column, bins=20, kde=False, title="", xlabel="", ylabel=""):
    sns.histplot(data=data, x=column, bins=bins, kde=kde)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def main():
    base_path = './data/'
    event_data, fight_data, fighter_data, fight_stat_data = load_data(base_path)
    event_data = preprocess_event_data(event_data)
    fighter_data = calculate_age_at_last_fight(fight_data, event_data, fighter_data)
    
    # Display data and plot histogram
    print("\nMissing or Incorrect Data in Final Dataset:")
    print(fighter_data[['fighter_id', 'event_date', 'age_at_last_fight']].isnull().sum())
    print("\nSample Data with Ages at Last Fight:")
    print(fighter_data[['fighter_id', 'event_date', 'age_at_last_fight']].head())
    
    plot_histogram(fighter_data, 'age_at_last_fight', bins=20, kde=True, title='Distribution of Fighter Ages at Last Fight',
                   xlabel='Age at Last Fight', ylabel='Frequency')

if __name__ == "__main__":
    main()

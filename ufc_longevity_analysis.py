import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

def load_data(base_path):
    event_data = pd.read_csv(base_path + 'ufc_event_data.csv')
    fight_data = pd.read_csv(base_path + 'ufc_fight_data.csv')
    fighter_data = pd.read_csv(base_path + 'ufc_fighter_data.csv')
    return event_data, fight_data, fighter_data

def preprocess_event_data(event_data):
    """Ensure event_date is in datetime format."""
    event_data['event_date'] = pd.to_datetime(event_data['event_date'], errors='coerce')
    return event_data

def calculate_fighter_age_and_career_length(fight_data, event_data, fighter_data):
    """Calculate age at last fight and career length by merging fight, event, and fighter data."""
    fight_data = pd.merge(fight_data, event_data[['event_id', 'event_date']], on='event_id', how='left')
    fights_f1 = fight_data[['f_1', 'event_date']].rename(columns={'f_1': 'fighter_id'})
    fights_f2 = fight_data[['f_2', 'event_date']].rename(columns={'f_2': 'fighter_id'})
    all_fights = pd.concat([fights_f1, fights_f2])
    
    last_fights = all_fights.groupby('fighter_id')['event_date'].max().reset_index().rename(columns={'event_date': 'event_date_last'})
    first_fights = all_fights.groupby('fighter_id')['event_date'].min().reset_index().rename(columns={'event_date': 'event_date_first'})
    
    fighter_data = pd.merge(fighter_data, last_fights, on='fighter_id', how='left')
    fighter_data = pd.merge(fighter_data, first_fights, on='fighter_id', how='left')
    fighter_data['event_date_last'] = pd.to_datetime(fighter_data['event_date_last'], errors='coerce')
    fighter_data['event_date_first'] = pd.to_datetime(fighter_data['event_date_first'], errors='coerce')

    # Calculate the age at last fight and career length in years for valid entries
    fighter_data = fighter_data.dropna(subset=['event_date_last', 'event_date_first'])
    fighter_data['age_at_last_fight'] = fighter_data['event_date_last'].dt.year - pd.to_datetime(fighter_data['fighter_dob']).dt.year
    fighter_data['career_length_years'] = (fighter_data['event_date_last'] - fighter_data['event_date_first']).dt.days / 365.25

    return fighter_data

def perform_correlation_analysis(fighter_data):
    plt.figure(figsize=(12, 10))  # Adjust the size as needed
    correlation_matrix = fighter_data[['fighter_height_cm', 'fighter_weight_lbs', 'fighter_reach_cm', 'age_at_last_fight', 'career_length_years']].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Correlation Matrix of Fighter Attributes')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
    plt.yticks(rotation=0)  # Ensure y-axis labels are horizontal for clarity
    plt.tight_layout()  # Adjust layout to make room for label
    plt.show()

def plot_relationship(data, x, y):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x=x, y=y)
    plt.title(f'Relationship between {x} and {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

def main():
    base_path = './data/'
    event_data, fight_data, fighter_data = load_data(base_path)
    event_data = preprocess_event_data(event_data)
    fighter_data = calculate_fighter_age_and_career_length(fight_data, event_data, fighter_data)
    perform_correlation_analysis(fighter_data)
     # Plotting relationships with career length
    plot_relationship(fighter_data, 'fighter_height_cm', 'career_length_years')
    plot_relationship(fighter_data, 'fighter_weight_lbs', 'career_length_years')
    plot_relationship(fighter_data, 'fighter_reach_cm', 'career_length_years')
    plot_relationship(fighter_data, 'age_at_debut', 'career_length_years')

if __name__ == "__main__":
    main()
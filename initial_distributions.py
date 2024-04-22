import pandas as pd
import matplotlib.pyplot as plt

def load_data(base_path):
    # Load event, fight, fight stats, and fighter data from CSV files
    event_data = pd.read_csv(base_path + 'ufc_event_data.csv')
    fight_data = pd.read_csv(base_path + 'ufc_fight_data.csv')
    fight_stat_data = pd.read_csv(base_path + 'ufc_fight_stat_data.csv')
    fighter_data = pd.read_csv(base_path + 'ufc_fighter_data.csv')
    return event_data, fight_data, fight_stat_data, fighter_data

def main():
    # Base path to the directory where your data files are stored
    base_path = './data/'  # Update this path as needed for your environment
    
    # Load the data
    event_data, fight_data, fight_stat_data, fighter_data = load_data(base_path)

    # Histogram for Fighter Weight
    plt.figure(figsize=(12, 6))
    plt.hist(fighter_data['fighter_weight_lbs'].dropna(), bins=30, color='blue', alpha=0.7)
    plt.title('Distribution of Fighter Weights')
    plt.xlabel('Weight in lbs')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Histogram for Fighter Height
    plt.figure(figsize=(12, 6))
    plt.hist(fighter_data['fighter_height_cm'].dropna(), bins=30, color='green', alpha=0.7)
    plt.title('Distribution of Fighter Heights')
    plt.xlabel('Height in cm')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Histogram for Knockdowns
    plt.figure(figsize=(12, 6))
    plt.hist(fight_stat_data['knockdowns'].dropna(), bins=30, color='red', alpha=0.7)
    plt.title('Distribution of Knockdowns per Fight')
    plt.xlabel('Knockdowns')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

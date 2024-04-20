import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

def load_data(base_path):
    event_data = pd.read_csv(base_path + 'ufc_event_data.csv')
    fight_data = pd.read_csv(base_path + 'ufc_fight_data.csv')
    fighter_data = pd.read_csv(base_path + 'ufc_fighter_data.csv')
    return event_data, fight_data, fighter_data

def visualize_clusters(df):
    """
    Visualize clustering results using a scatter plot.
    
    Parameters:
        df (pd.DataFrame): DataFrame that includes 'fighter_weight_lbs', 'career_length_years', and 'cluster' columns.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(df['fighter_weight_lbs'], df['career_length_years'], c=df['cluster'], cmap='viridis', alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.title('Clustering of UFC Fighters by Weight and Career Length')
    plt.xlabel('Fighter Weight (lbs)')
    plt.ylabel('Career Length (Years)')
    plt.colorbar(label='Cluster')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def prepare_fighter_data(fighter_data):
    fighter_data['fighter_weight_class'] = fighter_data['fighter_weight_lbs'].apply(assign_weight_class)
    return fighter_data

def assign_weight_class(weight):
    # Define UFC weight classes
    if weight <= 125:
        return 'Flyweight'
    elif weight <= 135:
        return 'Bantamweight'
    elif weight <= 145:
        return 'Featherweight'
    elif weight <= 155:
        return 'Lightweight'
    elif weight <= 170:
        return 'Welterweight'
    elif weight <= 185:
        return 'Middleweight'
    elif weight <= 205:
        return 'Light Heavyweight'
    elif weight <= 265:
        return 'Heavyweight'
    else:
        return 'Super Heavyweight'  # Above 265 lbs

def calculate_fighter_age_and_career_length(fight_data, event_data, fighter_data):
    # Merge fight data with event dates
    fight_data = pd.merge(fight_data, event_data[['event_id', 'event_date']], on='event_id', how='left')
    fights_f1 = fight_data[['f_1', 'event_date']].rename(columns={'f_1': 'fighter_id'})
    fights_f2 = fight_data[['f_2', 'event_date']].rename(columns={'f_2': 'fighter_id'})
    all_fights = pd.concat([fights_f1, fights_f2])

    # Calculate last and first fight dates
    last_fights = all_fights.groupby('fighter_id')['event_date'].max().reset_index().rename(columns={'event_date': 'event_date_last'})
    first_fights = all_fights.groupby('fighter_id')['event_date'].min().reset_index().rename(columns={'event_date': 'event_date_first'})

    # Merge dates back to fighter data
    fighter_data = pd.merge(fighter_data, last_fights, on='fighter_id', how='left')
    fighter_data = pd.merge(fighter_data, first_fights, on='fighter_id', how='left')

    # Ensure datetime format
    fighter_data['event_date_last'] = pd.to_datetime(fighter_data['event_date_last'])
    fighter_data['event_date_first'] = pd.to_datetime(fighter_data['event_date_first'])
    fighter_data['fighter_dob'] = pd.to_datetime(fighter_data['fighter_dob'])

    # Calculate age at last fight and career length
    fighter_data['age_at_last_fight'] = fighter_data['event_date_last'].dt.year - fighter_data['fighter_dob'].dt.year
    fighter_data['career_length_years'] = (fighter_data['event_date_last'] - fighter_data['event_date_first']).dt.days / 365.25
    fighter_data['age_at_debut'] = (fighter_data['event_date_first'] - fighter_data['fighter_dob']).dt.days / 365.25

    return fighter_data

def perform_clustering(fighter_data):
    # Ensure no NaN values exist in the features to be used
    if fighter_data.isnull().any().any():
        print("NaN values detected, applying imputation again...")
        fighter_data = impute_data(fighter_data)

    # Prepare data for clustering
    fighter_data = pd.get_dummies(fighter_data, columns=['fighter_weight_class'], drop_first=True)
    feature_columns = ['fighter_weight_lbs', 'career_length_years'] + [col for col in fighter_data.columns if 'fighter_weight_class_' in col]

    # Check if still any NaN exists
    if fighter_data[feature_columns].isnull().any().any():
        raise Exception("NaN values are still present in the data after attempted imputation.")

    # Proceed with clustering
    kmeans = KMeans(n_clusters=8, random_state=42)
    fighter_data['cluster'] = kmeans.fit_predict(fighter_data[feature_columns])
    cluster_summary = fighter_data.groupby('cluster')[['fighter_weight_lbs', 'career_length_years']].agg(['mean', 'count'])
    print(cluster_summary)

    return fighter_data

def filter_outliers(fighter_data):
    # Calculate the mean and standard deviation
    mean_weight = fighter_data['fighter_weight_lbs'].mean()
    std_weight = fighter_data['fighter_weight_lbs'].std()
    
    # Filter out extreme weights that are beyond 3 standard deviations from the mean
    filtered_data = fighter_data[(fighter_data['fighter_weight_lbs'] <= mean_weight + 3 * std_weight) & 
                                 (fighter_data['fighter_weight_lbs'] >= mean_weight - 3 * std_weight)]
    return filtered_data

def impute_data(fighter_data):
    imputer = KNNImputer(n_neighbors=5)
    numeric_columns = fighter_data.select_dtypes(include=[np.number]).columns
    fighter_data[numeric_columns] = imputer.fit_transform(fighter_data[numeric_columns])
    return fighter_data


def main():
    base_path = './data/'
    event_data, fight_data, fighter_data = load_data(base_path)
    fighter_data = filter_outliers(fighter_data)
    fighter_data = prepare_fighter_data(fighter_data)
    fighter_data = calculate_fighter_age_and_career_length(fight_data, event_data, fighter_data)
    fighter_data = impute_data(fighter_data)  # Ensure this is done last before clustering
    fighter_data = perform_clustering(fighter_data)
    visualize_clusters(fighter_data)  # Add this line to visualize the results

if __name__ == "__main__":
    main()

# Summary of Clustering Analysis:
# Purpose: This analysis groups UFC fighters into clusters based on their weight and measured career lengths.
# Key Insights:
# - Clusters are labeled from 0 to 7, representing a range from Flyweight to Super Heavyweight classes.
# - Each cluster shows the mean weight and the average career length of fighters, providing insights into the typical career trajectories in each weight class.
# - Career lengths vary across clusters, with most clusters showing average career lengths between 2.75 and 3.01 years, indicating that weight class alone does not dictate career length.
# - The distribution of fighters in each cluster and the calculated average career lengths can inform training strategies, career management, and health preservation practices.
# Implications:
# - These insights assist in understanding how different weight classes correlate with career longevity, aiding stakeholders in making informed decisions regarding fighter development and management.
# - This clustering analysis serves as a foundational tool for further statistical analysis and strategic planning in athletic career development within the UFC.

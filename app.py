import pandas as pd

# loading in data
base_path = './data/'
ufc_event_data = pd.read_csv(base_path + 'ufc_event_data.csv')
ufc_fight_data = pd.read_csv(base_path + 'ufc_fight_data.csv')
ufc_fight_stat_data = pd.read_csv(base_path + 'ufc_fight_stat_data.csv')
ufc_fighter_data = pd.read_csv(base_path + 'ufc_fighter_data.csv')

print("UFC Event Data:")
print(ufc_event_data.head())

print("\nUFC Fight Data:")
print(ufc_fight_data.head())

print("\nUFC Fight Stat Data:")
print(ufc_fight_stat_data.head())

print("\nUFC Fighter Data:")
print(ufc_fighter_data.head())

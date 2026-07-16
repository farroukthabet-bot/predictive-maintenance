import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_data, load_rul

def add_rul_column(df):
    """Calculates Remaining Useful Life for each row in the TRAINING data."""
    # Find the max cycle (i.e. the failure point) for each engine
    max_cycles = df.groupby('unit_number')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycle']
    
    # Merge that back into the main dataframe
    df = df.merge(max_cycles, on='unit_number', how='left')
    
    # RUL = max_cycle (failure point) minus current cycle
    df['RUL'] = df['max_cycle'] - df['time_cycles']
    df = df.drop('max_cycle', axis=1)
    return df

def cap_rul(df, cap=125):
    """Caps RUL at a max value - early-life cycles all get treated as 'plenty of life left'."""
    df['RUL'] = df['RUL'].clip(upper=cap)
    return df

if __name__ == "__main__":
    train = load_data('../archive/CMaps/train_FD001.txt')
    train = add_rul_column(train)
    train = cap_rul(train)

    print("RUL STATS AFTER CAPPING:")
    print(train['RUL'].describe())

    print("TRAIN WITH RUL:")
    print(train[['unit_number', 'time_cycles', 'RUL']].head(10))

    print("\nRUL STATS:")
    print(train['RUL'].describe())

    # Plot how a couple of sensors degrade over an engine's life - engine 1 as example
    engine_1 = train[train['unit_number'] == 1]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(engine_1['time_cycles'], engine_1['sensor_2'])
    axes[0].set_title('Sensor 2 over Engine 1 lifetime')
    axes[0].set_xlabel('Cycle')

    axes[1].plot(engine_1['time_cycles'], engine_1['sensor_11'])
    axes[1].set_title('Sensor 11 over Engine 1 lifetime')
    axes[1].set_xlabel('Cycle')

    plt.tight_layout()
    plt.savefig('sensor_degradation.png')
    print("\nPlot saved as sensor_degradation.png")

    
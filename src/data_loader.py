import pandas as pd

# Column names based on the readme: unit number, time cycles, 3 op settings, 21 sensors
COLUMN_NAMES = ['unit_number', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
               [f'sensor_{i}' for i in range(1, 22)]

def load_data(file_path):
    """Loads a CMAPSS train/test file (space-separated, no header)."""
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=COLUMN_NAMES)
    return df

def load_rul(file_path):
    """Loads the RUL answer-key file for the test set."""
    rul = pd.read_csv(file_path, sep=r'\s+', header=None, names=['RUL'])
    return rul

if __name__ == "__main__":
    train = load_data('archive/CMaps/train_FD001.txt')
    test = load_data('archive/CMaps/test_FD001.txt')
    rul = load_rul('archive/CMaps/RUL_FD001.txt')

    print("TRAIN SHAPE:", train.shape)
    print(train.head())

    print("\nTEST SHAPE:", test.shape)
    print(test.head())

    print("\nRUL SHAPE:", rul.shape)
    print(rul.head())

    print("\nNumber of unique engines in train:", train['unit_number'].nunique())
    print("Number of unique engines in test:", test['unit_number'].nunique())
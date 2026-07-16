import numpy as np
from data_loader import load_data
from feature_engineering import add_rul_column, cap_rul

SEQUENCE_LENGTH = 30  # how many cycles of history the LSTM sees at once

def create_sequences(df, feature_cols, sequence_length=SEQUENCE_LENGTH):
    """
    For each engine, slides a window across its life and creates
    (sequence_of_readings, RUL_at_end_of_sequence) pairs.
    """
    sequences = []
    labels = []

    for unit_id in df['unit_number'].unique():
        engine_data = df[df['unit_number'] == unit_id].sort_values('time_cycles')
        engine_features = engine_data[feature_cols].values
        engine_rul = engine_data['RUL'].values

        # Only engines with enough history to fill at least one window
        if len(engine_data) < sequence_length:
            continue

        for i in range(len(engine_data) - sequence_length + 1):
            seq = engine_features[i:i + sequence_length]
            label = engine_rul[i + sequence_length - 1]  # RUL at the LAST cycle in the window
            sequences.append(seq)
            labels.append(label)

    return np.array(sequences), np.array(labels)

if __name__ == "__main__":
    train = load_data('../archive/CMaps/train_FD001.txt')
    train = add_rul_column(train)
    train = cap_rul(train)

    feature_cols = [col for col in train.columns if col not in ['unit_number', 'time_cycles', 'RUL']]

    X, y = create_sequences(train, feature_cols)

    print("Sequence data shape:", X.shape)  # (num_sequences, sequence_length, num_features)
    print("Labels shape:", y.shape)
    print("\nExample: first sequence's shape:", X[0].shape)
    print("Example: first sequence's label (RUL):", y[0])
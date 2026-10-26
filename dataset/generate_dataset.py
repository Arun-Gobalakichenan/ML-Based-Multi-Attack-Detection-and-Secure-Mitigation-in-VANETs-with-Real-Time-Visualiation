import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42

FEATURES = [
    "packet_rate",
    "avg_speed",
    "speed_std",
    "neighbor_count",
    "packet_drop_rate",
    "position_deviation",
    "message_consistency",
    "signal_strength",
]

CLASSES = [
    "Normal",
    "DoS",
    "Sybil",
    "Blackhole",
    "False_Information",
]


def generate_dataset(samples_per_class=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []

    # These are illustrative synthetic distributions, not measured
    # observations from an OMNeT++/Veins/SUMO simulation.
    profiles = {
        "Normal": {
            "packet_rate": (20, 5),
            "avg_speed": (45, 12),
            "speed_std": (5, 2),
            "neighbor_count": (12, 4),
            "packet_drop_rate": (0.03, 0.02),
            "position_deviation": (1.0, 0.5),
            "message_consistency": (0.95, 0.04),
            "signal_strength": (-65, 8),
        },
        "DoS": {
            "packet_rate": (150, 25),
            "avg_speed": (40, 15),
            "speed_std": (8, 4),
            "neighbor_count": (15, 5),
            "packet_drop_rate": (0.35, 0.15),
            "position_deviation": (2.0, 1.0),
            "message_consistency": (0.65, 0.18),
            "signal_strength": (-55, 12),
        },
        "Sybil": {
            "packet_rate": (55, 15),
            "avg_speed": (45, 12),
            "speed_std": (6, 3),
            "neighbor_count": (35, 8),
            "packet_drop_rate": (0.08, 0.05),
            "position_deviation": (3.5, 1.5),
            "message_consistency": (0.55, 0.18),
            "signal_strength": (-60, 10),
        },
        "Blackhole": {
            "packet_rate": (25, 8),
            "avg_speed": (42, 14),
            "speed_std": (7, 3),
            "neighbor_count": (12, 4),
            "packet_drop_rate": (0.75, 0.12),
            "position_deviation": (1.5, 0.8),
            "message_consistency": (0.60, 0.18),
            "signal_strength": (-68, 9),
        },
        "False_Information": {
            "packet_rate": (22, 7),
            "avg_speed": (46, 12),
            "speed_std": (6, 3),
            "neighbor_count": (13, 4),
            "packet_drop_rate": (0.06, 0.04),
            "position_deviation": (12, 4),
            "message_consistency": (0.25, 0.15),
            "signal_strength": (-66, 9),
        },
    }

    for label, profile in profiles.items():
        data = {}

        for feature in FEATURES:
            mean, std = profile[feature]
            values = rng.normal(mean, std, samples_per_class)

            if feature in ("packet_rate", "neighbor_count"):
                values = np.maximum(0, np.rint(values)).astype(int)
            elif feature in ("packet_drop_rate", "message_consistency"):
                values = np.clip(values, 0, 1)
            elif feature in ("speed_std", "position_deviation"):
                values = np.maximum(0, values)

            data[feature] = values

        frame = pd.DataFrame(data)
        frame["label"] = label
        rows.append(frame)

    dataset = pd.concat(rows, ignore_index=True)
    return dataset.sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a reproducible synthetic VANET security dataset."
    )
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.samples_per_class < 10:
        parser.error("--samples-per-class must be at least 10")

    root = Path(__file__).resolve().parent
    raw_dir = root / "raw"
    processed_dir = root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_dataset(args.samples_per_class, args.seed)

    raw_path = raw_dir / "vanet_synthetic_dataset.csv"
    dataset.to_csv(raw_path, index=False)

    train, test = train_test_split(
        dataset,
        test_size=0.20,
        random_state=args.seed,
        stratify=dataset["label"],
    )

    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    # Fit scaling parameters on training data only to prevent data leakage.
    scaler = StandardScaler()
    scaler.fit(train[FEATURES])

    scaled_train = train.copy()
    scaled_test = test.copy()
    scaled_train[FEATURES] = scaler.transform(train[FEATURES])
    scaled_test[FEATURES] = scaler.transform(test[FEATURES])

    scaled_train.to_csv(processed_dir / "train_scaled.csv", index=False)
    scaled_test.to_csv(processed_dir / "test_scaled.csv", index=False)

    print("\nDataset generation successful")
    print(f"Total samples: {len(dataset)}")
    print(f"Features: {len(FEATURES)}")
    print(f"Classes: {len(CLASSES)}")
    print(f"Training samples: {len(train)}")
    print(f"Testing samples: {len(test)}")
    print(f"\nClass distribution:\n{dataset['label'].value_counts().sort_index()}")
    print(f"\nRaw dataset: {raw_path}")
    print(f"Training data: {train_path}")
    print(f"Testing data: {test_path}")
    print("\nNote: These are synthetic samples, not real simulation measurements.")


if __name__ == "__main__":
    main()

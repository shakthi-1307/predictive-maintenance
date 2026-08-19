import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class RULSequenceDataset(Dataset):

    def __init__(
        self,
        df,
        feature_columns,
        sequence_length=30,
    ):

        self.sequence_length = (
            sequence_length
        )

        self.samples = []

        # Process each engine independently.
        for engine_id, engine_df in (
            df.groupby("unit")
        ):

            engine_df = (
                engine_df
                .sort_values("cycle")
                .reset_index(drop=True)
            )

            features = (
                engine_df[
                    feature_columns
                ]
                .values
                .astype(np.float32)
            )

            targets = (
                engine_df["RUL"]
                .values
                .astype(np.float32)
            )

            # Generate sliding windows.
            for end_idx in range(
                sequence_length - 1,
                len(engine_df),
            ):

                start_idx = (
                    end_idx
                    - sequence_length
                    + 1
                )

                sequence = features[
                    start_idx:
                    end_idx + 1
                ]

                target = targets[
                    end_idx
                ]

                self.samples.append(
                    (
                        sequence,
                        target,
                    )
                )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(self, index):

        sequence, target = (
            self.samples[index]
        )

        return (
            torch.tensor(
                sequence,
                dtype=torch.float32,
            ),

            torch.tensor(
                target,
                dtype=torch.float32,
            ),
        )
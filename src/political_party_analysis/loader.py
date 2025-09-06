from pathlib import Path
from typing import List
from urllib.request import urlretrieve

import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """Class to load the political parties dataset"""

    data_url: str = "https://www.chesdata.eu/s/CHES2019V3.dta"

    def __init__(self):
        self.party_data = self._download_data()
        # Exclude these columns that are missing for non-Turkish parties
        self.non_features = ["eu_econ_require", "eu_political_require", "eu_googov_require"]
        self.index = ["party_id", "party", "country"]

    def _download_data(self) -> pd.DataFrame:
        data_path, _ = urlretrieve(
            self.data_url,
            Path(__file__).parents[2].joinpath(*["data", "CHES2019V3.dta"]),
        )
        return pd.read_stata(data_path)

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to remove duplicates in a dataframe"""
        ##### YOUR CODE GOES HERE #####
        return df.drop_duplicates(subset=self.index)

    def remove_nonfeature_cols(self, df: pd.DataFrame, non_features: List[str], index: List[str]) -> pd.DataFrame:
        """Write a function to remove certain features cols and set certain cols as indices
        in a dataframe"""
        ##### YOUR CODE GOES HERE #####
        df = df.drop(columns=non_features)
        df = df.set_index(index)
        return df

    def handle_NaN_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to handle NaN values in a dataframe"""
        ##### YOUR CODE GOES HERE #####
        # Fill NaNs with mean of the column
        df = df.fillna(df.mean())
        # If any remaining columns are all NaNs, drop them
        df = df.dropna(axis=1, how="all")
        return df

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to normalise values in a dataframe. Use StandardScaler."""
        ##### YOUR CODE GOES HERE #####
        # If there is no existing scaler, create one and fit it to the data
        if not hasattr(self, "scaler"):
            self.scaler = StandardScaler()
            df = pd.DataFrame(self.scaler.fit_transform(df), columns=df.columns, index=df.index)
        # Else transform the data using parameters from the existing scaler
        else:
            df = pd.DataFrame(self.scaler.transform(df), columns=df.columns, index=df.index)
        return df

    def preprocess_data(self):
        """Write a function to combine all pre-processing steps for the dataset"""
        ##### YOUR CODE GOES HERE #####
        self.party_data = self.remove_duplicates(self.party_data)
        self.party_data = self.remove_nonfeature_cols(self.party_data, self.non_features, self.index)
        self.party_data = self.handle_NaN_values(self.party_data)
        self.party_data = self.scale_features(self.party_data)

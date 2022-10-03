"""Vecto Testing Utility Functions

This script contains the utility functions needed to run Vecto API testing.
Utility functions are categorized into 3 classes (aka groups):
1. TestDataset class: A class with static methods for getting data to be ingested into Vecto
2. VectoAPI class: A class for users to instantiate a VectoAPI object, e.g. public_vecto and private_vecto object
3. DatabaseTwin class: A class for users to instantiate a DatabaseTwin object
"""


import pathlib
import pandas as pd
import configparser
from requests_toolbelt import MultipartEncoder
import random
import json

random.seed(1234)

# Fetch Vecto config from environment
import os
token = os.environ['user_token']
vector_space_id = os.environ['vector_space_id']

# Set paths
base_dir = pathlib.Path().absolute()
path_to_dataset = 'vecto/api-tests/demo_dataset'
dataset_path = base_dir.joinpath(path_to_dataset)

class TestDataset:
    
    
    # Get dataset

    @classmethod
    def get_image_dataset(cls) -> list[str]:
        """Gets and returns the list of image paths to be ingested into Vecto.

        Args: None

        Returns: 
            list: a list of image paths
        """
        dataset_images = list(dataset_path.glob('**/*.png'))

        return dataset_images

    @classmethod
    def get_random_image(cls) -> list[str]:
        """Gets and returns randomly one image path to be ingested into Vecto.

        Args: None

        Returns: 
            list: a random image path
        """
        dataset_images = cls.get_image_dataset()
        random_image = dataset_images[random.randrange(len(dataset_images))]
        return [random_image]
    
    @classmethod
    def get_text_dataset(cls) -> list[str]:
        """Gets and returns the list of input text to be ingested into Vecto.

        Args: None

        Returns: 
            list: a list of input text
        """
        df = pd.read_csv(dataset_path.joinpath('colors.csv'), 
                names=['color', 'name', 'hex', 'R', 'G', 'B'])
        df = df[:100]

        return df['name']
    
    @classmethod
    def get_random_text(cls) -> list[str]:
        """Gets and returns the list of image paths to be ingested into Vecto.

        Args: None

        Returns: 
            list: a random text
        """
        dataset_text = cls.get_text_dataset()
        random_text = dataset_text.iloc[random.randrange(len(dataset_text))]
        return [random_text]


class DatabaseTwin:
    """A class to represent a twin of the Vecto database, 
    to be used to check against the entries in Vecto.

    Args: None
    """

    def __init__(self) -> None:
        self.ref_db = []
        self.deleted_ids = []

    def update_database(self, results, metadata) -> None:
        """A function to update the database twin with new entries, 
        which will be used to check if Vecto ingested the entries correctly.

        Args:
            results (list): A list of vector ids ingested into Vecto
            metadata (list): A list of vector metadata

        Returns: None
        """
        for id, path in zip(results, metadata['data']):
            self.ref_db.append([id, json.loads(path)])

    def get_database(self) -> pd.DataFrame:
        """A function to get the latest database twin, 
        which will be used to check if Vecto ingested the entries correctly.

        Args: None

        Returns:
            DataFrame: A Pandas dataframe
        """
        ref_df = pd.DataFrame(self.ref_db, columns=['id', 'metadata'])
        
        return ref_df

    def update_deleted_ids(self, vector_ids) -> None:
        """A function to update the database twin with deleted vector ids, 
        which will be used to check if Vecto deleted the entries correctly.

        Args:
            results (list): A list of vector ids deleted from Vecto

        Returns: None
        """
        for vector_id in vector_ids:
            self.deleted_ids.append(vector_id)

    def get_deleted_ids(self) -> list[int]:
        """A function to get the latest deleted vector ids, 
        which will be used to check if Vecto deleted the entries correctly.

        Args: None

        Returns:
            list: A list of deleted vector ids
        """
        return self.deleted_ids
        
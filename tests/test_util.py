# Copyright 2022 Xpress AI

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vecto Testing Utility Functions

This script contains the utility functions needed to run Vecto API testing.
Utility functions are categorized into 3 classes (aka groups):
1. TestDataset class: A class with static methods for getting data to be ingested into Vecto
2. Vecto class: A class for users to instantiate a Vecto object, e.g. public_vecto and private_vecto object
3. DatabaseTwin class: A class for users to instantiate a DatabaseTwin object
"""


import pathlib
import pandas as pd
import random
import json
from typing import List

random.seed(1234)

# Fetch Vecto config from environment
import os
token = os.environ['user_token']
vector_space_id = int(os.environ['vector_space_id'])

# Set paths
base_dir = pathlib.Path().absolute()
path_to_dataset = 'tests/demo_dataset'
dataset_path = base_dir.joinpath(path_to_dataset)

class TestDataset:
    
    
    # Get dataset

    @classmethod
    def get_image_dataset(cls) -> List[str]:
        """Gets and returns the list of image paths to be ingested into Vecto.

        Args: None

        Returns: 
            list: a list of image paths
        """
        dataset_images = list(dataset_path.glob('**/*.png'))

        return dataset_images

    @classmethod
    def get_random_image(cls) -> List[str]:
        """Gets and returns randomly one image path to be ingested into Vecto.

        Args: None

        Returns: 
            list: a random image path
        """
        dataset_images = cls.get_image_dataset()
        random_image = dataset_images[random.randrange(len(dataset_images))]
        return [random_image]
    
    @classmethod
    def get_color_dataset(cls) -> List[str]:
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
    def get_profession_dataset(cls) -> List[str]:
        """Gets and returns the list of input text to be ingested into Vecto.

        Args: None

        Returns: 
            list: a list of input text
        """

        file = "tests/demo_dataset/profession.txt"
        with open(file) as f:
            professions = [profession.rstrip() for profession in f]        
            
        return professions

    @classmethod
    def get_random_text(cls, text_dataset) -> List[str]:
        """Gets and returns the list of image paths to be ingested into Vecto.

        Args: None

        Returns: 
            list: a random text
        """
        dataset_text = text_dataset()
        random_text = dataset_text.iloc[random.randrange(len(dataset_text))]
        return [random_text]

    @classmethod
    def get_image_attribute(cls, batch_path_list) -> dict:
        """Computes the attribute that is done in ingest_image.

        Args: None

        Returns: 
            dict: the attribute
        """
        data = {'vector_space_id': vector_space_id, 'data': [], 'modality': 'IMAGE'}
        files = []
        for path in batch_path_list:
            relative = "%s/%s" % (path.parent.name, path.name)
            # relative = ""
            data['data'].append(json.dumps(relative))
            files.append(open(path, 'rb'))
        
        for f in files:
            f.close()

        return data

    @classmethod
    def get_text_attribute(cls, batch_text_list:list, batch_index_list:list) -> dict:
        """Computes the attribute that is done in ingest_text.

        Args: None

        Returns: 
            dict: the attribute
        """
        data = []

        for index, text in zip(batch_index_list, batch_text_list):
            data.append('text_{}'.format(index) + '_{}'.format(text))

        return data

class DatabaseTwin:
    """A class to represent a twin of the Vecto database, 
    to be used to check against the entries in Vecto.

    Args: None
    """

    def __init__(self) -> None:
        self.ref_db = []
        self.deleted_ids = []

    def update_database(self, results, attribute) -> None:
        """A function to update the database twin with new entries, 
        which will be used to check if Vecto ingested the entries correctly.

        Args:
            results (list): A list of vector ids ingested into Vecto
            attribute (list): A list of vector attribute

        Returns: None
        """

        for id, path in zip(results, attribute):
            self.ref_db.append([id, path])

    def get_database(self) -> pd.DataFrame:
        """A function to get the latest database twin, 
        which will be used to check if Vecto ingested the entries correctly.

        Args: None

        Returns:
            DataFrame: A Pandas dataframe
        """
        ref_df = pd.DataFrame(self.ref_db, columns=['id', 'attribute'])
        
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

    def get_deleted_ids(self) -> List[int]:
        """A function to get the latest deleted vector ids, 
        which will be used to check if Vecto deleted the entries correctly.

        Args: None

        Returns:
            list: A list of deleted vector ids
        """
        return self.deleted_ids
        
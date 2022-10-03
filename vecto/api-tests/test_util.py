"""Vecto Testing Utility Functions

This script contains the utility functions needed to run Vecto API testing.
Utility functions are categorized into 3 classes (aka groups):
1. TestDataset class: A class with static methods for getting data to be ingested into Vecto
2. VectoAPI class: A class for users to instantiate a VectoAPI object, e.g. public_vecto and private_vecto object
3. DatabaseTwin class: A class for users to instantiate a DatabaseTwin object
"""

from typing import List, Tuple
import requests
import pathlib
import io
import pandas as pd
import configparser
from requests_toolbelt import MultipartEncoder
import pdb # to be used in pdb.set_trace()
import random
import json

random.seed(1234)

# Parse Vecto config
vecto_config = configparser.ConfigParser()
vecto_config.read('vecto_config.env')
vecto_base_url = vecto_config['vecto']['vecto_base_url']
assert len(vecto_base_url) != 0

# Set paths
base_dir = pathlib.Path().absolute()
path_to_dataset = 'demo_dataset'
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


class VectoAPI():

    def __init__(self, token, vector_space_id, client=requests) -> None:
        self.token = token
        self.vector_space_id = vector_space_id
        self.client = client


    # Ingest
    def ingest(self, data, files, **kwargs):
        """A function to ingest a batch of data into Vecto.
        Also works with single entry aka batch of 1.

        Args:
            data (dict): Dictionary containing regular fields
            files (list): List of file-like objects to be ingested
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            client response
        """
        results = self.client.post("%s/api/v0/index" % vecto_base_url,
                                   data=data,
                                   files=[('input', ('_', f, '_')) for f in files],
                                   headers={"Authorization": "Bearer %s" % self.token},
                                   **kwargs)

        return results

    def ingest_image_batch(self, batch_path_list, **kwargs) -> tuple[dict, dict]:
        """A function to ingest a batch of images into Vecto.
        Also works with single image aka batch of 1.

        Args:
            batch_path_list (list): List of image paths (or list of one image path if batch of 1)
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            tuple: A tuple of two dictionaries (client response body, client request body)
        """
        data = {'vector_space_id': self.vector_space_id, 'data': [], 'modality': 'IMAGE'}
        files = []
        for path in batch_path_list:
            relative = "%s/%s" % (path.parent.name, path.name)
            data['data'].append(json.dumps(relative))
            files.append(open(path, 'rb'))

        results = self.ingest(data, files)
        for f in files:
            f.close()
        
        return results, data

    def ingest_text_batch(self, batch_index_list, batch_text_list, **kwargs) -> tuple[dict, dict]:
        """A function to ingest a batch of text into Vecto. 
        Also works with single text aka batch of 1.

        Args:
            batch_text_list (list): List of texts (or list of one text if batch of 1)
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            tuple: A tuple of two dictionaries (client response body, client request body)
        """
        data = {'vector_space_id': self.vector_space_id, 'data': [], 'modality': 'TEXT'}
        files = []
        for index, text in zip(batch_index_list, batch_text_list):
            data['data'].append(json.dumps('text_{}'.format(index) + '_{}'.format(text)))

        results = self.ingest(data, batch_text_list)
        for f in files:
            f.close()
        
        return results, data


    # Lookup

    def lookup_single(self, f, modality, top_k, ids=None, **kwargs) -> dict:
        """A function to search on Vecto, based on the lookup item.

        Args:
            f (str): A string of either image path or text to search on
            modality (str): The type of the file - "IMAGE" or "TEXT"
            top_k (int): The number of results to return
            ids (list): A list of vector ids to search on aka subset of vectors, defaults to None
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        """
        results = self.client.post("%s/api/v0/lookup" % vecto_base_url,
                            data={'vector_space_id': self.vector_space_id, 'modality': modality, 'top_k': top_k, 'ids': ids},
                            files={'query': f},
                            headers={"Authorization":"Bearer %s" % self.token},
                            **kwargs)

        return results


    # Update

    def update_batch_vector_embeddings(self, batch, modality, **kwargs) -> dict:
        """A function to update current vector embeddings with new one.

        Args:
            batch (list): A list of image paths or texts
            modality (str): The type of the file - "IMAGE" or "TEXT"
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        """
        vector_ids = random.sample(range(len(batch)), len(batch))
        files = []
        if modality == 'TEXT':
            for string in batch:
                files.append(io.StringIO(string))
        elif modality == 'IMAGE':
            for path in batch:
                files.append(open(path, 'rb'))
        
        results = self.client.post("%s/api/v0/update/vectors" % vecto_base_url,
                    data={'vector_space_id': self.vector_space_id, 'id': vector_ids, 'modality': modality},
                    files=[('input', ('_', f, '_')) for f in files],
                    headers={"Authorization":"Bearer %s" % self.token},
                    **kwargs)

        if modality == 'IMAGE':
            for f in files:
                f.close()

        return results

    def update_batch_vector_metadata(self, vector_ids, new_metadata, **kwargs) -> dict:
        """A function to update current vector metadata with new one.

        Args:
            vector_ids (list): A list of vector ids to update
            new_metadata (list): A list of new metadata (str) to replace the old metadata
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        """
        payload = MultipartEncoder(fields=[('vector_space_id', str(self.vector_space_id))] + 
                                            [('id', str(id)) for id in vector_ids] + 
                                            [('metadata', json.dumps(md)) for md in new_metadata])
        results = self.client.post("%s/api/v0/update/metadata" % vecto_base_url,
                    data=payload,
                    headers={"Authorization":"Bearer %s" % self.token, 'Content-Type': payload.content_type},
                    **kwargs)

        return results


    # Analogy

    def get_analogy(self, query, analogy_from, analogy_to, top_k, **kwargs) -> dict: # can be text or images
        """A function to get an analogy from Vecto.
        It is also possible to do multiple analogies in one request body.

        Args:
            query (str): Path to text file as query, e.g. orange
            analogy_from (str): Path to text file as analogy from, e.g. ocean blue
            analogy_to (str): Path to text file as analogy to, e.g. navy blue
            top_k (int): The number of results to return
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        """
        query_f = dataset_path.joinpath(query)
        afrom_f = dataset_path.joinpath(analogy_from)
        ato_f = dataset_path.joinpath(analogy_to)
        data = MultipartEncoder(fields=[
            ('vector_space_id', str(self.vector_space_id)), ('top_k', str(top_k)), ('modality', 'TEXT'),
            ('query', ('_', open(query_f, 'rb'), 'text/plain')), 
            ('from', ('_', open(afrom_f, 'rb'), 'text/plain')), # Analogy 1
            ('to', ('_', open(ato_f, 'rb'), 'text/plain')), # Analogy 1
            ('from', ('_', open(afrom_f, 'rb'), 'text/plain')), # Analogy 2
            ('to', ('_', open(ato_f, 'rb'), 'text/plain')), # Analogy 2
        ])
        results = self.client.post("%s/api/v0/analogy" % vecto_base_url,
                    data=data,
                    headers={"Authorization":"Bearer %s" % self.token, 'Content-Type': data.content_type},
                    **kwargs)

        return results

    def create_analogy(self, analogy_id, analogy_from, analogy_to, **kwargs) -> dict:
        """A function to create an analogy and store in Vecto.
        It is also possible to do multiple analogies in one request body.

        Args:
            analogy_id (int): The id for the analogy to be stored as
            analogy_from (str): Path to text file as analogy from, e.g. ocean blue
            analogy_to (str): Path to text file as analogy to, e.g. navy blue
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        """
        afrom_f = dataset_path.joinpath(analogy_from)
        ato_f = dataset_path.joinpath(analogy_to)
        data = MultipartEncoder(fields=[
            ('vector_space_id', str(self.vector_space_id)), ('analogy_id', str(analogy_id)), ('modality', 'TEXT'),
            ('from', ('_', open(afrom_f, 'rb'), 'text/plain')), # Analogy 1
            ('to', ('_', open(ato_f, 'rb'), 'text/plain')), # Analogy 1
            ('from', ('_', open(afrom_f, 'rb'), 'text/plain')), # Analogy 2
            ('to', ('_', open(ato_f, 'rb'), 'text/plain')), # Analogy 2
        ])
        results = self.client.post("%s/api/v0/analogy/create" % vecto_base_url,
                    data=data,
                    headers={"Authorization":"Bearer %s" % self.token, 'Content-Type': data.content_type},
                    **kwargs)

        return results

    def delete_analogy(self, analogy_id, **kwargs) -> dict:
        """A function to delete an analogy that is stored in Vecto.

        Args:
            analogy_id (int): The id of the analogy to be deleted
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        """
        data = MultipartEncoder(fields={'vector_space_id': str(self.vector_space_id), 'analogy_id': str(analogy_id)})
        results = self.client.post("%s/api/v0/analogy/delete" % vecto_base_url,
                    data=data,
                    headers={"Authorization":"Bearer %s" % self.token, 'Content-Type': data.content_type},
                    **kwargs)

        return results


    # Delete

    def delete_batch_vector_embeddings(self, vector_ids, **kwargs) -> dict:
        """A function to delete vector embeddings that is stored in Vecto.

        Args:
            vector_ids (list): A list of vector ids to be deleted
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        """
        payload = MultipartEncoder(fields=[('vector_space_id', str(self.vector_space_id))] + [('id', str(id)) for id in vector_ids])
        results = self.client.post("%s/api/v0/delete" % vecto_base_url,
                    data=payload,
                    headers={"Authorization":"Bearer %s" % self.token, 'Content-Type': payload.content_type},
                    **kwargs)
        print(results.status_code, results.content)

        return results

    def delete_vector_space_entries(self, **kwargs) -> dict:
        """A function to delete the current vector space in Vecto. 
        All ingested entries will be deleted as well.

        Args:
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        """
        payload = MultipartEncoder({'vector_space_id': str(self.vector_space_id)})
        results = self.client.post("%s/api/v0/delete_all" % vecto_base_url,
                    data=payload,
                    headers={"Authorization":"Bearer %s" % self.token, 'Content-Type': payload.content_type},
                    **kwargs)
        print(results.status_code, results.content)

        return results

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
        
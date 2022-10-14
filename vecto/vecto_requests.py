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


import requests
import io
from requests_toolbelt import MultipartEncoder
import random
import json

class Vecto():

    def __init__(self, token:str, vector_space_id:int or str, vecto_base_url="https://api.vecto.ai", client=requests) -> None:
        self.token = token
        self.vector_space_id = vector_space_id
        self.vecto_base_url = vecto_base_url
        self.client = client

    # Ingest
    def ingest(self, data:dict, files:list, **kwargs) -> object:
        """A function to ingest a batch of data into Vecto.
        Also works with single entry aka batch of 1.

        Args:
            data (dict): Dictionary containing regular fields
            files (list): List of file-like objects to be ingested
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            client response
        """
        results = self.client.post("%s/api/v0/index" % self.vecto_base_url,
                                   data=data,
                                   files=[('input', ('_', f, '_')) for f in files],
                                   headers={"Authorization": "Bearer %s" % self.token},
                                   **kwargs)

        return results

    def ingest_image(self, batch_path_list:list, **kwargs) -> object:
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
        
        return results

    def ingest_text(self, batch_index_list:list, batch_text_list:list, **kwargs) -> object:
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
        
        return results


    # Lookup

    def lookup(self, f:str, modality:str, top_k:int, ids:list=None, **kwargs) -> object:
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
        results = self.client.post("%s/api/v0/lookup" % self.vecto_base_url,
                            data={'vector_space_id': self.vector_space_id, 'modality': modality, 'top_k': top_k, 'ids': ids},
                            files={'query': f},
                            headers={"Authorization":"Bearer %s" % self.token},
                            **kwargs)

        return results


    # Update

    def update_vector_embeddings(self, vector_id, batch:list, modality:str, **kwargs) -> object:
        """A function to update current vector embeddings with new one.

        Args:
            batch (list): A list of image paths or texts
            modality (str): The type of the file - "IMAGE" or "TEXT"
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        """

        files = []
        if modality == 'TEXT':
            for string in batch:
                files.append(io.StringIO(string))
        elif modality == 'IMAGE':
            for path in batch:
                files.append(open(path, 'rb'))
        
        results = self.client.post("%s/api/v0/update/vectors" % self.vecto_base_url,
                    data={'vector_space_id': self.vector_space_id, 'id': vector_id, 'modality': modality},
                    files=[('input', ('_', f, '_')) for f in files],
                    headers={"Authorization":"Bearer %s" % self.token},
                    **kwargs)

        if modality == 'IMAGE':
            for f in files:
                f.close()

        return results

    def update_vector_metadata(self, vector_ids:list, new_metadata:list, **kwargs) -> object:
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
        results = self.client.post("%s/api/v0/update/metadata" % self.vecto_base_url,
                    data=payload,
                    headers={"Authorization":"Bearer %s" % self.token, 'Content-Type': payload.content_type},
                    **kwargs)

        return results


    # Analogy

    def get_analogy(self, query:str, analogy_from:str, analogy_to:str, top_k:int, **kwargs) -> object: # can be text or images
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

        data = MultipartEncoder(fields=[
            ('vector_space_id', str(self.vector_space_id)), ('top_k', str(top_k)), ('modality', 'TEXT'),
            ('query', ('_', open(query, 'rb'), 'text/plain')), 
            ('from', ('_', open(analogy_from, 'rb'), 'text/plain')), # Analogy 1
            ('to', ('_', open(analogy_to, 'rb'), 'text/plain')), # Analogy 1
            ('from', ('_', open(analogy_from, 'rb'), 'text/plain')), # Analogy 2
            ('to', ('_', open(analogy_to, 'rb'), 'text/plain')), # Analogy 2
        ])
        results = self.client.post("%s/api/v0/analogy" % self.vecto_base_url,
                    data=data,
                    headers={"Authorization":"Bearer %s" % self.token, 'Content-Type': data.content_type},
                    **kwargs)

        return results

    def create_analogy(self, analogy_id:int, analogy_from:str, analogy_to:str, **kwargs) -> object:
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

        data = MultipartEncoder(fields=[
            ('vector_space_id', str(self.vector_space_id)), ('analogy_id', str(analogy_id)), ('modality', 'TEXT'),
            ('from', ('_', open(analogy_from, 'rb'), 'text/plain')), # Analogy 1
            ('to', ('_', open(analogy_to, 'rb'), 'text/plain')), # Analogy 1
            ('from', ('_', open(analogy_from, 'rb'), 'text/plain')), # Analogy 2
            ('to', ('_', open(analogy_to, 'rb'), 'text/plain')), # Analogy 2
        ])
        results = self.client.post("%s/api/v0/analogy/create" % self.vecto_base_url,
                    data=data,
                    headers={"Authorization":"Bearer %s" % self.token, 'Content-Type': data.content_type},
                    **kwargs)

        return results

    def delete_analogy(self, analogy_id:int, **kwargs) -> object:
        """A function to delete an analogy that is stored in Vecto.

        Args:
            analogy_id (int): The id of the analogy to be deleted
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        """
        data = MultipartEncoder(fields={'vector_space_id': str(self.vector_space_id), 'analogy_id': str(analogy_id)})
        results = self.client.post("%s/api/v0/analogy/delete" % self.vecto_base_url,
                    data=data,
                    headers={"Authorization":"Bearer %s" % self.token, 'Content-Type': data.content_type},
                    **kwargs)

        return results


    # Delete

    def delete_vector_embeddings(self, vector_ids:list, **kwargs) -> object:
        """A function to delete vector embeddings that is stored in Vecto.

        Args:
            vector_ids (list): A list of vector ids to be deleted
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        """
        payload = MultipartEncoder(fields=[('vector_space_id', str(self.vector_space_id))] + [('id', str(id)) for id in vector_ids])
        results = self.client.post("%s/api/v0/delete" % self.vecto_base_url,
                    data=payload,
                    headers={"Authorization":"Bearer %s" % self.token, 'Content-Type': payload.content_type},
                    **kwargs)

        return results

    def delete_vector_space_entries(self, **kwargs) -> object:
        """A function to delete the current vector space in Vecto. 
        All ingested entries will be deleted as well.

        Args:
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        """
        payload = MultipartEncoder({'vector_space_id': str(self.vector_space_id)})
        results = self.client.post("%s/api/v0/delete_all" % self.vecto_base_url,
                    data=payload,
                    headers={"Authorization":"Bearer %s" % self.token, 'Content-Type': payload.content_type},
                    **kwargs)

        return results


    def check_common_error(self, status_code: int):
        if status_code == 400:
            raise Exception("Requested data is incorrect, please check your request.")
        elif status_code == 401:
            raise Exception("User is unauthorized, please check your access token or user/password.")
        elif status_code == 404:
            raise Exception("Object not found, please check your object id.")
        elif status_code == 405:
            raise Exception("Object is in use, please use another object.")
        elif status_code == 409:
            raise Exception("Object name already exists, please try another name.")
        else:
            raise Exception("Error status code <"+str(status_code)+">.")

class ExceptionWithCode(Exception):

    def __init__(self, code, message="Unexpected error received."):
        self.code = code
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'



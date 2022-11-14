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


from logging import raiseExceptions
import requests
import io
import os
from requests_toolbelt import MultipartEncoder
import json

from typing import NamedTuple, List
from .exceptions import VectoException, UnpairedAnalogy, ForbiddenException, LookupException


class Client:
    def __init__(self, token:str, vector_space_id:str or int, vecto_base_url: str, client) -> None:
        self.token = token
        self.vector_space_id = vector_space_id
        self.vecto_base_url = vecto_base_url
        self.client = client

    def post(self, url, data, files, kwargs=None):

        headers = {"Authorization": "Bearer %s" % self.token}
        response = self.client.post("%s/%s" % (self.vecto_base_url, url),
                                        data=data,
                                        files=files,
                                        headers=headers,
                                        **kwargs)

        return response


    def post_form(self, url, data, kwargs=None):

        headers = {"Authorization": "Bearer %s" % self.token, 'Content-Type': data.content_type}
        response = self.client.post("%s/%s" % (self.vecto_base_url, url),
                                data=data,
                                headers=headers,
                                **kwargs)

        return response

class IngestResponse(NamedTuple):
    ids: List[int]
class LookupResult(NamedTuple):
    data: object
    id: int
    similarity: float

class LookupResponse(NamedTuple):
    results: List[LookupResult]

class Vecto():


    def __init__(self, token:str=None, 
                 vector_space_id:int or str=None, 
                 vecto_base_url:str="https://api.vecto.ai", 
                 client=requests):

        if (token or vector_space_id) is None:

            try:
                token = os.environ['user_token']
                vector_space_id = os.environ['vector_space_id']
                print("Loaded token and vector_space_id from environment.")
            
            #TODO: make this into an exception
            except Exception as e:
                print("Unable to find vector space credentials.")
                print(e)

        self._client = Client(token, vector_space_id, vecto_base_url, client)

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

        files = [('input', ('_', f, '_')) for f in files]
        response = self._client.post('/api/v0/index', data, files, kwargs)

        if not response.ok:
            raise VectoException(response)

        return IngestResponse(response.json()['ids'])


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

        data={'vector_space_id': self._client.vector_space_id, 'modality': modality, 'top_k': top_k, 'ids': ids}
        files={'query': f}
        response = self._client.post('/api/v0/lookup', data, files, kwargs)

        if response.ok != True:
            raise LookupException(response)
            
        else:
            return LookupResponse(results=[LookupResult(**r) for r in response.json()['results']])


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
            for file in batch:
                files.append(open(file, 'rb'))

        #TODO Probably add vector_space_id to default call
        data={'vector_space_id': self._client.vector_space_id, 'id': vector_id, 'modality': modality}

        #TODO probably make this into a wrapper call
        temp_files=[('input', ('_', f, '_')) for f in files]
        response = self._client.post('/api/v0/update/vectors', data, temp_files, kwargs)

        if modality == 'IMAGE':
            for f in files:
                f.close()

        return response

    def update_vector_metadata(self, vector_ids:list, new_metadata:list, **kwargs) -> object:
        """A function to update current vector metadata with new one.

        Args:
            vector_ids (list): A list of vector ids to update
            new_metadata (list): A list of new metadata (str) to replace the old metadata
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        """
        data = MultipartEncoder(fields=[('vector_space_id', str(self._client.vector_space_id))] + 
                                            [('id', str(id)) for id in vector_ids] + 
                                            [('metadata', json.dumps(md)) for md in new_metadata])

        response = self._client.post_form('/api/v0/update/metadata', data, kwargs)

        if response.ok != True:
            raise VectoException(response)


    # Analogy

    @classmethod
    def multipartencoder_query_builder(self, query_category, query_string):

        if query_string is os.path.exists:
            return [(query_category, ('_', open(query_string, 'rb')))]

        else:
            return [(query_category, ('_', query_string))]

    @classmethod
    def build_analogy_query(self, analogy_fields, query, analogy_from, analogy_to):
        
        analogy_fields.extend(self.multipartencoder_query_builder("query", query))

        if analogy_from is list:
            if len(analogy_from) != len(analogy_to):
                raise UnpairedAnalogy(Exception)

        for analogy_from, analogy_to in zip(analogy_from, analogy_to):
            
            analogy_fields.extend(self.multipartencoder_query_builder("from", analogy_from))
            analogy_fields.extend(self.multipartencoder_query_builder("to", analogy_to))

        return analogy_fields


    def compute_analogy(self, query:str, analogy_from:str or list, analogy_to:str or list, top_k:int, modality:str="TEXT", **kwargs) -> object: # can be text or images
        """A function to compute an analogy using Vecto.
        It is also possible to do multiple analogies in one request body.
        The computed analogy is not stored in Vecto.

        Args:
            query (str): Path to text file as query, e.g. orange
            analogy_from (str): Path to text file as analogy from, e.g. ocean blue
            analogy_to (str): Path to text file as analogy to, e.g. navy blue
            top_k (int): The number of results to return
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        """

        
        init_analogy_fields = [('vector_space_id', str(self._client.vector_space_id)), ('top_k', str(top_k)), ('modality', modality)]
        analogy_fields = self.build_analogy_query(init_analogy_fields, query, analogy_from, analogy_to)
        
        data = MultipartEncoder(fields=analogy_fields)
                
        response = self._client.post_form('/api/v0/analogy', data, kwargs)

        return LookupResponse(results=[LookupResult(**r) for r in response.json()['results']])


    def compute_text_analogy(self, query:str, analogy_from:str or list, analogy_to:str or list, top_k:int, **kwargs) -> object: # can be text or images
        """A function to compute a text analogy using Vecto.
        It is also possible to do multiple analogies in one request body.
        The computed analogy is not stored in Vecto.

        Args:
            query (str): Path to text file as query, e.g. orange
            analogy_from (str): Path to text file as analogy from, e.g. ocean blue
            analogy_to (str): Path to text file as analogy to, e.g. navy blue
            top_k (int): The number of results to return
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        """

        init_analogy_fields = [('vector_space_id', str(self._client.vector_space_id)), ('top_k', str(top_k)), ('modality', "TEXT")]
        analogy_fields = self.build_analogy_query(init_analogy_fields, query, analogy_from, analogy_to)
        data = MultipartEncoder(fields=analogy_fields)
        
        response = self._client.post_form('/api/v0/analogy', data, kwargs)

        return response


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
            ('vector_space_id', str(self._client.vector_space_id)), ('analogy_id', str(analogy_id)), ('modality', 'TEXT'),
            ('from', ('_', open(analogy_from, 'rb'))),
            ('to', ('_', open(analogy_to, 'rb'))), 
        ])

        response = self._client.post_form('/api/v0/analogy/create', data, kwargs)

        if response.ok != True:
            raise VectoException(response)

    def delete_analogy(self, analogy_id:int, **kwargs) -> object:
        """A function to delete an analogy that is stored in Vecto.

        Args:
            analogy_id (int): The id of the analogy to be deleted
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        """
        data = MultipartEncoder(fields={'vector_space_id': str(self._client.vector_space_id), 'analogy_id': str(analogy_id)})
        response = self._client.post_form('/api/v0/analogy/delete', data, kwargs)

        if response.ok != True:
            raise VectoException(response)


    # Delete

    def delete_vector_embeddings(self, vector_ids:list, **kwargs) -> object:
        """A function to delete vector embeddings that is stored in Vecto.

        Args:
            vector_ids (list): A list of vector ids to be deleted
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        """

        data = MultipartEncoder(fields=[('vector_space_id', str(self._client.vector_space_id))] + [('id', str(id)) for id in vector_ids])
        response = self._client.post_form('/api/v0/delete', data, kwargs)
        
        if response.ok != True:
            raise VectoException(response)

    def delete_vector_space_entries(self, **kwargs) -> object:
        """A function to delete the current vector space in Vecto. 
        All ingested entries will be deleted as well.

        Args:
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        """

        data = MultipartEncoder({'vector_space_id': str(self._client.vector_space_id)})
        response = self._client.post_form('/api/v0/delete_all', data, kwargs)

        if response.ok != True:
            raise VectoException(response)
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

from typing import NamedTuple, List, IO
from .exceptions import VectoException, UnauthorizedException, UnpairedAnalogy, ForbiddenException, NotFoundException, ServiceException

class IngestResponse(NamedTuple):
    ids: List[int]
class LookupResult(NamedTuple):
    data: object
    id: int
    similarity: float
class LookupResponse(NamedTuple):
    results: List[LookupResult]

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
        if not response.ok:
            self.check_common_error(response.status_code)
        
        return response


    def post_form(self, url, data, kwargs=None):

        headers = {"Authorization": "Bearer %s" % self.token, 'Content-Type': data.content_type}
        response = self.client.post("%s/%s" % (self.vecto_base_url, url),
                                data=data,
                                headers=headers,
                                **kwargs)

        if not response.ok:
            self.check_common_error(response.status_code)

        return response


    def check_common_error(self, status_code: int):
        if status_code == 400:
            raise VectoException("Requested data is incorrect, please check your request.")
        elif status_code == 401:
            raise UnauthorizedException()
        elif status_code == 403:
            raise ForbiddenException()
        elif status_code == 404:
            raise NotFoundException()
        elif 500 <= status_code <= 599:
            raise ServiceException()
        else:
            raise VectoException("Error status code ["+str(status_code)+"].")

class Vecto():

    def __init__(self, token:str, 
                 vector_space_id:int or str, 
                 vecto_base_url:str="https://api.vecto.ai", 
                 client=requests):

        if (token is None) or (vector_space_id is None):

            raise ValueError("Both token and vector space id are necessary.")

        self._client = Client(token, vector_space_id, vecto_base_url, client)


    # Ingest

    def ingest(self, ingest_data, modality:str, **kwargs) -> object:
        """A function to ingest a batch of data into Vecto.
        Also works with single entry aka batch of 1.

        Args:
            data (dict): Dictionary containing regular fields
            files (list): List of file-like objects to be ingested
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            client response
        """
        files = [('input', ('_', r['data'], '_')) for r in ingest_data]
        metadata = [json.dumps(r['attributes']) for r in ingest_data]
        
        data = {'vector_space_id': self._client.vector_space_id, 'data': metadata, 'modality': modality}

        response = self._client.post('/api/v0/index', data, files, kwargs)

        return IngestResponse(response.json()['ids'])


    # Lookup

    def lookup(self, query:IO, modality:str, top_k:int, ids:list=None, **kwargs) -> object:
        """A function to search on Vecto, based on the lookup item.

        Args:
            query (str): A string of either image path or text to search on
            modality (str): The type of the file - "IMAGE" or "TEXT"
            top_k (int): The number of results to return
            ids (list): A list of vector ids to search on aka subset of vectors, defaults to None
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
            
        """

        data={'vector_space_id': self._client.vector_space_id, 'modality': modality, 'top_k': top_k, 'ids': ids}
        files={'query': query}
        response = self._client.post('/api/v0/lookup', data, files, kwargs)
            
        return LookupResponse(results=[LookupResult(**r) for r in response.json()['results']])


    # Update

    def update_vector_embeddings(self, embedding_data, modality:str, **kwargs) -> object:
        """A function to update current vector embeddings with new one.

        Args:
            batch (list): A list of image paths or texts
            modality (str): The type of the file - "IMAGE" or "TEXT"
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        """

        files = [('input', ('_', r['data'], '_')) for r in embedding_data]
        vector_id = [(r['ids']) for r in embedding_data]

        data={'vector_space_id': self._client.vector_space_id, 'id': vector_id, 'modality': modality}
        response = self._client.post('/api/v0/update/vectors', data, files, kwargs) 

        return response

    def update_vector_metadata(self, update_metadata, **kwargs) -> object:
        """A function to update current vector metadata with new one.

        Args:
            vector_ids (list): A list of vector ids to update
            new_metadata (list): A list of new metadata (str) to replace the old metadata
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        """

        new_metadata = [( r['attribute']) for r in update_metadata]
        vector_ids = [(r['ids']) for r in update_metadata]

        data = MultipartEncoder(fields=[('vector_space_id', str(self._client.vector_space_id))] + 
                                            [('id', str(id)) for id in vector_ids] + 
                                            [('metadata', md) for md in new_metadata])

        response = self._client.post_form('/api/v0/update/metadata', data, kwargs)


    # Analogy

    @classmethod
    def multipartencoder_query_builder(self, query_category, query_string):

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


    def compute_analogy(self, query:IO, analogy_from_to:dict or list, top_k:int, modality:str, **kwargs) -> object: # can be text or images
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

        if not type(analogy_from_to) == list:
            analogy_from_to = [analogy_from_to]

        for analogy_data in analogy_from_to:
            analogy_from = analogy_data['from']
            analogy_to = analogy_data['to']

        init_analogy_fields = [('vector_space_id', str(self._client.vector_space_id)), ('top_k', str(top_k)), ('modality', modality)]
        analogy_fields = self.build_analogy_query(init_analogy_fields, query, analogy_from, analogy_to)
        
        data = MultipartEncoder(fields=analogy_fields)
                
        response = self._client.post_form('/api/v0/analogy', data, kwargs)

        return LookupResponse(results=[LookupResult(**r) for r in response.json()['results']])

    def compute_text_analogy(self, query:IO, analogy_from_to:dict or list, top_k:int, **kwargs) -> object: 
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
        response = self.compute_analogy(query, analogy_from_to, top_k, 'TEXT')

        return response

        #TODO: call the other compute analogy and pass text as modality

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
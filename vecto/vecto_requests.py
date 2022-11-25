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
from .exceptions import VectoException, UnauthorizedException, UnpairedAnalogy, ForbiddenException, NotFoundException, ServiceException, InvalidModality

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
            raise VectoException("Submitted data is incorrect, please check your request.")
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

    def ingest(self, ingest_data: dict or list, modality:str, **kwargs) -> IngestResponse:
        """A function to ingest data into Vecto. 
            
        Args:    
            ingest_data (dict or list): Ingest data must follow the ingest data format: 
                {
                    'data': data, 
                    'attributes': metadata
                }
                Where data should be an IO file-like object.
                You can use open(path, 'rb') for IMAGE queries and io.StringIO(text) for TEXT queries.
            modality (str): 'IMAGE' or 'TEXT'
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            IngestResponse: named tuple that contains the list of index of ingested objects.
                IngestResponse(NamedTuple):
                    ids: List[int]
        """
        if type(ingest_data) != list:
            ingest_data = [ingest_data]

        if modality != 'IMAGE' and modality != 'TEXT':
            raise InvalidModality()

        files = [('input', ('_', r['data'], '_')) for r in ingest_data]
        metadata = [json.dumps(r['attributes']) for r in ingest_data]
        
        data = {'vector_space_id': self._client.vector_space_id, 'data': metadata, 'modality': modality}

        response = self._client.post('/api/v0/index', data, files, kwargs)

        return IngestResponse(response.json()['ids'])


    # Lookup

    def lookup(self, query:IO, modality:str, top_k:int, ids:list=None, **kwargs) -> LookupResponse:
        """A function to search on Vecto, based on the lookup item.

        Args:
            query (IO): A IO file-like object. 
                        You can use open(path, 'rb') for IMAGE queries and io.StringIO(text) for TEXT queries.

            modality (str): The type of the file - "IMAGE" or "TEXT"
            top_k (int): The number of results to return
            ids (list): A list of vector ids to search on aka subset of vectors, defaults to None
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            LookupResponse: named tuple that contains a list of LookupResult named tuples.
                results: List[LookupResult]
            
            where LookResult is named tuple with `data`, `id`, and `similarity` keys.
                data: object
                id: int
                similarity: float
        """

        if modality != 'IMAGE' and modality != 'TEXT':
            raise InvalidModality()

        data={'vector_space_id': self._client.vector_space_id, 'modality': modality, 'top_k': top_k, 'ids': ids}
        files={'query': query}
        response = self._client.post('/api/v0/lookup', data, files, kwargs)
            
        return LookupResponse(results=[LookupResult(**r) for r in response.json()['results']])


    # Update

    def update_vector_embeddings(self, embedding_data: dict or list, modality:str, **kwargs) -> object:
        """A function to update current vector embeddings with new one.

        Args:
            embedding_data (dict or list): dict or a list dicts that contains the embedding data to be updated. 
            It must have the following keys: 
                {
                    'data': data, 
                    'id': ids
                }

            modality (str): The type of the file - "IMAGE" or "TEXT"
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        """

        if type(embedding_data) != list:
            embedding_data = [embedding_data]

        if modality != 'IMAGE' and modality != 'TEXT':
            raise InvalidModality()

        files = [('input', ('_', r['data'], '_')) for r in embedding_data]
        vector_id = [(r['id']) for r in embedding_data]

        data={'vector_space_id': self._client.vector_space_id, 'id': vector_id, 'modality': modality}
        response = self._client.post('/api/v0/update/vectors', data, files, kwargs) 

        return response


    def update_vector_metadata(self, update_metadata: dict or list, **kwargs) -> object:
        """A function to update current vector metadata with new one.

        Args:
            update_metadata (dict or list) : metadata to be updated. It must follow the format: 
                {
                    'attribute': metadata, 
                    'id': id
                }

                where: 
                    'attribute`: metadata to update
                    'id': vector id to update

            **kwargs: Other keyword arguments for clients other than `requests`

        """

        if type(update_metadata) != list:
            update_metadata = [update_metadata]

        new_metadata = [( r['attribute']) for r in update_metadata]
        vector_ids = [(r['id']) for r in update_metadata]

        data = MultipartEncoder(fields=[('vector_space_id', str(self._client.vector_space_id))] + 
                                            [('id', str(id)) for id in vector_ids] + 
                                            [('metadata', md) for md in new_metadata])

        self._client.post_form('/api/v0/update/metadata', data, kwargs)


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


    def compute_analogy(self, query:IO, analogy_start_end:dict or list, top_k:int, modality:str, **kwargs) -> LookupResponse: # can be text or images
        """A function to compute an analogy using Vecto.
        It is also possible to do multiple analogies in one request body.
        The computed analogy is not stored in Vecto.

        Args:
            query (IO): query in the form of an IO object query.
            analogy_start_end (dict or list): start and end analogy to be computed. 
            It must follow the format: 
                
                {
                    'start': analogy_start, 
                    'end': analogy_end
                }

                where: 

                    'start`: the starting point of the analogy
                    'end': the ending point of the analogy

                You can use open(path, 'rb') for IMAGE queries and io.StringIO(text) for TEXT queries.

            top_k (int): The number of results to return
            modality (str): The type of the file, 'IMAGE' or 'TEXT'
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            LookupResponse: named tuple that contains a list of LookupResult named tuples.
                results: List[LookupResult]
            
            where LookResult is named tuple with `data`, `id`, and `similarity` keys.
                data: object
                id: int
                similarity: float
        """

        if not type(analogy_start_end) == list:
            analogy_start_end = [analogy_start_end]

        for analogy_data in analogy_start_end:
            analogy_from = analogy_data['start']
            analogy_to = analogy_data['end']

        init_analogy_fields = [('vector_space_id', str(self._client.vector_space_id)), ('top_k', str(top_k)), ('modality', modality)]
        analogy_fields = self.build_analogy_query(init_analogy_fields, query, analogy_from, analogy_to)
        
        data = MultipartEncoder(fields=analogy_fields)
                
        response = self._client.post_form('/api/v0/analogy', data, kwargs)

        return LookupResponse(results=[LookupResult(**r) for r in response.json()['results']])


    def compute_text_analogy(self, query:IO, analogy_start_end:dict or list, top_k:int, **kwargs) -> LookupResponse: 
        """A function to compute an Text analogy using Vecto.
        It is also possible to do multiple analogies in one request body.
        The computed analogy is not stored in Vecto.

        Args:
            query (IO): query in the form of an IO object query.
            analogy_start_end (dict or list): start and end analogy to be computed. 
            It must follow the format: 
                
                {
                    'start': analogy_start, 
                    'end': analogy_end
                }

                where: 

                    'start`: the starting point of the analogy
                    'end': the ending point of the analogy

                Use io.StringIO(text) for TEXT queries.

            top_k (int): The number of results to return
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            LookupResponse: named tuple that contains a list of LookupResult named tuples.
                results: List[LookupResult]
            
            where LookResult is named tuple with `data`, `id`, and `similarity` keys.
                data: object
                id: int
                similarity: float
        """

        response = self.compute_analogy(query, analogy_start_end, top_k, 'TEXT')

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

        self._client.post_form('/api/v0/analogy/create', data, kwargs)


    def delete_analogy(self, analogy_id:int, **kwargs) -> object:
        """A function to delete an analogy that is stored in Vecto.

        Args:
            analogy_id (int): The id of the analogy to be deleted
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        """
        data = MultipartEncoder(fields={'vector_space_id': str(self._client.vector_space_id), 'analogy_id': str(analogy_id)})
        self._client.post_form('/api/v0/analogy/delete', data, kwargs)

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
        self._client.post_form('/api/v0/delete_all', data, kwargs)
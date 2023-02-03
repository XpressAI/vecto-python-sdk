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
from requests_toolbelt import MultipartEncoder
import json

from typing import IO, List, Union, NamedTuple
from .exceptions import ( VectoException, UnauthorizedException, UnpairedAnalogy, 
                        ForbiddenException, NotFoundException, ServiceException, 
                        InvalidModality, ConsumedResourceException )

import sys

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

class VectoIngestData(TypedDict):
    '''A named tuple that contains the expected Vecto input format.
    For data, you may use open(path, 'rb') for IMAGE queries or io.StringIO(text) for TEXT queries.
    You may append as many metadata to attributes as needed.'''
    data: IO
    attributes: dict

class VectoEmbeddingData(TypedDict):
    '''A named tuple that contains the expected Vecto embedding format for updating.
    For data, you may use open(path, 'rb') for IMAGE queries or io.StringIO(text) for TEXT queries.
    '''
    id: int
    data: IO

class VectoMetadata(TypedDict):
    '''A named tuple that contains the expected Vecto metadata format for updating.
    You may append as many metadata to attributes as needed'''
    id: int
    attributes: dict

class VectoAnalogyStartEnd(TypedDict):
    '''A named tuple that contains the expected Vecto analogy start-end input format.
    For data, you may use open(path, 'rb') for IMAGE queries or io.StringIO(text) for TEXT queries.'''
    start: IO
    end: IO

class IngestResponse(NamedTuple):
    '''A named tuple that contains a list of ids of ingested data.'''
    ids: List[int]

class LookupResult(NamedTuple):
    '''A named tuple that contains the lookup result content: data (metadata), id, and similarity.'''
    data: object
    id: int
    similarity: float

class LookupResponse(NamedTuple):
    '''A named tuple that contains a list of LookupResults.'''
    results: List[LookupResult]

class Client:
    def __init__(self, token:str, vecto_base_url: str, client) -> None:
        self.token = token
        self.vecto_base_url = vecto_base_url
        self.client = client

    def post(self, url, data, files, kwargs=None):

        self.validate_input(url=url, data=data, files=files)

        headers = {"Authorization": "Bearer %s" % self.token}
        response = self.client.post("%s/%s" % (self.vecto_base_url, url),
                                        data=data,
                                        files=files,
                                        headers=headers,
                                        **kwargs)

        self.check_common_error(response)

        return response


    def post_form(self, url, data, kwargs=None):

        self.validate_input(url=url, data=data)

        headers = {"Authorization": "Bearer %s" % self.token, 'Content-Type': data.content_type}
        response = self.client.post("%s/%s" % (self.vecto_base_url, url),
                                data=data,
                                headers=headers,
                                **kwargs)

        self.check_common_error(response)
        
        return response

    def validate_input(self, url=None, data=None, files=None, headers=None):

        def _check_input_buffer(files):
            '''Currently ingest files are formatted as:
            [('input', ('_', <_io.BufferedReader name='file.png'>, '_'))]
            '''

            for file in files:
                for ingest_input in file:
                    for buffer in ingest_input:
                        if str(buffer.__class__.__name__) == "BufferedReader":
                            if buffer.peek() == b'':
                                raise ConsumedResourceException()

        if files != None:
            _check_input_buffer(files)


    def check_common_error(self, response):

        if response.ok:
            return

        status_code = response.status_code

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
                 vector_space_id:Union[int, str], 
                 vecto_base_url:str="https://api.vecto.ai", 
                 client=requests):

        if (token is None) or (vector_space_id is None):

            raise ValueError("Both token and vector space id are necessary.")

        self.vector_space_id = vector_space_id
        self._client = Client(token, vecto_base_url, client)

    ##########
    # Ingest #
    ##########

    def ingest(self, ingest_data: Union[VectoIngestData, List[VectoIngestData]], modality:str, **kwargs) -> IngestResponse:
        '''A function to ingest data into Vecto. 
            
        Args:    
            ingest_data (VectoIngestData or list of VectoIngestData): you can also provide a dict, but ensure that it complies with VectoIngestData.
            modality (str): 'IMAGE' or 'TEXT'
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            IngestResponse: named tuple that contains the list of index of ingested objects.
        '''
        if type(ingest_data) != list:
            ingest_data = [ingest_data]

        if modality != 'IMAGE' and modality != 'TEXT':
            raise InvalidModality()

        files = [('input', ('_', r['data'], '_')) for r in ingest_data]
        metadata = [json.dumps(r['attributes']) for r in ingest_data]
        
        data = {'data': metadata, 'modality': modality}
        response = self._client.post(('/api/v0/space/%s/index' % self.vector_space_id), data, files, kwargs)

        return IngestResponse(response.json()['ids'])

    ##########
    # Lookup #
    ##########

    def lookup(self, query:IO, modality:str, top_k:int, ids:list=None, **kwargs) -> LookupResponse:
        '''A function to search on Vecto, based on the lookup item.

        Args:
            query (IO): A IO file-like object. 
                        You can use open(path, 'rb') for IMAGE queries and io.StringIO(text) for TEXT queries.
            modality (str): The type of the file - "IMAGE" or "TEXT"
            top_k (int): The number of results to return
            ids (list): A list of vector ids to search on aka subset of vectors, defaults to None
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            LookupResponse: named tuple that contains a list of LookupResult named tuples.            
            where LookResult is named tuple with `data`, `id`, and `similarity` keys.
        '''

        if modality != 'IMAGE' and modality != 'TEXT':
            raise InvalidModality()

        data={'modality': modality, 'top_k': top_k, 'ids': ids}
        files={'query': query}
        response = self._client.post(('/api/v0/space/%s/lookup' % self.vector_space_id), data, files, kwargs)
        
        if not response.json()['results']:
            return  LookupResponse(results=[])

        return LookupResponse(results=[LookupResult(**r) for r in response.json()['results']])

    ##########
    # Update #
    ##########

    def update_vector_embeddings(self, embedding_data: Union[VectoEmbeddingData, List[VectoEmbeddingData]], modality:str, **kwargs) -> object:
        '''A function to update current vector embeddings with new one.

        Args:
            embedding_data (VectoEmbeddingData or list of VectoEmbeddingData): data that contains the embedding data to be updated. 
            modality (str): The type of the file - "IMAGE" or "TEXT"
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        '''

        if type(embedding_data) != list:
            embedding_data = [embedding_data]

        if modality != 'IMAGE' and modality != 'TEXT':
            raise InvalidModality()

        vector_id = [(r['id']) for r in embedding_data]
        files = [('input', ('_', r['data'], '_')) for r in embedding_data]

        data={'vector_space_id': self.vector_space_id, 'id': vector_id, 'modality': modality}
        response = self._client.post(('/api/v0/space/%s/update/vectors' % self.vector_space_id), data, files, kwargs) 

        return response


    def update_vector_metadata(self, update_metadata: Union[VectoMetadata, List[VectoMetadata]], **kwargs) -> object:
        '''A function to update current vector metadata with new one.

        Args:
            update_metadata (VectoMetadata or list of VectoMetadata) : metadata to be updated.
            **kwargs: Other keyword arguments for clients other than `requests`

        '''

        if type(update_metadata) != list:
            update_metadata = [update_metadata]

        vector_ids = [(r['id']) for r in update_metadata]
        new_metadata = [( r['attribute']) for r in update_metadata]

        data = MultipartEncoder(fields=[('vector_space_id', str(self.vector_space_id))] + 
                                            [('id', str(id)) for id in vector_ids] + 
                                            [('metadata', md) for md in new_metadata])

        self._client.post_form(('/api/v0/space/%s/update/metadata' % self.vector_space_id), data, kwargs)

    ###########
    # Analogy #
    ###########

    @classmethod
    def multipartencoder_query_builder(self, query_category, query_string):
        '''Returns a list of tuples which is expected by analogy form post.
            EG: [('from', ('_', 'tests/demo_dataset/blue.txt'))]'''
        return [(query_category, ('_', query_string))]

    @classmethod
    def build_analogy_query(self, analogy_fields, query, start, end):
        '''Accepts the init analogy field, file-like objects for query, start and end, 
        and returns a list of tuples which contains one query tuple, 
        and one or more start and end tuples formatted for analogy query.
        
        Sample output:
        [('vector_space_id', '28148'), ('top_k', '20'), ('modality', 'TEXT'),
         ('query', ('_', 'King')), 
         ('from', ('_', 'Male')), ('to', ('_', 'Female')), 
         ('from', ('_', 'Husband')), ('to', ('_', 'Wife'))]
        '''
        analogy_fields.extend(self.multipartencoder_query_builder("query", query))

        if type(start) == list:
            if len(start) != len(end):
                raise UnpairedAnalogy()
        else:
            start = [start]
            end = [end]

        for start, end in zip(start, end):
            
            analogy_fields.extend(self.multipartencoder_query_builder("from", start))
            analogy_fields.extend(self.multipartencoder_query_builder("to", end))

        return analogy_fields


    def compute_analogy(self, query:IO, analogy_start_end:Union[VectoAnalogyStartEnd, List[VectoAnalogyStartEnd]], top_k:int, modality:str, **kwargs) -> LookupResponse: # can be text or images
        '''A function to compute an analogy using Vecto.
        It is also possible to do multiple analogies in one request body.
        The computed analogy is not stored in Vecto.

        Args:
            query (IO): query in the form of an IO object query.
            analogy_start_end (VectoAnalogyStartEnd or list of VectoAnalogyStartEnd): start and end analogy to be computed.
            Use open(path, 'rb') for IMAGE or io.StringIO(text) for TEXT analogies.
            top_k (int): The number of results to return
            modality (str): The type of the file, 'IMAGE' or 'TEXT'
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            LookupResponse: named tuple that contains a list of LookupResult named tuples.
            where LookResult is named tuple with `data`, `id`, and `similarity` keys.
        '''
        
        if not type(analogy_start_end) == list:
            analogy_start_end = [analogy_start_end]

        start = []
        end = []

        for analogy_data in analogy_start_end:
            start.append(analogy_data['start'])
            end.append(analogy_data['end'])

        init_analogy_fields = [('vector_space_id', str(self.vector_space_id)), ('top_k', str(top_k)), ('modality', modality)]
        analogy_fields = self.build_analogy_query(init_analogy_fields, query, start, end)
        
        data = MultipartEncoder(fields=analogy_fields)
                
        response = self._client.post_form(('/api/v0/space/%s/analogy' % self.vector_space_id), data, kwargs)
        
        return LookupResponse(results=[LookupResult(**r) for r in response.json()['results']])


    def compute_text_analogy(self, query:IO, analogy_start_end:Union[VectoAnalogyStartEnd, List[VectoAnalogyStartEnd]], top_k:int, **kwargs) -> LookupResponse: 
        '''A function to compute a Text analogy using Vecto.
        It is also possible to do multiple analogies in one request body.
        The computed analogy is not stored in Vecto.

        Args:
            query (IO): query in the form of an IO object query.
            analogy_start_end (VectoAnalogyStartEnd or list of VectoAnalogyStartEnd): start and end analogy to be computed. 
                You may use io.StringIO(text) for TEXT data.
            top_k (int): The number of results to return
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            LookupResponse: named tuple that contains a list of LookupResult named tuples.
            where LookResult is named tuple with `data`, `id`, and `similarity` keys.

        '''

        response = self.compute_analogy(query, analogy_start_end, top_k, 'TEXT')

        return response

    def compute_image_analogy(self, query:IO, analogy_start_end:Union[VectoAnalogyStartEnd, List[VectoAnalogyStartEnd]], top_k:int, **kwargs) -> LookupResponse: 
        '''A function to compute an IMAGE analogy using Vecto.
        It is also possible to do multiple analogies in one request body.
        The computed analogy is not stored in Vecto.

        Args:
            query (IO): query in the form of an IO object query.
            analogy_start_end (VectoAnalogyStartEnd or list of VectoAnalogyStartEnd): start and end analogy to be computed. 
                Use open(path, 'rb') for IMAGE data.
            top_k (int): The number of results to return
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            LookupResponse: named tuple that contains a list of LookupResult named tuples.
            where LookResult is named tuple with `data`, `id`, and `similarity` keys.
        '''

        response = self.compute_analogy(query, analogy_start_end, top_k, 'IMAGE')

        return response

    def create_analogy(self, analogy_id:int, start:str, end:str, **kwargs) -> object:
        '''A function to create an analogy and store in Vecto.
        It is also possible to do multiple analogies in one request body.

        Args:
            analogy_id (int): The id for the analogy to be stored as
            start (str): Path to text file as analogy from, e.g. ocean blue
            end (str): Path to text file as analogy to, e.g. navy blue
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        '''

        data = MultipartEncoder(fields=[
            ('vector_space_id', str(self.vector_space_id)), ('analogy_id', str(analogy_id)), ('modality', 'TEXT'),
            ('from', ('_', open(start, 'rb'))),
            ('to', ('_', open(end, 'rb'))), 
        ])

        self._client.post_form(('/api/v0/space/%s/analogy/create' % self.vector_space_id), data, kwargs)


    def delete_analogy(self, analogy_id:int, **kwargs) -> object:
        '''A function to delete an analogy that is stored in Vecto.

        Args:
            analogy_id (int): The id of the analogy to be deleted
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        '''
        data = MultipartEncoder(fields={'vector_space_id': str(self.vector_space_id), 'analogy_id': str(analogy_id)})
        self._client.post_form(('/api/v0/space/%s/analogy/delete' % self.vector_space_id), data, kwargs)

    # Delete

    def delete_vector_embeddings(self, vector_ids:list, **kwargs) -> object:
        '''A function to delete vector embeddings that is stored in Vecto.

        Args:
            vector_ids (list): A list of vector ids to be deleted
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        '''

        data = MultipartEncoder(fields=[('vector_space_id', str(self.vector_space_id))] + [('id', str(id)) for id in vector_ids])
        response = self._client.post_form(('/api/v0/space/%s/delete' % self.vector_space_id), data, kwargs)
        

    def delete_vector_space_entries(self, **kwargs) -> object:
        '''A function to delete the current vector space in Vecto. 
        All ingested entries will be deleted as well.

        Args:
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            dict: Client response body
        '''

        data = MultipartEncoder({'vector_space_id': str(self.vector_space_id)})
        self._client.post_form(('/api/v0/space/%s/delete_all' % self.vector_space_id), data, kwargs)
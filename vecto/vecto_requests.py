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
import math
import io
import os
import pathlib

from typing import IO, List, Union
from .exceptions import ( VectoException, UnauthorizedException, UnpairedAnalogy, 
                        ForbiddenException, NotFoundException, ServiceException, 
                        InvalidModality, ConsumedResourceException, ModelNotFoundException )

from .schema import (VectoIngestData, VectoEmbeddingData, VectoAttribute, VectoAnalogyStartEnd,
                    IngestResponse, LookupResult, VectoModel, VectoVectorSpace, VectoUser,
                    VectoToken, VectoNewTokenResponse, MODEL_MAP)


class Client:
    def __init__(self, token:str, vecto_base_url: str, client) -> None:
        self.token = token
        self.vecto_base_url = vecto_base_url
        self.client = client
        

    def get(self, url, **kwargs):

        self.validate_input(url=url)
        headers = {"Authorization": "Bearer %s" % self.token}
        response = self.client.get("%s/%s" % (self.vecto_base_url, url),
                                        headers=headers,
                                        **kwargs)
        
        self.check_common_error(response)
        return response


    def put(self, url, json=None, data=None, files=None, **kwargs):

        self.validate_input(url=url, data=data, files=files)
        headers = {"Authorization": "Bearer %s" % self.token}
        response = self.client.put("%s/%s" % (self.vecto_base_url, url),
                                        headers=headers,
                                        **kwargs)
        self.check_common_error(response)
        return response
    
    def put_json(self, url, json, **kwargs):

        self.validate_input(url=url)
        headers = {"Authorization": "Bearer %s" % self.token, 'Content-Type': 'application/json'}
        response = self.client.put("%s/%s" % (self.vecto_base_url, url),
                                        json=json,
                                        headers=headers,
                                        **kwargs)
        self.check_common_error(response)
        return response


    def delete(self, url, data=None, files=None, **kwargs):

        self.validate_input(url=url, data=data, files=files)
        headers = {"Authorization": "Bearer %s" % self.token}
        response = self.client.delete("%s/%s" % (self.vecto_base_url, url),
                                        data=data,
                                        files=files,
                                        headers=headers,
                                        **kwargs)
        self.check_common_error(response)
        return response

    def post(self, url, data, files, **kwargs):

        self.validate_input(url=url, data=data, files=files)
        headers = {"Authorization": "Bearer %s" % self.token}
        response = self.client.post("%s/%s" % (self.vecto_base_url, url),
                                        data=data,
                                        files=files,
                                        headers=headers,
                                        **kwargs)
        self.check_common_error(response)
        return response

    def post_json(self, url, json, **kwargs):

        self.validate_input(url=url)
        headers = {"Authorization": "Bearer %s" % self.token, 'Content-Type': 'application/json'}
        response = self.client.post("%s/%s" % (self.vecto_base_url, url),
                                        json=json,
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
        attribute = [json.dumps(r['attributes']) for r in ingest_data]
        
        data = {'attributes': attribute, 'modality': modality}

        response = self._client.post(('/api/v0/space/%s/index' % self.vector_space_id), data=data, files=files, **kwargs)

        return IngestResponse(response.json()['ids'])

    ##########
    # Lookup #
    ##########

    def lookup(self, query:IO, modality:str, top_k:int, ids:list=None, **kwargs) -> List:
        '''A function to search on Vecto, based on the lookup item.

        Args:
            query (IO): A IO file-like object. 
                        You can use open(path, 'rb') for IMAGE queries and io.StringIO(text) for TEXT queries.
            modality (str): The type of the file - "IMAGE" or "TEXT"
            top_k (int): The number of results to return
            ids (list): A list of vector ids to search on aka subset of vectors, defaults to None
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            list of LookupResult named tuples, where LookResult is named tuple with `data`, `id`, and `similarity` keys.
        '''

        if modality != 'IMAGE' and modality != 'TEXT':
            raise InvalidModality()

        data={'modality': modality, 'top_k': top_k, 'ids': ids}
        files={'query': query}
        response = self._client.post(('/api/v0/space/%s/lookup' % self.vector_space_id), data=data, files=files, **kwargs)
        
        if not response.json()['results']:
            return []

        return [LookupResult(**r) for r in response.json()['results']]

    def url_to_binary_stream(self, url: str) -> io.BytesIO:
 
        from urllib.request import urlopen
        from urllib.parse import urlparse

        def is_url(s: str) -> bool:
            try:
                result = urlparse(s)
                return all([result.scheme, result.netloc])
            except ValueError:
                return False

        if is_url(url):
            content = urlopen(url).read()
            binary_stream = io.BytesIO(content)
            return binary_stream
        else:
            raise ValueError(f'Invalid URL: {url}')
    

    def lookup_image_from_url(self, query:str, top_k:int, ids:list=None, **kwargs) -> List:
        '''A function to perform image search on Vecto by passing it an url.

        Args:
            query (str): url str to an image resource
            top_k (int): The number of results to return
            ids (list): A list of vector ids to search on aka subset of vectors, defaults to None
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            list of LookupResult named tuples, where LookResult is named tuple with `data`, `id`, and `similarity` keys.
        '''

        content = self.url_to_binary_stream(query)
        response = self.lookup(content, modality='IMAGE', top_k=top_k, ids=ids)

        return response
    

    def lookup_image_from_filepath(self, query:Union[str, pathlib.Path, os.PathLike], top_k:int, ids:list=None, **kwargs) -> List:
        '''A function to perform image search on Vecto by passing it an image path.

        Args:
            query (Union[str, pathlib.Path, os.PathLike]): the path to the image query
            top_k (int): The number of results to return
            ids (list): A list of vector ids to search on aka subset of vectors, defaults to None
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            list of LookupResult named tuples, where LookResult is named tuple with `data`, `id`, and `similarity` keys.
        '''

        if os.path.exists(str(query)):
            query = open(query, 'rb')

        else:
            raise FileNotFoundError("The file was not found.")

        response = self.lookup(query, modality='IMAGE', top_k=top_k, ids=ids)

        return response

    def lookup_image_from_binary(self, query:IO, top_k:int, ids:list=None, **kwargs) -> List:
        '''A function to perform image search on Vecto.

        Args:
            query (IO): query already in IO form
            top_k (int): The number of results to return
            ids (list): A list of vector ids to search on aka subset of vectors, defaults to None
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            list of LookupResult named tuples, where LookResult is named tuple with `data`, `id`, and `similarity` keys.
        '''

        response = self.lookup(query, modality='IMAGE', top_k=top_k, ids=ids)

        return response

    def lookup_text_from_str(self, query:str, top_k:int, ids:list=None, **kwargs) -> List:
        '''A function to perform text search on Vecto by passing it a string.

        Args:
            query(str): query in string
            top_k (int): The number of results to return
            ids (list): A list of vector ids to search on aka subset of vectors, defaults to None
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            list of LookupResult named tuples, where LookResult is named tuple with `data`, `id`, and `similarity` keys.
        '''
 
        response = self.lookup(io.StringIO(query), modality='TEXT', top_k=top_k, ids=ids)

        return response


    def lookup_text_from_filepath(self, query:Union[str, pathlib.Path, os.PathLike], top_k:int, ids:list=None, **kwargs) -> List:
        '''A function to perform text search on Vecto by providing it a readable text file path.

        Args:
            query (Union[str, IO, pathlib.Path, os.PathLike]): 
                If `query` is a path-like object or file-like object, it will be read as a text file.
            top_k (int): The number of results to return
            ids (list): A list of vector ids to search on aka subset of vectors, defaults to None
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            list of LookupResult named tuples, where LookResult is named tuple with `data`, `id`, and `similarity` keys.
        '''

        if os.path.exists(str(query)):
            query = open(query, 'rb')

        else:
            raise FileNotFoundError("The file was not found.")

        response = self.lookup(query, modality='TEXT', top_k=top_k, ids=ids)

        return response


    def lookup_text_from_url(self, query:str, top_k:int, ids:list=None, **kwargs) -> List:
        '''A function to perform text search on Vecto by passing it an url.

        Args:
            query (str): url str to an text resource
            top_k (int): The number of results to return
            ids (list): A list of vector ids to search on aka subset of vectors, defaults to None
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            list of LookupResult named tuples.           , where LookResult is named tuple with `data`, `id`, and `similarity` keys.
        '''

        content = self.url_to_binary_stream(query)
        response = self.lookup(content, modality='TEXT', top_k=top_k, ids=ids)

        return response
    

    def lookup_text_from_binary(self, query:IO, top_k:int, ids:list=None, **kwargs) -> List:
        '''A function to perform text search on Vecto by passing it an IO object.

        Args:
            query (IO): query already in IO form
            top_k (int): The number of results to return
            ids (list): A list of vector ids to search on aka subset of vectors, defaults to None
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            list of LookupResult named tuples.           , where LookResult is named tuple with `data`, `id`, and `similarity` keys.
        '''

        response = self.lookup(query, modality='TEXT', top_k=top_k, ids=ids)

        return response

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
        response = self._client.post(('/api/v0/space/%s/update/vectors' % self.vector_space_id), data=data, files=files, **kwargs) 

        return response


    def update_vector_attribute(self, update_attribute: Union[VectoAttribute, List[VectoAttribute]], **kwargs) -> object:
        '''A function to update current vector attribute with new one.

        Args:
            update_attribute (VectoAttribute or list of VectoAttribute) : attribute to be updated.
            **kwargs: Other keyword arguments for clients other than `requests`

        '''

        if type(update_attribute) != list:
            update_attribute = [update_attribute]

        vector_ids = [(r['id']) for r in update_attribute]
        new_attribute = [( r['attributes']) for r in update_attribute]

        data = MultipartEncoder(fields=[('vector_space_id', str(self.vector_space_id))] + 
                                            [('id', str(id)) for id in vector_ids] + 
                                            [('attributes', md) for md in new_attribute])

        self._client.post_form(('/api/v0/space/%s/update/attributes' % self.vector_space_id), data, kwargs)

    ###########
    # Analogy #
    ###########

    @classmethod
    def _multipartencoder_query_builder(self, query_category, query_string):
        '''Returns a list of tuples which is expected by analogy form post.
            EG: [('from', ('_', 'tests/demo_dataset/blue.txt'))]'''
        return [(query_category, ('_', query_string))]

    @classmethod
    def _build_analogy_query(self, analogy_fields, query, start, end):
        '''Accepts the init analogy field, file-like objects for query, start and end, 
        and returns a list of tuples which contains one query tuple, 
        and one or more start and end tuples formatted for analogy query.
        
        Sample output:
        [('vector_space_id', '28148'), ('top_k', '20'), ('modality', 'TEXT'),
         ('query', ('_', 'King')), 
         ('start', ('_', 'Male')), ('end', ('_', 'Female')), 
         ('start', ('_', 'Husband')), ('end', ('_', 'Wife'))]
        '''
        analogy_fields.extend(self._multipartencoder_query_builder("query", query))

        if type(start) == list:
            if len(start) != len(end):
                raise UnpairedAnalogy()
        else:
            start = [start]
            end = [end]

        for start, end in zip(start, end):
            
            analogy_fields.extend(self._multipartencoder_query_builder("start", start))
            analogy_fields.extend(self._multipartencoder_query_builder("end", end))

        return analogy_fields


    def compute_analogy(self, query:IO, analogy_start_end:Union[VectoAnalogyStartEnd, List[VectoAnalogyStartEnd]], top_k:int, modality:str, **kwargs) -> List : # can be text or images
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
            list of LookupResult named tuples, where LookResult is named tuple with `data`, `id`, and `similarity` keys.
        '''
        
        if not type(analogy_start_end) == list:
            analogy_start_end = [analogy_start_end]

        start = []
        end = []

        for analogy_data in analogy_start_end:
            start.append(analogy_data['start'])
            end.append(analogy_data['end'])

        init_analogy_fields = [('vector_space_id', str(self.vector_space_id)), ('top_k', str(top_k)), ('modality', modality)]
        analogy_fields = self._build_analogy_query(init_analogy_fields, query, start, end)
        
        data = MultipartEncoder(fields=analogy_fields)

        response = self._client.post_form(('/api/v0/space/%s/analogy' % self.vector_space_id), data, kwargs)
        
        return[LookupResult(**r) for r in response.json()['results']]


    def compute_text_analogy(self, query:IO, analogy_start_end:Union[VectoAnalogyStartEnd, List[VectoAnalogyStartEnd]], top_k:int, **kwargs) -> List: 
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
            list of LookupResult named tuples, where LookResult is named tuple with `data`, `id`, and `similarity` keys.

        '''

        response = self.compute_analogy(query, analogy_start_end, top_k, 'TEXT')

        return response

    def compute_image_analogy(self, query:IO, analogy_start_end:Union[VectoAnalogyStartEnd, List[VectoAnalogyStartEnd]], top_k:int, **kwargs) -> List: 
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
            list of LookupResult named tuples, where LookResult is named tuple with `data`, `id`, and `similarity` keys.
        '''

        response = self.compute_analogy(query, analogy_start_end, top_k, 'IMAGE')

        return response

    def create_analogy(self, start:str, end:str, **kwargs) -> object:
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
            ('modality', 'TEXT'),
            ('start', ('_', open(start, 'rb'))),
            ('end', ('_', open(end, 'rb'))), 
        ])

        self._client.post_form(('/api/v0/space/%s/analogy' % self.vector_space_id), data, kwargs)


    def delete_analogy(self, analogy_id:int, **kwargs):
        '''A function to delete an analogy that is stored in Vecto.

        Args:
            analogy_id (int): The id of the analogy to be deleted
            **kwargs: Other keyword arguments for clients other than `requests`
        '''
        data = MultipartEncoder(fields={'vector_space_id': str(self.vector_space_id), 'analogy_id': str(analogy_id)})
        self._client.post_form(('/api/v0/space/%s/analogy/delete' % self.vector_space_id), data, kwargs)

    # Delete

    def delete_vector_embeddings(self, vector_ids:list, **kwargs)
        '''A function to delete vector embeddings that is stored in Vecto.

        Args:
            vector_ids (list): A list of vector ids to be deleted
            **kwargs: Other keyword arguments for clients other than `requests`
        '''

        data = MultipartEncoder(fields=[('vector_space_id', str(self.vector_space_id))] + [('id', str(id)) for id in vector_ids])
        self._client.post_form(('/api/v0/space/%s/delete' % self.vector_space_id), data, kwargs)
        

    def delete_vector_space_entries(self, **kwargs)
        '''A function to delete the current vector space in Vecto. 
        All ingested entries will be deleted as well.

        Args:
            **kwargs: Other keyword arguments for clients other than `requests`
        '''

        data = MultipartEncoder({'vector_space_id': str(self.vector_space_id)})
        self._client.post_form(('/api/v0/space/%s/delete_all' % self.vector_space_id), data, kwargs)

    ##################
    # Toolbelt Utils #
    ##################

    try:
        from tqdm import tqdm
        progress_bar = True
    except ImportError:
        progress_bar = False

    @classmethod
    def _custom_progress_bar(self, iterable, desc=None, total=None, progress_bar=False):
        if progress_bar:
            return tqdm(iterable, desc=desc, total=total)
        else:
            return iterable

    @classmethod
    def _batch(self, input_list: list, batch_size:int):
        
        batch_count = math.ceil(len(input_list) / batch_size)
        for i in range(batch_count):
            yield input_list[i * batch_size : (i+1) * batch_size]


    def ingest_image(self, batch_path_list:Union[str, list], attribute_list:Union[str, list], **kwargs) -> IngestResponse:
        '''A function that accepts a str or list of image paths and their attribute, formats it 
        in a list of dicts to be accepted by the ingest function. 

        Args:
            batch_path_list (str or list): Str or list of image paths.
            attribute_list (str or list): Str or list of image attribute.
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            IngestResponse: named tuple that contains the list of index of ingested objects.
        '''

        if type(batch_path_list) != list:
            batch_path_list = [batch_path_list]

        if type(attribute_list) != list:
            attribute_list = [attribute_list]

        vecto_data = []
            
        for path, attribute in zip(batch_path_list, attribute_list):

            data = {'data': open(path, 'rb'), 
                    'attributes': attribute}

            vecto_data.append(data)

        response = self.ingest(vecto_data, "IMAGE")
        
        for data in vecto_data:
            data['data'].close()

        return response

    def ingest_all_images(self, path_list:list, attribute_list:list, batch_size:int=64) -> List[IngestResponse]:
        '''A function that accepts a list of image paths and their attribute, then send them
        to the ingest_image function in batches.

        Args:
            path_list (list): List of image paths.
            attribute_list (list): List of image attribute.
            batch_size (int): batch size of images to be sent at one request. Default 64.
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            IngestResponse: named tuple that contains the list of index of ingested objects.
        '''
        batch_count = math.ceil(len(path_list) / batch_size)

        path_batches = self._batch(path_list, batch_size)
        attribute_batches = self._batch(attribute_list, batch_size)

        ingest_ids = []

        for path_batch, attribute_batch in self._custom_progress_bar(zip(path_batches, attribute_batches), total=batch_count, progress_bar=progress_bar):
            try:
                ids = self.ingest_image(path_batch, attribute_batch)
                ingest_ids.append(ids)
            except:
                print("Error in ingesting:\n", path_batch)

        return ingest_ids

    def ingest_text(self, batch_text_list:Union[str, list], attribute_list:Union[str, list], **kwargs) -> IngestResponse:
        '''A function that accepts a str or list of text and their attribute, formats it 
        in a list of dicts to be accepted by the ingest function. 

        Args:
            batch_text_list (str or list): Str or list of text.
            attribute_list (str or list): Str or list of the text attribute.
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            IngestResponse: named tuple that contains the list of index of ingested objects.
        '''

        vecto_data = []

        if type(batch_text_list) != list:
            batch_text_list = [batch_text_list]

        if type(attribute_list) != list:
            attribute_list = [attribute_list]
        
        for text, attribute in zip(batch_text_list, attribute_list):

            data = {'data': io.StringIO(str(text)),
                    'attributes': attribute}

            vecto_data.append(data)

        response = self.ingest(vecto_data, "TEXT")

        return response

    def ingest_all_text(self, text_list:list, attribute_list:list, batch_size=64) -> List[IngestResponse]:
        '''A function that accepts a list of text and their attribute, then send them
        to the ingest_text function in batches.

        Args:
            text_list (list): List of text paths.
            attribute_list (list): List of text attribute.
            batch_size (int): batch size of text to be sent at one request. Default 64.
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            IngestResponse: named tuple that contains the list of index of ingested objects.
        '''
        
        batch_count = math.ceil(len(text_list) / batch_size)

        text_batches = self._batch(text_list, batch_size)
        attribute_batches = self._batch(attribute_list, batch_size)
        ingest_ids = []

        for path_batch, attribute_batch in (tqdm(zip(text_batches, attribute_batches), total=batch_count) if progress_bar else zip(text_batches, attribute_batches)):
            try:
                ids = self.ingest_text(path_batch, attribute_batch)
                ingest_ids.append(ids)
            except:
                print("Error in ingesting:\n", path_batch)

        return ingest_ids

    ##################
    # Management API #
    ##################

    def _get_model_type(self, input_value):
        if isinstance(input_value, int):
            if input_value in MODEL_MAP:
                return input_value
            else:
                raise ModelNotFoundException(f"Model not found for integer value: {input_value}")
        elif isinstance(input_value, str):
            input_value = input_value.upper()
            for model_int, model_str in MODEL_MAP.items():
                if model_str == input_value:
                    return model_int
            else:
                raise ModelNotFoundException(f"Model not found for string value: {input_value}")
        else:
            raise TypeError(f"Invalid input type: {type(input_value)}.")


    def list_models(self, **kwargs) -> List[VectoModel]:
        '''List all available models in the user's account.

        Returns:
            List[VectoModel]: A list of available VectoModel instances.
        '''

        url = "/api/v0/account/model"
        response = self._client.get(url, **kwargs)

        if not response.json():
            return []

        return [VectoModel(**r) for r in response.json()]
    
    def list_vector_spaces(self, **kwargs) -> List[VectoVectorSpace]:
        '''List all available vector spaces in the user's account.

        Returns:
            List[VectoVectorSpace]: A list of available VectoVectorSpace instances.
        '''
        url = "/api/v0/account/space"
        response = self._client.get(url, **kwargs)

        if not response.json():
            return []

        return [
            VectoVectorSpace(id=r['id'], model=VectoModel(**r['model']), name=r['name'])
            for r in response.json()
        ]
    
    def create_vector_space(self, name:str, model: Union[int, str], **kwargs) -> VectoVectorSpace:
        '''Create a new vector space in the user's account.

        Args:
            name (str): The name of the new vector space.
            model (int or str): The model identifier or name.
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            VectoVectorSpace: The newly created VectoVectorSpace instance.
        '''

        url = "/api/v0/account/space"
        id = self._get_model_type(model)
        json={'name': name, 'modelId': id}

        response = self._client.post_json(url, json, **kwargs)
        response_json = response.json()

        return VectoVectorSpace(id=response_json["id"], model=VectoModel(**response_json["model"]), name=response_json["name"])

    def get_vector_space(self, id:int, **kwargs) -> VectoVectorSpace:
        '''Retrieve a vector space by its ID.

        Args:
            id (int): The ID of the vector space.
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            VectoVectorSpace: The VectoVectorSpace instance with the specified ID.
        '''

        url = f"/api/v0/account/space/{id}"
        response = self._client.get(url, **kwargs)    
        response_json = response.json()

        return VectoVectorSpace(id=response_json["id"], model=VectoModel(**response_json["model"]), name=response_json["name"])
    

    def get_vector_space_by_name(self, name:str, **kwargs) -> List[VectoVectorSpace]:

        '''Retrieve a list of vector spaces by their name.

        Args:
            name (str): The name of the vector spaces.
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            List[VectoVectorSpace]: A list of matching VectoVectorSpace instances.
        '''

        vector_spaces = self.list_vector_spaces()
        matching_spaces = [vs for vs in vector_spaces if vs.name == name]

        return matching_spaces

    def rename_vector_space(self, id:str, new_name:str, **kwargs) -> VectoVectorSpace:
        '''Rename an existing vector space by its ID.

        Args:
            id (str): The ID of the vector space.
            new_name (str): The new name for the vector space.
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            VectoVectorSpace: The renamed VectoVectorSpace instance.
        '''

        url = f"/api/v0/account/space/{id}"
        json = {'name' : new_name}
        response = self._client.put_json(url, json=json, **kwargs)
        return VectoVectorSpace(**response.json())

    def delete_vector_space(self, id, **kwargs) -> object:
        '''Delete a vector space by its ID.

        Args:
            id: The ID of the vector space to be deleted.
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            object: None if the operation is successful.
        '''

        url = f"/api/v0/account/space/{id}"
        self._client.delete(url, **kwargs)

    def delete_all_vector_spaces(self, **kwargs) -> object:
        '''Delete all vector spaces in the user's account.

        **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            object: None if the operation is successful.
        '''
        vector_spaces = self.list_vector_spaces()
        
        for vs in vector_spaces:
            try:
                self.delete_vector_space(vs.id)
            except:
                print("fail in deleting vs " + str(vs.name))

    def get_user_information(self, **kwargs) -> VectoUser:
        '''Retrieve the user information associated with the account.

        **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            VectoUser: A VectoUser instance containing the user's information.
        '''

        url = "/api/v0/account/user"
        response = self._client.get(url, **kwargs)
        return VectoUser(**response.json())
    

    def list_tokens(self, **kwargs) -> List[VectoToken]:

        '''List all available tokens in the user's account.

        **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            List[VectoToken]: A list of available VectoToken instances.
        '''

        url = "/api/v0/account/tokens"
        response = self._client.get(url, **kwargs)
        return [VectoToken(**token) for token in response.json()]
    

    def create_token(self, token_name:str, tokenType:str, vectorSpaceIds:List[int], allowsAccessToAllVectorSpaces:bool, **kwargs) -> VectoNewTokenResponse:
        '''Create a new token for the user's account.

        Args:
            token_name (str): The name of the new token.
            tokenType (str): The type of the token, must be one of 'USAGE', 'PUBLIC', or 'ACCOUNT_MANAGEMENT'.
            vectorSpaceIds (List[int]): A list of vector space IDs the token is associated with.
            allowsAccessToAllVectorSpaces (bool): A flag indicating if the token allows access to all vector spaces.
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            VectoNewTokenResponse: A VectoNewTokenResponse instance containing the newly created token information.
        '''

        tokenType = tokenType.upper()
        if isinstance(vectorSpaceIds, int):
            vectorSpaceIds = [vectorSpaceIds]
        if tokenType not in ["USAGE", "PUBLIC", "ACCOUNT_MANAGEMENT"]:
                raise ValueError("Invalid tokenType. Must be one of 'USAGE', 'PUBLIC', or 'ACCOUNT_MANAGEMENT'.")

        url = "/api/v0/account/tokens"
        json={'name': token_name, 'tokenType':tokenType, 'vectorSpaceIds': vectorSpaceIds, 
              'allowsAccessToAllVectorSpaces': allowsAccessToAllVectorSpaces}
        response = self._client.post_json(url, json, **kwargs)
        return VectoNewTokenResponse(**response.json())
    

    def delete_token(self, token_id:int, **kwargs):
        '''Delete a token by its ID.

        Args:
            token_id (int): The ID of the token to be deleted.
            **kwargs: Other keyword arguments for clients other than `requests`
        '''

        url = f"/api/v0/account/tokens/{token_id}"
        self._client.delete(url, **kwargs)


    def list_analogies(self, vector_space_id, **kwargs) -> object:
        url = f"/api/v0/account/space/{vector_space_id}/analogy"
        response = self._client.get(url, **kwargs)
        return response.json()
    
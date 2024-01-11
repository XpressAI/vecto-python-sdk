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
from datetime import date

from typing import IO, List, Union, Any, Dict
from .exceptions import (UnpairedAnalogy, InvalidModality, ModelNotFoundException )

from .schema import (VectoIngestData, VectoEmbeddingData, VectoAttribute, VectoAnalogyStartEnd,
                    IngestResponse, LookupResult, VectoModel, VectoVectorSpace, VectoUser,
                    VectoToken, VectoNewTokenResponse, MODEL_MAP, VectoAnalogy, 
                    DailyUsageMetric, UsageMetric, VectoUsageMetrics, MonthlyUsageResponse, 
                    DataEntry, DataPage)

from .client import Client
import vecto

class Vecto():
    '''
    Initializes a new Vecto object with the provided configuration.

    `user_vecto = Vecto(token, vector_space_id)`

    Args:
        token (str): The API token used for authentication with the Vecto API.
                        Defaults to the value of the VECTO_API_KEY environment variable.
        vector_space_id (Union[int, str]): The ID of the vector space to interact with.
        vecto_base_url (str): The base URL of the Vecto API. Defaults to "https://api.vecto.ai".
        client: The HTTP client used to send requests to the Vecto API. Defaults to the "requests" library.
    '''

    def __init__(self, token:str=None, 
                 vector_space_id:Union[int, str]=None, 
                 vecto_base_url:str="https://api.vecto.ai", 
                 client=requests):
    
        api_key = token
        if api_key is None:
            api_key = vecto.api_key
                
        self.vector_space_id = vector_space_id
        self._client = Client(api_key, vecto_base_url, client)


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

    def lookup(self, query:IO, modality:str, top_k:int, ids:list=None, **kwargs) -> List[LookupResult]:
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

    def _url_to_binary_stream(self, url: str) -> io.BytesIO:
 
        from urllib.request import urlopen
        from urllib.parse import urlparse

        def _is_url(s: str) -> bool:
            try:
                result = urlparse(s)
                return all([result.scheme, result.netloc])
            except ValueError:
                return False

        if _is_url(url):
            content = urlopen(url).read()
            binary_stream = io.BytesIO(content)
            return binary_stream
        else:
            raise ValueError(f'Invalid URL: {url}')
    

    def lookup_image_from_url(self, query:str, top_k:int, ids:list=None, **kwargs) -> List[LookupResult]:
        '''A function to perform image search on Vecto by passing it an url.

        Args:
            query (str): url str to an image resource
            top_k (int): The number of results to return
            ids (list): A list of vector ids to search on aka subset of vectors, defaults to None
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            list of LookupResult named tuples, where LookResult is named tuple with `data`, `id`, and `similarity` keys.
        '''

        content = self._url_to_binary_stream(query)
        response = self.lookup(content, modality='IMAGE', top_k=top_k, ids=ids)

        return response
    

    def lookup_image_from_filepath(self, query:Union[str, pathlib.Path, os.PathLike], top_k:int, ids:list=None, **kwargs) -> List[LookupResult]:
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

    def lookup_image_from_binary(self, query:io.IOBase, top_k:int, ids:list=None, **kwargs) -> List[LookupResult]:
        '''A function to perform image search on Vecto.

        Args:
            query (io.IOBase): query already in IO form.
            top_k (int): The number of results to return
            ids (list): A list of vector ids to search on aka subset of vectors, defaults to None
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            list of LookupResult named tuples, where LookResult is named tuple with `data`, `id`, and `similarity` keys.
        '''

        response = self.lookup(query, modality='IMAGE', top_k=top_k, ids=ids)

        return response

    def lookup_text_from_str(self, query:str, top_k:int, ids:list=None, **kwargs) -> List[LookupResult]:
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


    def lookup_text_from_filepath(self, query:Union[str, pathlib.Path, os.PathLike], top_k:int, ids:list=None, **kwargs) -> List[LookupResult]:
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


    def lookup_text_from_url(self, query:str, top_k:int, ids:list=None, **kwargs) -> List[LookupResult]:
        '''A function to perform text search on Vecto by passing it an url.

        Args:
            query (str): url str to an text resource
            top_k (int): The number of results to return
            ids (list): A list of vector ids to search on aka subset of vectors, defaults to None
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            list of LookupResult named tuples.           , where LookResult is named tuple with `data`, `id`, and `similarity` keys.
        '''

        content = self._url_to_binary_stream(query)
        response = self.lookup(content, modality='TEXT', top_k=top_k, ids=ids)

        return response
    

    def lookup_text_from_binary(self, query:IO, top_k:int, ids:list=None, **kwargs) -> List[LookupResult]:
        '''A function to perform text search on Vecto by passing it an IO object.

        Args:
            query (IO): query already in IO form
            top_k (int): The number of results to return
            ids (list): A list of vector ids to search on aka subset of vectors, defaults to None
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            list of LookupResult named tuples, where LookResult is named tuple with `data`, `id`, and `similarity` keys.
        '''

        response = self.lookup(query, modality='TEXT', top_k=top_k, ids=ids)

        return response

    ##########
    # Update #
    ##########

    def update_vector_embeddings(self, embedding_data: Union[VectoEmbeddingData, List[VectoEmbeddingData]], modality:str, **kwargs):
        '''A function to update current vector embeddings with new one.

        Args:
            embedding_data (VectoEmbeddingData or list of VectoEmbeddingData): data that contains the embedding data to be updated. 
            modality (str): The type of the file - "IMAGE" or "TEXT"
            **kwargs: Other keyword arguments for clients other than `requests`
        '''

        if type(embedding_data) != list:
            embedding_data = [embedding_data]

        if modality != 'IMAGE' and modality != 'TEXT':
            raise InvalidModality()

        vector_id = [(r['id']) for r in embedding_data]
        files = [('input', ('_', r['data'], '_')) for r in embedding_data]

        data={'vector_space_id': self.vector_space_id, 'id': vector_id, 'modality': modality}
        self._client.post(('/api/v0/space/%s/update/vectors' % self.vector_space_id), data=data, files=files, **kwargs) 


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


    def compute_analogy(self, query:IO, analogy_start_end:Union[VectoAnalogyStartEnd, List[VectoAnalogyStartEnd]], top_k:int, modality:str, **kwargs) -> List[LookupResult]:
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


    def compute_text_analogy(self, query: IO, analogy_start_end: Union[VectoAnalogyStartEnd, List[VectoAnalogyStartEnd]], top_k: int, **kwargs) -> List[LookupResult]:
        '''
        A function to compute a Text analogy using Vecto.
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


    def compute_image_analogy(self, query: IO, analogy_start_end: Union[VectoAnalogyStartEnd, List[VectoAnalogyStartEnd]], top_k: int, **kwargs) -> List[LookupResult]:
        '''
        A function to compute an IMAGE analogy using Vecto.
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

    def delete_vector_embeddings(self, vector_ids:list, **kwargs):
        '''A function to delete vector embeddings that is stored in Vecto.

        Args:
            vector_ids (list): A list of vector ids to be deleted
            **kwargs: Other keyword arguments for clients other than `requests`
        '''

        data = MultipartEncoder(fields=[('vector_space_id', str(self.vector_space_id))] + [('id', str(id)) for id in vector_ids])
        self._client.post_form(('/api/v0/space/%s/delete' % self.vector_space_id), data, kwargs)
        

    def delete_vector_space_entries(self, **kwargs):
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

    @property
    def progress_bar(self):
        try:
            from tqdm import tqdm
            return True
        except ImportError:
            return False

    def _custom_progress_bar(self, iterable, desc=None, total=None):
        if self.progress_bar:
            from tqdm import tqdm
            return tqdm(iterable, desc=desc, total=total)
        else:
            return iterable
        

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

    def ingest_all_images(self, path_list:list, attribute_list:list, batch_size:int=64) -> IngestResponse:
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

        ingested_ids = [] 
        for path_batch, attribute_batch in self._custom_progress_bar(zip(path_batches, attribute_batches), total=batch_count):
            try:
                response = self.ingest_image(path_batch, attribute_batch)
                ingested_ids.extend(response.ids) 
            except:
                print("Error in ingesting:\n", path_batch)

        return IngestResponse(ingested_ids)

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

    def ingest_all_text(self, text_list:list, attribute_list:list, batch_size=64) -> IngestResponse:
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

        ingested_ids = [] 
        for path_batch, attribute_batch in self._custom_progress_bar(zip(text_batches, attribute_batches), total=batch_count):
            try:
                response = self.ingest_text(path_batch, attribute_batch)
                ingested_ids.extend(response.ids)
            except:
                print("Error in ingesting:\n", path_batch)

        return IngestResponse(ingested_ids)

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

    def delete_vector_space(self, id, **kwargs):
        '''Delete a vector space by its ID.

        Args:
            id: The ID of the vector space to be deleted.
            **kwargs: Other keyword arguments for clients other than `requests`
        '''

        url = f"/api/v0/account/space/{id}"
        self._client.delete(url, **kwargs)

    def delete_all_vector_spaces(self, **kwargs):
        '''Delete all vector spaces in the user's account.

        **kwargs: Other keyword arguments for clients other than `requests`
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


    def list_analogies(self, vector_space_id:int, **kwargs) -> List[VectoAnalogy]:
        '''List all available analogies in the specified vector space.

        Args:
            vector_space_id (int): The ID of the vector space.
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            List[VectoAnalogy]: A list of available VectoAnalogy instances in the specified vector space.
        '''

        url = f"/api/v0/account/space/{vector_space_id}/analogy"
        response = self._client.get(url, **kwargs)
        return [VectoAnalogy(**analogy) for analogy in response.json()]

    # TODO: Update create analogy when API is completed
    # def create_analogy(self, vector_space_id:int, **kwargs) -> object:
        
    #     url = f"/api/v0/account/space/{vector_space_id}/analogy"
    #     response = self._client.post(url, data=None, files=None, **kwargs)
    #     return response.json()
    
    def get_analogy(self, vector_space_id:int, analogy_id:int, **kwargs) -> VectoAnalogy:
        '''Retrieve an analogy by its ID from the specified vector space.

        Args:
            vector_space_id (int): The ID of the vector space.
            analogy_id (int): The ID of the analogy.
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            VectoAnalogy: The VectoAnalogy instance containing the specified analogy information.
        '''

        url = f"/api/v0/account/space/{vector_space_id}/analogy/{analogy_id}"
        response = self._client.get(url, **kwargs)
        return VectoAnalogy(**response.json())
    
    def delete_analogy(self, vector_space_id:int, **kwargs):
        '''Delete an analogy from the specified vector space.

        Args:
            vector_space_id (int): The ID of the vector space.
            **kwargs: Other keyword arguments for clients other than `requests`
        '''
        url = f"/api/v0/account/space/{vector_space_id}/analogy"
        self._client.delete(url, **kwargs)

    def list_vector_space_data(self, vector_space_id: int, limit: int = None, offset: int = None, **kwargs):
        '''
        List the attributes of all entries in the given vector space.

        Args:
            vector_space_id (int): The ID of the vector space.
            limit (int, optional): The maximum number of entries to return. If not specified, it will be set to 0.
            offset (int, optional): The offset from the start of the list to begin returning entries.
            **kwargs: Other keyword arguments for clients other than `requests`

        Returns:
            DataPage: A DataPage instance containing the list of entries and their attributes.
        '''

        url = f"/api/v0/space/{vector_space_id}/data"
        params = {'limit': limit, 'offset': offset}
        response = self._client.get(url, params=params, **kwargs)
        response_json = response.json()

        # Create DataEntry instances for each element in the response
        data_entries = [DataEntry(**entry) for entry in response_json["elements"]]

        # Create and return the DataPage instance
        return DataPage(count=response_json["count"], elements=data_entries)

    def delete_vector_space_entry(self, vector_space_id: int, entry_id: int, **kwargs):
        '''
        Delete an entry in a vector space.

        Args:
            vector_space_id (int): The ID of the vector space.
            entry_id (int): The ID of the entry to be deleted.
            **kwargs: Other keyword arguments for clients other than `requests`
        '''

        url = f"/api/v0/space/{vector_space_id}/data/{entry_id}"
        self._client.delete(url, **kwargs)

    ###############
    # Metrics API #
    ###############

    def _parse_daily_usage_metric(self, d: Dict[str, Any]) -> DailyUsageMetric:
        return DailyUsageMetric(
            date=date.fromisoformat(d['date']),
            count=d['count'],
            cumulativeCount=d['cumulativeCount']
        )

    def _parse_usage_metric(self, u: Dict[str, Any]) -> UsageMetric:
        return UsageMetric(
            dailyMetrics=[self._parse_daily_usage_metric(d) for d in u['dailyMetrics']]
        )

    def _parse_vecto_usage_metrics(self, u: Dict[str, Any]) -> VectoUsageMetrics:
        return VectoUsageMetrics(
            lookups=self._parse_usage_metric(u['lookups']),
            indexing=self._parse_usage_metric(u['indexing'])
        )

    def usage(self, year: int, month: int, vector_space_id: int = None, **kwargs) -> MonthlyUsageResponse:
        '''Return the usage metrics for the selected month

        Args:
            year (int): The year for the usage data.
            month (int): The month for the usage data.
            vector_space_id (int, optional): The ID of the vector space. Falls back to self vector id if not provided.
            **kwargs: Other keyword arguments for clients other than `requests`
        Returns:
            MonthlyUsageResponse: Named tuple that contains the usage metrics for a specified vector space and month.
        '''

        # Use provided vector_space_id or fallback to self.vector_space_id
        vector_space_id = vector_space_id or getattr(self, 'vector_space_id', None)
        
        # Raise an error if vector_space_id is still not available
        if vector_space_id is None:
            raise ValueError("A vector space ID must be provided either as a parameter or set in the instance.")

        url = f"/api/v0/space/{vector_space_id}/usage/{year}/{month}"
        response = self._client.get(url, **kwargs)
        response_data = response.json()
        
        usage_metrics = self._parse_vecto_usage_metrics(response_data['usage'])

        monthly_usage_response = MonthlyUsageResponse(
            year=response_data['year'],
            month=response_data['month'],
            usage=usage_metrics
        )

        return monthly_usage_response
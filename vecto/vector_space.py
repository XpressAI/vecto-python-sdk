import vecto
from vecto import Vecto
import os
import io
from typing import IO, List, Union
from .schema import LookupResult, IngestResponse, VectoAnalogyStartEnd
from .exceptions import InvalidModality
from urllib.parse import urlparse
import pathlib

class VectorSpace():
    '''
    Initialize the VectorSpace class.

    Args:
        name (str): The name of the vector space. If multiple vector spaces have the same name, will return the first instance.
        token (str, optional): The API token. If not set, it will check if `VECTO_API_KEY` exists in the env.
        modality (str, optional): The modality of the vector space (TEXT or IMAGE).
    '''
    def __init__(self, name: str, token: str = None, modality: str = None, *args, **kwargs):
        api_key = token

        if api_key is None:
            api_key = os.getenv("VECTO_API_KEY")

        self.vecto_instance = Vecto(token=api_key, *args, **kwargs)
        self.name = name
        self.vector_space_id = None
        self.model = None
        self.modality = modality

        vector_spaces = self.vecto_instance.get_vector_space_by_name(self.name)
        if len(vector_spaces) > 1:
            print("Warning: Multiple vector spaces with the same name found. Using the first one.")

        if vector_spaces:
            self.vector_space_id = vector_spaces[0].id
            self.model = vector_spaces[0].model
            if self.modality is None:
                self.modality = self.model.modality
            self.vecto_instance = Vecto(token=token, vector_space_id=self.vector_space_id, *args, **kwargs)
            print("Vector space " + name + " loaded.")

    def exists(self) -> bool:
        '''
        Check if the vector space exists.

        Returns:
            bool: True if the vector space exists, False otherwise
        '''
        return self.vector_space_id is not None

    def create(self, model: str, token: str = None, modality: str = None):
        '''
        Create a new vector space.

        Args:
            model (str): The name of the model to be used
            modality (str, optional): The modality of the vector space (TEXT or IMAGE), defaults to None
        '''
        if not self.exists():
            created_vector_space = self.vecto_instance.create_vector_space(self.name, model=model)
            
            if token is None:
                token = os.getenv("VECTO_API_KEY", "-1")

            self.vecto_instance = Vecto(token, vector_space_id=created_vector_space.id)
            self.vector_space_id = created_vector_space.id
            self.model = created_vector_space.model

            if modality is not None:
                self.modality = modality
            else:
                self.modality = self.model.modality

            print(f"Created VectorSpace: {created_vector_space.name}")
        else:
            print("VectorSpace already exists.")

    def lookup(self, query: IO, top_k: int, ids: list = None, **kwargs) -> List[LookupResult]:
        '''
        Perform a lookup query on the vector space.

        Args:
            query (IO): The query as an IO object
            top_k (int): The number of results to return
            ids (list, optional): A list of vector ids to search on (subset of vectors), defaults to None

        Returns:
            list of LookupResult: A list of LookupResult named tuples containing 'data', 'id', and 'similarity' keys
        '''
        if self.modality not in ["TEXT", "IMAGE"]:
            raise InvalidModality(f"The current modality '{self.modality}' is not supported. Please update the modality to either 'TEXT' or 'IMAGE'.")

        return self.vecto_instance.lookup(query=query, modality=self.modality, top_k=top_k, ids=ids, **kwargs)
    
    def lookup_image(self, query, top_k: int, ids: list = None, **kwargs) -> List[LookupResult]:
        '''
        Perform an image lookup query on the vector space.

        Args:
            query: The image query (URL, filepath, or IO object)
            top_k (int): The number of results to return
            ids (list, optional): A list of vector ids to search on (subset of vectors), defaults to None

        Returns:
            list of LookupResult: A list of LookupResult named tuples containing 'data', 'id', and 'similarity' keys
        '''
        if isinstance(query, (str, os.PathLike)):
            query_str = str(query)
            parsed_url = urlparse(query_str)
            if bool(parsed_url.scheme) and bool(parsed_url.netloc):
                return self.vecto_instance.lookup_image_from_url(query, top_k=top_k, ids=ids, **kwargs)
            else:
                return self.vecto_instance.lookup_image_from_filepath(query, top_k=top_k, ids=ids, **kwargs)
        if isinstance(query, io.IOBase):
            return self.vecto_instance.lookup_image_from_binary(query, top_k=top_k, ids=ids, **kwargs)
        else:
            raise ValueError("Invalid query type. Must be a string (URL or filepath), os.PathLike, or IO object.")

    def lookup_text(self, query, top_k: int, ids: list = None, **kwargs) -> List[LookupResult]:
        '''
        Perform a text lookup query on the vector space.

        Args:
            query: The text query (string, path-like object, or IO object)
            top_k (int): The number of results to return
            ids (list, optional): A list of vector ids to search on (subset of vectors), defaults to None

        Returns:
            list of LookupResult: A list of LookupResult named tuples containing 'data', 'id', and 'similarity' keys
        '''
        if isinstance(query, str):
            if query.startswith('http://') or query.startswith('https://'):
                return self.vecto_instance.lookup_text_from_url(query, top_k=top_k, ids=ids, **kwargs)
            else:
                return self.vecto_instance.lookup_text_from_str(query, top_k=top_k, ids=ids, **kwargs)
        elif isinstance(query, (pathlib.Path, os.PathLike)):
            return self.vecto_instance.lookup_text_from_filepath(query, top_k=top_k, ids=ids, **kwargs)
        elif isinstance(query, IO):
            return self.vecto_instance.lookup_text_from_binary(query, top_k=top_k, ids=ids, **kwargs)
        else:
            raise ValueError("Invalid query type. Please provide a string, path-like object, or IO object.")
        

    def ingest_image(self, image_path: str, attribute: str, **kwargs) -> IngestResponse:
        '''
        Ingest an image into the vector space.

        Args:
            image_path (str): The path of the image to ingest
            attribute (str): The attribute associated with the image

        Returns:
            IngestResponse: An IngestResponse object containing the response data
        '''
        return self.vecto_instance.ingest_image(image_path, attribute, **kwargs)
    
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
            
            return self.vecto_instance.ingest_all_images(path_list, attribute_list, batch_size)


    def ingest_text(self, text: str, attribute: str, **kwargs) -> IngestResponse:
        '''
        Ingest text into the vector space.

        Args:
            text (str): The text to ingest
            attribute (str): The attribute associated with the text

        Returns:
            IngestResponse: An IngestResponse object containing the response data
        '''
        return self.vecto_instance.ingest_text(text, attribute, **kwargs)

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
        
        return self.vecto_instance.ingest_all_text(text_list, attribute_list, batch_size)


    def compute_text_analogy(self, query: IO, analogy_start_end: Union[VectoAnalogyStartEnd, List[VectoAnalogyStartEnd]], top_k: int, **kwargs) -> List[LookupResult]:
        '''
        Compute text analogy on the vector space.

        Args:
            query (IO): The query as an IO object
            analogy_start_end (Union[VectoAnalogyStartEnd, List[VectoAnalogyStartEnd]]): The start and end points of the analogy
            top_k (int): The number of results to return
            **kwargs: Other keyword arguments

        Returns:
            list of LookupResult: A list of LookupResult named tuples containing 'data', 'id', and 'similarity' keys
        '''
        return self.vecto_instance.compute_text_analogy(query, analogy_start_end, top_k, **kwargs)

    def compute_image_analogy(self, query: IO, analogy_start_end: Union[VectoAnalogyStartEnd, List[VectoAnalogyStartEnd]], top_k: int, **kwargs) -> List[LookupResult]:
        '''
        Compute image analogy on the vector space.

        Args:
            query (IO): The query as an IO object
            analogy_start_end (Union[VectoAnalogyStartEnd, List[VectoAnalogyStartEnd]]): The start and end points of the analogy
            top_k (int): The number of results to return

        Returns:
            list of LookupResult: A list of LookupResult named tuples containing 'data', 'id', and 'similarity' keys
        '''

        return self.vecto_instance.compute_image_analogy(query, analogy_start_end, top_k, **kwargs)

    def clear_entries(self, **kwargs):
        '''
        Clear all entries in the vector space.
        '''
        self.vecto_instance.delete_vector_space_entries(**kwargs)

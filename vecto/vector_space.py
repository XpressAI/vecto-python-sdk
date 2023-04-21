from vecto import Vecto
import os
from typing import IO, List, Union
from .schema import LookupResult, IngestResponse, VectoAnalogyStartEnd
from .exceptions import InvalidModality
from urllib.parse import urlparse

class VectorSpace:
    def __init__(self, name: str, token: str = None, modality: str = None, *args, **kwargs):
        if token is None:
            token = os.getenv("VECTO_API_KEY", "-1")

        self.vecto_instance = Vecto(token=token, *args, **kwargs)
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

    def exists(self) -> bool:
        return self.vector_space_id is not None

    def create(self, model: str, modality: str = None):
        if not self.exists():
            created_vector_space = self.vecto_instance.create_vector_space(self.name, model=model)
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
        if self.modality not in ["TEXT", "IMAGE"]:
            raise InvalidModality(f"The current modality '{self.modality}' is not supported. Please update the modality to either 'TEXT' or 'IMAGE'.")

        return self.vecto_instance.lookup(query=query, modality=self.modality, top_k=top_k, ids=ids, **kwargs)
    
    def lookup_image(self, query, top_k: int, ids: list = None, **kwargs) -> List[LookupResult]:
        if isinstance(query, (str, os.PathLike)):
            query_str = str(query)
            parsed_url = urlparse(query_str)
            if bool(parsed_url.scheme) and bool(parsed_url.netloc):
                return self.vecto_instance.lookup_image_from_url(query, top_k=top_k, ids=ids, **kwargs)
            else:
                return self.vecto_instance.lookup_image_from_filepath(query, top_k=top_k, ids=ids, **kwargs)
        elif isinstance(query, IO):
            return self.vecto_instance.lookup_image_from_binary(query, top_k=top_k, ids=ids, **kwargs)
        else:
            raise ValueError("Invalid query type. Must be a string (URL or filepath), os.PathLike, or IO object.")

    def lookup_text(self, query, top_k: int, ids: list = None, **kwargs) -> List[LookupResult]:
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
        return self.vecto_instance.ingest_image(image_path, attribute, **kwargs)

    def ingest_text(self, text: str, attribute: str, **kwargs) -> IngestResponse:
        return self.vecto_instance.ingest_text(text, attribute, **kwargs)

    def compute_text_analogy(self, query: IO, analogy_start_end: Union[VectoAnalogyStartEnd, List[VectoAnalogyStartEnd]], top_k: int, **kwargs) -> List[LookupResult]:
        return self.vecto_instance.compute_text_analogy(query, analogy_start_end, top_k, **kwargs)

    def compute_image_analogy(self, query: IO, analogy_start_end: Union[VectoAnalogyStartEnd, List[VectoAnalogyStartEnd]], top_k: int, **kwargs) -> List[LookupResult]:
        return self.vecto_instance.compute_image_analogy(query, analogy_start_end, top_k, **kwargs)

    def clear_entries(self, **kwargs):
        return self.vecto_instance.delete_vector_space_entries(**kwargs)
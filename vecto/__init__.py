from vecto.vecto_requests import Vecto
from vecto.vector_space import VectorSpace
import os

api_key = api_key = os.environ.get("VECTO_API_KEY")

__all__ = [
    "Vecto", 
    "VectorSpace", 
    "api_key"
]

from vecto import Vecto
import os

class VectorSpace(Vecto):
    def __init__(self, name: str, token: str = None, *args, **kwargs):
        if token is None:
            token = os.getenv("VECTO_API_KEY", "-1")

        super().__init__(token=token, *args, **kwargs)
        self.name = name
        self.id = None
        self.model = None

        vector_spaces = self.get_vector_space_by_name(self.name)
        if len(vector_spaces) > 1:
            print("Warning: Multiple vector spaces with the same name found. Using the first one.")

        if vector_spaces:
            self.id = vector_spaces[0].id
            self.model = vector_spaces[0].model
            self.vector_space_id = self.id

    def exists(self) -> bool:
        return self.id is not None

    def create(self, model: str, modality: str):
        if not self.exists():
            created_vector_space = self.create_vector_space(self.name, model=model)
            self.id = created_vector_space.id
            self.model = created_vector_space.model
            self.vector_space_id = self.id
            print(f"Created VectorSpace: {created_vector_space.name}")
        else:
            print("VectorSpace already exists.")
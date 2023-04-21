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
        if vector_spaces:
            self.id = vector_spaces[0].id
            self.model = vector_spaces[0].model

    def exists(self) -> bool:
        return self.id is not None

    def create(self, model: str):
        if not self.exists():
            created_vector_space = self.create_vector_space(self.name, model=model)
            self.id = created_vector_space.id
            self.model = created_vector_space.model
            print(f"Created VectorSpace: {created_vector_space.name}")
        else:
            print("VectorSpace already exists.")
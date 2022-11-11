class VectoException(Exception):

    status_code: int

    def __init__(self, response):
        self.status_code = response.status_code
        self.message = "Error! Received error code " + str(self.status_code)

        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'


class UnknownVectorSpace(VectoException):
    def __init__(self, code, message="Unknown Vector Space"):
        self.code = code
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'

class UnpairedAnalogy(VectoException):
    def __init__(self, code, message="Unpaired analogy error received."):
        self.code = code
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'

class LookupException(VectoException):
    pass
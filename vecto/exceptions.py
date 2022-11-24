class VectoException(Exception):
    """The base exception class for all Vecto exceptions."""



class UnauthorizedException(VectoException):
    def __init__(self, message="Authentication error. Ensure that you have the correct vector space and token."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'

class ForbiddenException(VectoException):
    def __init__(self, message="User is unauthorized, please check your access token or user/password."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'

class NotFoundException(VectoException):
    def __init__(self, message="The resource you've requested is not found."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'
class ServiceException(VectoException):
    def __init__(self, message="The request you've submitted did not return any valid response."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'

class UnpairedAnalogy(VectoException):
    def __init__(self, message="Unpaired analogy error received."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'

class InvalidModality(VectoException):
    def __init__(self, message="Ensure that you have used either IMAGE or TEXT as the modality."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'

class LookupException(VectoException):
    pass
class VectoException(Exception):

    status_code: int

    def __init__(self, response):
        self.status_code = response.status_code

        if self.status_code == 401:
            raise UnauthorizedException(self.status_code)

        if self.status_code == 403:
            raise ForbiddenException(self.status_code)

        if self.status_code == 404:
            raise NotFoundException(self.status_code)

        if 500 <= self.status_code <= 599:
            raise ServiceException(self.status_code)

        self.message = "Error! Received error code " + str(self.status_code)

        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'

class UnauthorizedException(VectoException):
    def __init__(self, code, message="Authentication error. Ensure that you have the correct vector space and token."):
        self.code = code
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'

class ForbiddenException(VectoException):
    def __init__(self, code, message="You do not have access to this resource."):
        self.code = code
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'

class NotFoundException(VectoException):
    def __init__(self, code, message="The resource you've requested is not found."):
        self.code = code
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'
class ServiceException(VectoException):
    def __init__(self, code, message="The request you've submitted did not return any valid response."):
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
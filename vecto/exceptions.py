class VectoException(Exception):

    def __init__(self, response):
    
        self.message = "Error! Received error code " + str(response.status_code)
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'


class UnpairedAnalogy(Exception):
    def __init__(self, code, message="Unpaired analogy error received."):
        self.code = code
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'
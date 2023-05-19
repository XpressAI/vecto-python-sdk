from .exceptions import ( VectoException, UnauthorizedException, 
                        ForbiddenException, NotFoundException, ServiceException, 
                        ConsumedResourceException )

class Client:
    def __init__(self, token:str, vecto_base_url: str, client) -> None:
        if not token:
            raise VectoException("Token not detected, please provide a valid token.")
        self.token = token
        self.vecto_base_url = vecto_base_url
        self.client = client
        

    def get(self, url, **kwargs):

        self.validate_input(url=url)
        headers = {"Authorization": "Bearer %s" % self.token}
        response = self.client.get("%s/%s" % (self.vecto_base_url, url),
                                        headers=headers,
                                        **kwargs)
        
        self.check_common_error(response)
        return response


    def put(self, url, json=None, data=None, files=None, **kwargs):

        self.validate_input(url=url, data=data, files=files)
        headers = {"Authorization": "Bearer %s" % self.token}
        response = self.client.put("%s/%s" % (self.vecto_base_url, url),
                                        headers=headers,
                                        **kwargs)
        self.check_common_error(response)
        return response
    
    def put_json(self, url, json, **kwargs):

        self.validate_input(url=url)
        headers = {"Authorization": "Bearer %s" % self.token, 'Content-Type': 'application/json'}
        response = self.client.put("%s/%s" % (self.vecto_base_url, url),
                                        json=json,
                                        headers=headers,
                                        **kwargs)
        self.check_common_error(response)
        return response


    def delete(self, url, data=None, files=None, **kwargs):

        self.validate_input(url=url, data=data, files=files)
        headers = {"Authorization": "Bearer %s" % self.token}
        response = self.client.delete("%s/%s" % (self.vecto_base_url, url),
                                        data=data,
                                        files=files,
                                        headers=headers,
                                        **kwargs)
        self.check_common_error(response)
        return response

    def post(self, url, data, files, **kwargs):

        self.validate_input(url=url, data=data, files=files)
        headers = {"Authorization": "Bearer %s" % self.token}
        response = self.client.post("%s/%s" % (self.vecto_base_url, url),
                                        data=data,
                                        files=files,
                                        headers=headers,
                                        **kwargs)
        self.check_common_error(response)
        return response

    def post_json(self, url, json, **kwargs):

        self.validate_input(url=url)
        headers = {"Authorization": "Bearer %s" % self.token, 'Content-Type': 'application/json'}
        response = self.client.post("%s/%s" % (self.vecto_base_url, url),
                                        json=json,
                                        headers=headers,
                                        **kwargs)
        self.check_common_error(response)
        return response

    def post_form(self, url, data, kwargs=None):

        self.validate_input(url=url, data=data)
        headers = {"Authorization": "Bearer %s" % self.token, 'Content-Type': data.content_type}
        response = self.client.post("%s/%s" % (self.vecto_base_url, url),
                                data=data,
                                headers=headers,
                                **kwargs)

        self.check_common_error(response)
        return response


    def validate_input(self, url=None, data=None, files=None, headers=None):

        def _check_input_buffer(files):
            '''Currently ingest files are formatted as:
            [('input', ('_', <_io.BufferedReader name='file.png'>, '_'))]
            '''

            for file in files:
                for ingest_input in file:
                    for buffer in ingest_input:
                        if str(buffer.__class__.__name__) == "BufferedReader":
                            if buffer.peek() == b'':
                                raise ConsumedResourceException()

        if files != None:
            _check_input_buffer(files)


    def check_common_error(self, response):

        if response.ok:
            return

        status_code = response.status_code

        if status_code == 400:
            if "vector_space_id" not in response.text:
                raise VectoException("Submitted data is incorrect, please check your request.")
            else:
                raise VectoException("Request failed because a vector_space_id was not provided.")
        elif status_code == 401:
            raise UnauthorizedException()
        elif status_code == 403:
            raise ForbiddenException()
        elif status_code == 404:
            raise NotFoundException()
        elif 500 <= status_code <= 599:
            raise ServiceException()
        else:
            raise VectoException("Error status code ["+str(status_code)+"].")


from .vecto_requests import Vecto, IngestResponse
import math
from tqdm import tqdm
import io
from typing import List, Union


def batch(input_list: list, batch_size:int):
    batch_count = math.ceil(len(input_list) / batch_size)
    for i in range(batch_count):
        yield input_list[i * batch_size : (i+1) * batch_size]


def ingest_image(vs:Vecto, batch_path_list:Union[str, list], attribute_list:Union[str, list], **kwargs) -> IngestResponse:
    """A function that accepts a str or list of image paths and their attribute, formats it 
    in a list of dicts to be accepted by the ingest function. 

    Args:
        batch_path_list (str or list): Str or list of image paths.
        attribute_list (str or list): Str or list of image attribute.
        **kwargs: Other keyword arguments for clients other than `requests`

    Returns:
        IngestResponse: named tuple that contains the list of index of ingested objects.
    """

    if type(batch_path_list) != list:
        batch_path_list = [batch_path_list]

    if type(attribute_list) != list:
        attribute_list = [attribute_list]

    vecto_data = []
        
    for path, attribute in zip(batch_path_list, attribute_list):

        data = {'data': open(path, 'rb'), 
                 'attributes': attribute}

        vecto_data.append(data)

    response = vs.ingest(vecto_data, "IMAGE")
    
    for data in vecto_data:
        data['data'].close()

    return response

def ingest_all_images(vs:Vecto, path_list:list, attribute_list:list, batch_size:int=64) -> List[IngestResponse]:
    """A function that accepts a list of image paths and their attribute, then send them
    to the ingest_image function in batches.

    Args:
        path_list (list): List of image paths.
        attribute_list (list): List of image attribute.
        batch_size (int): batch size of images to be sent at one request. Default 64.
        **kwargs: Other keyword arguments for clients other than `requests`

    Returns:
        IngestResponse: named tuple that contains the list of index of ingested objects.
    """
    batch_count = math.ceil(len(path_list) / batch_size)

    path_batches = batch(path_list, batch_size)
    attribute_batches = batch(attribute_list, batch_size)

    ingest_ids = []

    for path_batch, attribute_batch in tqdm(zip(path_batches, attribute_batches), total = batch_count):
        
        try:
            ids = ingest_image(vs, path_batch, attribute_batch)
            ingest_ids.append(ids)

        except:
            print("Error in ingesting:\n", path_batch)

    return ingest_ids

def ingest_text(vs:Vecto, batch_text_list:Union[str, list], attribute_list:Union[str, list], **kwargs) -> IngestResponse:
    """A function that accepts a str or list of text and their attribute, formats it 
    in a list of dicts to be accepted by the ingest function. 

    Args:
        batch_text_list (str or list): Str or list of text.
        attribute_list (str or list): Str or list of the text attribute.
        **kwargs: Other keyword arguments for clients other than `requests`

    Returns:
        IngestResponse: named tuple that contains the list of index of ingested objects.
    """

    vecto_data = []

    if type(batch_text_list) != list:
        batch_text_list = [batch_text_list]

    if type(attribute_list) != list:
        attribute_list = [attribute_list]
    
    for text, attribute in zip(batch_text_list, attribute_list):

        data = {'data': io.StringIO(str(text)),
                 'attributes': attribute}

        vecto_data.append(data)

    response = vs.ingest(vecto_data, "TEXT")

    return response

def ingest_all_text(vs:Vecto, text_list:list, attribute_list:list, batch_size=64) -> List[IngestResponse]:
    """A function that accepts a list of text and their attribute, then send them
    to the ingest_text function in batches.

    Args:
        text_list (list): List of text paths.
        attribute_list (list): List of text attribute.
        batch_size (int): batch size of text to be sent at one request. Default 64.
        **kwargs: Other keyword arguments for clients other than `requests`

    Returns:
        IngestResponse: named tuple that contains the list of index of ingested objects.
    """
    
    batch_count = math.ceil(len(text_list) / batch_size)

    text_batches = batch(text_list, batch_size)
    attribute_batches = batch(attribute_list, batch_size)
    ingest_ids = []

    for text_batch, attribute_batch in tqdm(zip(text_batches,attribute_batches), total = batch_count):
        ids = ingest_text(vs, text_batch, attribute_batch)
        ingest_ids.append(ids)

    return ingest_ids

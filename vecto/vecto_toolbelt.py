from .vecto_requests import Vecto, IngestResponse
import math
from tqdm import tqdm
import io

def ingest_image(vs:Vecto, batch_path_list:str or list, metadata_list:str or list, **kwargs) -> IngestResponse:
    """A function that accepts a str or list of image paths and their metadata, formats it 
    in a list of dicts to be accepted by the ingest function. 

    Args:
        batch_path_list (str or list): Str or list of image paths.
        metadata_list (str or list): Str or list of image metadata.
        **kwargs: Other keyword arguments for clients other than `requests`

    Returns:
        IngestResponse: named tuple that contains the list of index of ingested objects.
            IngestResponse(NamedTuple):
                ids: List[int]
    """

    if type(batch_path_list) != list:
        batch_path_list = [batch_path_list]

    if type(metadata_list) != list:
        metadata_list = [metadata_list]

    vecto_data = []
        
    for path, metadata in zip(batch_path_list, metadata_list):

        data = {'data': open(path, 'rb'), 
                 'attributes': metadata}

        vecto_data.append(data)

    response = vs.ingest(vecto_data, "IMAGE")
    
    for data in vecto_data:
        data['data'].close()

    return response

def ingest_all_images(vs:Vecto, path_list:list, metadata_list:list, batch_size:int=64) -> IngestResponse:
    """A function that accepts a list of image paths and their metadata, then send them
    to the ingest_image function in batches.

    Args:
        batch_path_list (list): List of image paths.
        metadata_list (list): List of image metadata.
        batch_size (int): batch size of images to be sent at one request. Default 64.
        **kwargs: Other keyword arguments for clients other than `requests`

    Returns:
        IngestResponse: named tuple that contains the list of index of ingested objects.
            IngestResponse(NamedTuple):
                ids: List[int]
    """
    batch_count = math.ceil(len(path_list) / batch_size)
    path_batches = [path_list[i * batch_size: (i + 1) * batch_size] for i in range(batch_count)]
    metadata_batches = [metadata_list[i * batch_size: (i + 1) * batch_size] for i in range(batch_count)]

    ingest_ids = []

    for path_batch, metadata_batch in tqdm(zip(path_batches, metadata_batches), total = len(path_batches)):
        
        try:
            ids = ingest_image(vs, path_batch, metadata_batch)
            ingest_ids.append(ids)

        except:
            print("Error in ingesting:\n", path_batch)

    return ingest_ids

def ingest_text(vs:Vecto, batch_text_list:list, metadata_list:list, **kwargs) -> IngestResponse:
    """A function that accepts a str or list of text and their metadata, formats it 
    in a list of dicts to be accepted by the ingest function. 

    Args:
        batch_path_list (str or list): Str or list of text.
        metadata_list (str or list): Str or list of the text metadata.
        **kwargs: Other keyword arguments for clients other than `requests`

    Returns:
        IngestResponse: named tuple that contains the list of index of ingested objects.
            IngestResponse(NamedTuple):
                ids: List[int]
    """

    vecto_data = []
    
    for text, metadata in zip(batch_text_list, metadata_list):

        data = {'data': io.StringIO(str(text)),
                 'attributes': metadata}

        vecto_data.append(data)

    response = vs.ingest(vecto_data, "TEXT")

    return response

def ingest_all_text(text_list:list, metadata_list:list, batch_size=64) -> IngestResponse:
    """A function that accepts a list of text and their metadata, then send them
    to the ingest_text function in batches.

    Args:
        batch_text_list (list): List of image paths.
        metadata_list (list): List of image metadata.
        batch_size (int): batch size of images to be sent at one request. Default 64.
        **kwargs: Other keyword arguments for clients other than `requests`

    Returns:
        IngestResponse: named tuple that contains the list of index of ingested objects.
            IngestResponse(NamedTuple):
                ids: List[int]
    """
    batch_count = math.ceil(len(text_list) / batch_size)
    batches_path = [text_list[i * batch_size: (i + 1) * batch_size] for i in range(batch_count)]
    batches_text = [metadata_list[i * batch_size: (i + 1) * batch_size] for i in range(batch_count)]
    
    ingest_ids = []

    for batch,text in tqdm(zip(batches_path,batches_text), total = len(batches_path)):
        ids = ingest_text(batch,text)
        ingest_ids.append(ids)

    return ingest_ids

import json
import math
from tqdm import tqdm

def ingest_image(vs, batch_path_list:list, **kwargs) -> object:
    """A function to ingest a batch of images into Vecto.
    Also works with single image aka batch of 1.

    Args:
        batch_path_list (list): List of image paths (or list of one image path if batch of 1)
        **kwargs: Other keyword arguments for clients other than `requests`

    Returns:
        tuple: A tuple of two dictionaries (client response body, client request body)
    """
    data = {'vector_space_id': vs._client.vector_space_id, 'data': [], 'modality': 'IMAGE'}
    files = []
    for path in batch_path_list:
        relative = "%s/%s" % (path.parent.name, path.name)
        # relative = ""
        data['data'].append(json.dumps(relative))
        files.append(open(path, 'rb'))

    

    response = vs.ingest(data, files)
    for f in files:
        f.close()
    
    return response

def ingest_all_images(vs, path_list, batch_size=64):
    batch_count = math.ceil(len(path_list) / batch_size)
    batches = [path_list[i * batch_size: (i + 1) * batch_size] for i in range(batch_count)]
    for batch in tqdm(batches):
        ingest_image(vs, batch)


def ingest_text(vs, batch_index_list:list, batch_text_list:list, **kwargs) -> object:
    """A function to ingest a batch of text into Vecto. 
    Also works with single text aka batch of 1.

    Args:
        batch_text_list (list): List of texts (or list of one text if batch of 1)
        **kwargs: Other keyword arguments for clients other than `requests`

    Returns:
        tuple: A tuple of two dictionaries (client response body, client request body)
    """
    data = {'vector_space_id': vs._client.vector_space_id, 'data': [], 'modality': 'TEXT'}
    for index, text in zip(batch_index_list, batch_text_list):
        data['data'].append(json.dumps('text_{}'.format(index) + '_{}'.format(text)))
    # import pdb; pdb.set_trace()

    response = vs.ingest(data, batch_text_list)

    return response

def ingest_all_text(path_list,text_list, batch_size=64):
    batch_count = math.ceil(len(path_list) / batch_size)
    batches_path = [path_list[i * batch_size: (i + 1) * batch_size] for i in range(batch_count)]
    batches_text = [text_list[i * batch_size: (i + 1) * batch_size] for i in range(batch_count)]
    for batch,text in tqdm(zip(batches_path,batches_text), total = len(batches_path)):
        ingest_text(batch,text)


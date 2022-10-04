import io
from vecto import VectoAPI
from test_util import DatabaseTwin, TestDataset
import random
import logging
import pytest
import json

'''
Please update token, vecto_base_url and vector_space_id in *vecto_config.env*
before running `pytest test_user.py` in terminal

You can choose a different seed in test_util.py

If you run into any errors, you can use pdb.set_trace() to set pytest debugger checkpoint.
See https://docs.pytest.org/en/6.2.x/usage.html#setting-breakpoints for more info.
'''

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
token = os.environ['user_token']
vector_space_id = os.environ['vector_space_id']

user_vecto = VectoAPI(token, vector_space_id)
user_db_twin = DatabaseTwin()

# Clear off vector space before start
@pytest.mark.clear
def test_clear_vector_space_entries():
    response = user_vecto.delete_vector_space_entries()
    
    f = io.StringIO('blue')
    lookup_response = user_vecto.lookup_single(f, modality='TEXT', top_k=100)
    results = lookup_response.json()['results']
    
    logger.info(response.status_code)
    assert response.status_code is 200
    assert response.content is not None
    logger.info("Checking if there's 0 lookup results: " + str(len(results) == 0))
    assert len(results) is 0

@pytest.mark.ingest
class TestIngesting:
    
    # Test ingesting one image into Vecto
    def test_ingest_single_image(self):
        image = TestDataset.get_random_image()
        response, metadata = user_vecto.ingest_image_batch(image)
        logger.info(response.status_code)
        assert response.status_code is 200
        assert response.content is not None

        results = response.json()['ids']
        user_db_twin.update_database(results, metadata)
        ref_db = user_db_twin.get_database()


        logger.info('Number of ingested input: ' + str(len(results)))
        assert len(results) == 1 # ingested only 1 input so it should return only list of length 1
        
        logger.info(f'Check if ID of last ingested input is {ref_db["id"].iloc[-1]}: ' + 
                        str(results[-1] == ref_db["id"].iloc[-1]))
        assert results[-1] == ref_db["id"].iloc[-1] # last ingested input of vector space so it should be the last entry in ref_db

    # Test ingesting multiple images into Vecto
    def test_ingest_image_batch(self):
        batch = TestDataset.get_image_dataset()[:5]
        response, metadata = user_vecto.ingest_image_batch(batch)
        results = response.json()['ids']
        user_db_twin.update_database(results, metadata)
        ref_db = user_db_twin.get_database()
        
        logger.info(response.status_code)      
        assert response.status_code is 200
        assert response.content is not None
        
        logger.info('Number of ingested input:' + str(len(results)))
        assert len(results) == 5 # ingested only 5 input so it should return only list of length 5
        
        logger.info(f'Check if ID of last ingested input is {ref_db["id"].iloc[-1]}: ' + 
                        str(results[-1] == ref_db["id"].iloc[-1]))
        assert results[-1] == ref_db["id"].iloc[-1] # last ingested batch input of vector space so it should be the last entry in ref_db

    # Test ingesting multiple images with invalid source attribute into Vecto
    def test_ingest_image_with_invalid_source(self):
        batch = TestDataset.get_image_dataset()[:5]
        data = {'vector_space_id': user_vecto.vector_space_id, 'data': [], 'modality': 'IMAGE'}
        files = []
        for path in batch:
            relative = "%s/%s" % (path.parent.name, path.name)
            data['data'].append(json.dumps({'relative': relative, "_source": relative}))
            files.append(open(path, 'rb'))

        response = user_vecto.ingest(data, files)
        for f in files:
            f.close()
        logger.info(response.status_code)
        assert response.status_code != 200
        assert response.content is not None

    # Test ingesting multiple images with source attribute into Vecto
    def test_ingest_image_with_valid_source(self):
        batch = TestDataset.get_image_dataset()[:5]
        data = {'vector_space_id': user_vecto.vector_space_id, 'data': [], 'modality': 'IMAGE'}
        files = []
        for path in batch:
            relative = "%s/%s" % (path.parent.name, path.name)
            data['data'].append(json.dumps({'relative': relative, "_source": "file:/./%s" % relative}))
            files.append(open(path, 'rb'))

        response = user_vecto.ingest(data, files)
        for f in files:
            f.close()
        results = response.json()['ids']
        user_db_twin.update_database(results, data)
        ref_db = user_db_twin.get_database()

        logger.info(response.status_code)
        assert response.status_code is 200
        assert response.content is not None

        logger.info('Number of ingested input:' + str(len(results)))
        assert len(results) == 5  # ingested only 5 input so it should return only list of length 5

        logger.info(f'Check if ID of last ingested input is {ref_db["id"].iloc[-1]}: ' +
                    str(results[-1] == ref_db["id"].iloc[-1]))
        assert results[-1] == ref_db["id"].iloc[-1]  # last ingested batch input of vector space so it should be the last entry in ref_db

    # Test ingesting one text into Vecto
    def test_ingest_single_text(self):
        text = TestDataset.get_random_text()
        index = [0]
        response, metadata = user_vecto.ingest_text_batch(index, text)
        results = response.json()['ids']
        user_db_twin.update_database(results, metadata)
        ref_db = user_db_twin.get_database()
        
        logger.info(response.status_code)
        assert response.status_code is 200
        assert response.content is not None
        logger.info('Number of ingested input:' + str(len(results)))
        assert len(results) is 1 # ingested only 1 input so it should return only list of length 1
        logger.info(f'Check if ID of last ingested input is {ref_db["id"].iloc[-1]}: ' + 
                        str(results[-1] == ref_db["id"].iloc[-1]))
        assert results[-1] == ref_db["id"].iloc[-1] # last ingested input of vector space so it should be the last entry in ref_db

    # Test ingesting multiple texts into Vecto
    def test_ingest_text_batch(self):
        batch = TestDataset.get_text_dataset()
        response, metadata = user_vecto.ingest_text_batch(batch.index.tolist()[:5], batch.tolist()[:5])
        results = response.json()['ids']
        user_db_twin.update_database(results, metadata)
        ref_db = user_db_twin.get_database()
        
        logger.info(response.status_code)
        assert response.status_code is 200
        assert response.content is not None
        logger.info('Number of ingested input:' + str(len(results)))
        assert len(results) is 5 # ingested only 5 input so it should return only list of length 5
        logger.info(f'Check if ID of last ingested input is {ref_db["id"].iloc[-1]}: ' + 
                        str(results[-1] == ref_db["id"].iloc[-1]))
        assert results[-1] == ref_db["id"].iloc[-1] # last ingested batch input of vector space so it should be the last entry in ref_db
    
    # Check if the number of entries in Vecto aligns with DatabaseTwin
    def test_ingested(self):
        ref_db = user_db_twin.get_database()

        logger.info('Length of ref_df is :' + str(len(ref_db)))
        assert len(ref_db) is 17

@pytest.mark.lookup
class TestLookup:
    
    # Test doing lookup / search using text on Vecto
    def test_lookup_single_text(self):
        f = io.StringIO('blue')
        response_k5 = user_vecto.lookup_single(f, modality='TEXT', top_k=5)
        results_k5 = response_k5.json()['results']

        logger.info(response_k5.status_code)
        assert response_k5.status_code is 200
        assert response_k5.content is not None
        logger.info("Checking if there's 5 lookup results: " + str(len(results_k5) == 5))
        assert len(results_k5) is 5
        logger.info("Checking if values in 'data' is string: " + str(isinstance(results_k5[0]['data'], str)))
        assert isinstance(results_k5[0]['data'], str)
        logger.info("Checking if values in 'id' is not empty: " + str(results_k5[round(len(results_k5) / 2)]['id'] is not None))
        assert results_k5[round(len(results_k5) / 2)]['id'] is not None
        logger.info("Checking if values in 'similarity' is float: " + str(isinstance(results_k5[-1]['similarity'], float)))
        assert isinstance(results_k5[-1]['similarity'], float)

        # top_k=100 is to return everything in the vector space
        response_k100 = user_vecto.lookup_single(f, modality='TEXT', top_k=100)
        results_k100 = response_k100.json()['results']

        logger.info(response_k100.status_code)
        assert response_k100.status_code is 200
        assert response_k100.content is not None
        logger.info("Checking if there's 17 lookup results: " + str(len(results_k100) == 17))
        assert len(results_k100) is 17
        logger.info("Checking if values in 'data' is string: " + str(isinstance(results_k100[0]['data'], str)))
        assert isinstance(results_k100[0]['data'], str)
        logger.info("Checking if values in 'id' is not empty: " + str(results_k100[round(len(results_k100) / 2)]['id'] is not None))
        assert results_k100[round(len(results_k100) / 2)]['id'] is not None
        logger.info("Checking if values in 'similarity' is float: " + str(isinstance(results_k100[-1]['similarity'], float)))
        assert isinstance(results_k100[-1]['similarity'], float)
    
    # Test doing lookup / search using image on Vecto
    def test_lookup_single_image(self):
        query = TestDataset.get_random_image()[0]
        with open(query, 'rb') as f:
            response_k5 = user_vecto.lookup_single(f, modality='IMAGE', top_k=5)
        results_k5 = response_k5.json()['results']

        logger.info(response_k5.status_code)
        assert response_k5.status_code is 200
        assert response_k5.content is not None
        logger.info("Checking if there's 5 lookup results: " + str(len(results_k5) == 5))
        assert len(results_k5) is 5
        logger.info("Checking if values in 'data' is string: " + str(isinstance(results_k5[0]['data'], str)))
        assert isinstance(results_k5[0]['data'], str)
        logger.info("Checking if values in 'id' is not empty: " + str(results_k5[round(len(results_k5) / 2)]['id'] is not None))
        assert results_k5[round(len(results_k5) / 2)]['id'] is not None
        logger.info("Checking if values in 'similarity' is float: " + str(isinstance(results_k5[-1]['similarity'], float)))
        assert isinstance(results_k5[-1]['similarity'], float)

        with open(query, 'rb') as f:
            response_k100 = user_vecto.lookup_single(f, modality='IMAGE', top_k=100)
        results_k100 = response_k100.json()['results']

        logger.info(response_k100.status_code)
        assert response_k100.status_code is 200
        assert response_k100.content is not None
        logger.info("Checking if there's 17 lookup results: " + str(len(results_k100) == 17))
        assert len(results_k100) is 17
        logger.info("Checking if values in 'data' is string: " + str(isinstance(results_k100[0]['data'], str)))
        assert isinstance(results_k100[0]['data'], str)
        logger.info("Checking if values in 'id' is not empty: " + str(results_k100[round(len(results_k100) / 2)]['id'] is not None))
        assert results_k100[round(len(results_k100) / 2)]['id'] is not None
        logger.info("Checking if values in 'similarity' is float: " + str(isinstance(results_k100[-1]['similarity'], float)))
        assert isinstance(results_k100[-1]['similarity'], float) 

@pytest.mark.update
class TestUpdating:
    
    # Test updating a vector embedding using text on Vecto
    def test_update_single_text_vector_embedding(self):
        text = TestDataset.get_random_text()
        response = user_vecto.update_batch_vector_embeddings(text, modality='TEXT')

        logger.info(response.status_code)
        assert response.status_code is 200
        assert response.content is not None

    # Test updating a vector embedding using image on Vecto
    def test_update_single_image_vector_embedding(self):
        image = TestDataset.get_random_image()
        response = user_vecto.update_batch_vector_embeddings(image, modality='IMAGE')

        logger.info(response.status_code)
        assert response.status_code is 200
        assert response.content is not None

    # Test updating multiple vector embeddings using text on Vecto
    def test_update_batch_text_vector_embedding(self):
        text = TestDataset.get_text_dataset()[:5]
        response = user_vecto.update_batch_vector_embeddings(text, modality='TEXT')

        logger.info(response.status_code)
        assert response.status_code is 200
        assert response.content is not None

    # Test updating multiple vector embeddings using image on Vecto
    def test_update_batch_image_vector_embedding(self):
        image = TestDataset.get_image_dataset()[:5]
        response = user_vecto.update_batch_vector_embeddings(image, modality='IMAGE')

        logger.info(response.status_code)
        assert response.status_code is 200
        assert response.content is not None
    
    # Test updating metadata of a vector embedding on Vecto
    def test_update_single_vector_metadata(self):
        vector_id = random.randrange(0, 10)
        new_metadata = 'new_metadata'
        response = user_vecto.update_batch_vector_metadata([vector_id], [new_metadata])
        ref_db = user_db_twin.get_database()

        # Just a dummy lookup to return the specified ID - check specific entry
        f = io.StringIO('blue')
        lookup_response = user_vecto.lookup_single(f, modality='TEXT', top_k=1, ids=vector_id)
        results = lookup_response.json()['results'][0]

        logger.info(response.status_code)
        assert response.status_code is 200
        assert response.content is not None

        logger.info("Checking if metadata is updated: " + str(results['data'] == new_metadata))
        assert results['data'] == new_metadata

        # Just a dummy lookup to return all the data in the vector space - check other entries
        f = io.StringIO('blue')
        lookup_response = user_vecto.lookup_single(f, modality='TEXT', top_k=100)
        lookup_metadata = []
        for result in lookup_response.json()['results']:
            if result['id'] != vector_id:
                lookup_metadata.append([result['id'], result['data']])
        logger.info("Checking if other metadata is not updated...")
        for result in lookup_metadata:
            id = result[0]
            metadata = result[1]
            assert metadata == ref_db.iloc[id]['metadata']
        logger.info("All other metadata unchanged.")

    # Test updating metadata of multiple vector embeddings on Vecto
    def test_update_batch_vector_metadata(self):
        batch_len = 3
        vector_ids = random.sample(range(10), batch_len)
        new_metadata = ['new_metadata_{}'.format(i) for i in range(batch_len)]
        response = user_vecto.update_batch_vector_metadata(vector_ids, new_metadata)
        ref_db = user_db_twin.get_database()
        
        # Just a dummy lookup to return all the data in the vector space - check other entries
        f = io.StringIO('blue')
        lookup_response = user_vecto.lookup_single(f, modality='TEXT', top_k=batch_len, ids=vector_ids)
        lookup_metadata = []
        for result in lookup_response.json()['results']:
            if result['id'] in vector_ids:
                lookup_metadata.append(result['data'])
        lookup_metadata.sort()

        logger.info(response.status_code)
        assert response.status_code is 200
        assert response.content is not None

        logger.info("Checking if metadata is updated: " + str(lookup_metadata == new_metadata))
        assert lookup_metadata == new_metadata

        # Just a dummy lookup to return all the data in the vector space - check other entries
        f = io.StringIO('blue')
        lookup_response = user_vecto.lookup_single(f, modality='TEXT', top_k=100)
        lookup_metadata = []
        for result in lookup_response.json()['results']:
            if result['id'] != vector_ids:
                lookup_metadata.append([result['id'], result['data']])

        logger.info("Checking if other metadata is not updated...")
        for result in lookup_metadata:
            id = result[0]
            if id not in vector_ids:
                metadata = result[1]
                assert metadata == ref_db.iloc[id]['metadata']
        logger.info("All other metadata unchanged.")
    
@pytest.mark.analogy
class TestAnalogy:
    
    # Test getting an analogy from Vecto
    def test_get_analogy(self): # can be text or images
        query = 'vecto/api-tests/demo_dataset/navy.txt'
        analogy_from = 'vecto/api-tests/demo_dataset/blue.txt'
        analogy_to = 'vecto/api-tests/demo_dataset/orange.txt'
        top_k = 10
        response = user_vecto.get_analogy(query, analogy_from, analogy_to, top_k)
        results = response.json()['results']

        logger.info(response.status_code)
        assert response.status_code is 200
        assert response.content is not None
        logger.info("Checking if number of lookup results is equal to top_k: " + str(len(results) == top_k))
        assert len(results) is top_k
        logger.info("Checking if values in 'data' is string: " + str(isinstance(results[0]['data'], str)))
        assert isinstance(results[0]['data'], str)
        logger.info("Checking if values in 'id' is int: " + str(isinstance(results[round(len(results) / 2)]['id'], int)))
        assert isinstance(results[round(len(results) / 2)]['id'], int)
        logger.info("Checking if values in 'similarity' is float: " + str(isinstance(results[-1]['similarity'], float)))
        assert isinstance(results[-1]['similarity'], float)

    # Test creating an analogy on Vecto
    # Create and delete analogy checks against each other - you need to create one first before you can delete
    def test_create_analogy(self):
        analogy_from = 'vecto/api-tests/demo_dataset/blue.txt'
        analogy_to = 'vecto/api-tests/demo_dataset/orange.txt'
        analogy_id = 1
        response = user_vecto.create_analogy(analogy_id, analogy_from, analogy_to)

        logger.info(response.status_code)
        assert response.status_code is 200
        assert response.content is not None

    # Test deleting an analogy from Vecto
    def test_delete_analogy(self):
        analogy_id = 1
        response = user_vecto.delete_analogy(analogy_id)
        
        logger.info(response.status_code)
        assert response.status_code is 200
        assert response.content is not None

@pytest.mark.delete
class TestDelete:

    # Test deleting a single vector embedding from Vecto
    def test_delete_single_vector_embedding(self):
        vector_id = random.randrange(0, 10)
        response = user_vecto.delete_batch_vector_embeddings([vector_id])
        ref_db = user_db_twin.get_database()
        user_db_twin.update_deleted_ids([vector_id])

        f = io.StringIO('blue')
        lookup_response = user_vecto.lookup_single(f, modality='TEXT', top_k=100)
        results = lookup_response.json()['results']
        deleted_ids = user_db_twin.get_deleted_ids()
       
        logger.info(response.status_code)
        assert response.status_code is 200
        assert response.content is not None
        logger.info("Checking if the length of result is 11: " + str(len(results) == (len(ref_db) - len(deleted_ids))))
        assert len(results) is (len(ref_db) - len(deleted_ids))

    # Test deleting multiple vector embeddings from Vecto
    def test_delete_batch_vector_embedding(self):
        batch_len = 5
        vector_ids = []
        deleted_ids = user_db_twin.get_deleted_ids()
        while len(vector_ids) < batch_len:
            rand_id = random.randrange(0, 10)
            if rand_id not in deleted_ids and rand_id not in vector_ids:
                vector_ids.append(rand_id)
        response = user_vecto.delete_batch_vector_embeddings(vector_ids)
        ref_db = user_db_twin.get_database()
        user_db_twin.update_deleted_ids(vector_ids)

        f = io.StringIO('blue')
        lookup_response = user_vecto.lookup_single(f, modality='TEXT', top_k=100)
        results = lookup_response.json()['results']
       
        logger.info(response.status_code)
        assert response.status_code is 200
        assert response.content is not None
        logger.info("Checking if the length of result is 6: " + str(len(results) == (len(ref_db) - len(deleted_ids))))
        assert len(results) is (len(ref_db) - len(deleted_ids))
    
# To be continued with test_public.py
import io
from vecto import VectoAPI
from test_util import DatabaseTwin, TestDataset
import random
import logging
import pytest

'''
Please update token, vecto_base_url and vector_space_id in *vecto_config.env*
before running `pytest test_public.py` in terminal

You can choose a different seed to in test_util.py

If you run into any errors, you can use pdb.set_trace() to set pytest debugger checkpoint.
See https://docs.pytest.org/en/6.2.x/usage.html#setting-breakpoints for more info.
'''

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fetch Vecto config from environment
import os
token = os.environ['public_token']
vector_space_id = int(os.environ['vector_space_id'])

public_vecto = VectoAPI(token, vector_space_id)
public_db_twin = DatabaseTwin()

# Continued from test_user.py

@pytest.mark.ingest
class TestIngesting:
    
    # Test ingesting one image into Vecto
    def test_ingest_single_image(self):
        image = TestDataset.get_random_image()
        response, _ = public_vecto.ingest_image_batch(image)

        logger.info(response.status_code)
        assert response.status_code == 403

    # Test ingesting multiple images into Vecto
    def test_ingest_image_batch(self):
        batch = TestDataset.get_image_dataset()[:5]
        response, _ = public_vecto.ingest_image_batch(batch)
        
        logger.info(response.status_code)      
        assert response.status_code == 403

    # Test ingesting one text into Vecto
    def test_ingest_single_text(self):
        text = TestDataset.get_random_text()
        response, _ = public_vecto.ingest_text_batch([0], text)
        
        logger.info(response.status_code)
        assert response.status_code == 403

    # Test ingesting multiple texts into Vecto
    def test_ingest_text_batch(self):
        batch = TestDataset.get_text_dataset()
        response, _ = public_vecto.ingest_text_batch(batch.index.tolist()[:5], batch.tolist()[:5])
        
        logger.info(response.status_code)
        assert response.status_code == 403
    

@pytest.mark.lookup
class TestLookup:
    
    # Test doing lookup / search using text on Vecto
    def test_lookup_single_text(self):
        f = io.StringIO('blue')
        response_k5 = public_vecto.lookup_single(f, modality='TEXT', top_k=5)
        results_k5 = response_k5.json()['results']
        
        logger.info(response_k5.status_code)
        assert response_k5.status_code == 200
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
        response_k100 = public_vecto.lookup_single(f, modality='TEXT', top_k=100)
        results_k100 = response_k100.json()['results']

        logger.info(response_k100.status_code)
        assert response_k100.status_code == 200
        assert response_k100.content is not None

        logger.info("Checking if there's 11 lookup results: " + str(len(results_k100) == 11))
        assert len(results_k100) is 11

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
            response_k5 = public_vecto.lookup_single(f, modality='IMAGE', top_k=5)
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
            response_k100 = public_vecto.lookup_single(f, modality='IMAGE', top_k=100)
        results_k100 = response_k100.json()['results']

        logger.info(response_k100.status_code)
        assert response_k100.status_code is 200
        assert response_k100.content is not None

        logger.info("Checking if there's 11 lookup results: " + str(len(results_k100) == 11))
        assert len(results_k100) is 11
        
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
        response = public_vecto.update_batch_vector_embeddings(text, modality='TEXT')

        logger.info(response.status_code)
        assert response.status_code == 403

    # Test updating a vector embedding using image on Vecto
    def test_update_single_image_vector_embedding(self):
        image = TestDataset.get_random_image()
        response = public_vecto.update_batch_vector_embeddings(image, modality='IMAGE')

        logger.info(response.status_code)
        assert response.status_code == 403

    # Test updating multiple vector embeddings using text on Vecto
    def test_update_batch_text_vector_embedding(self):
        text = TestDataset.get_text_dataset()[:5]
        response = public_vecto.update_batch_vector_embeddings(text, modality='TEXT')

        logger.info(response.status_code)
        assert response.status_code == 403

    # Test updating multiple vector embeddings using image on Vecto
    def test_update_batch_image_vector_embedding(self):
        image = TestDataset.get_image_dataset()[:5]
        response = public_vecto.update_batch_vector_embeddings(image, modality='IMAGE')

        logger.info(response.status_code)
        assert response.status_code == 403
    
    # Test updating metadata of a vector embedding on Vecto
    def test_update_single_vector_metadata(self):
        vector_id = random.randrange(0, 10)
        new_metadata = 'new_metadata'
        response = public_vecto.update_batch_vector_metadata([vector_id], [new_metadata])

        logger.info(response.status_code)
        assert response.status_code == 403

    # Test updating metadata of multiple vector embeddings on Vecto
    def test_update_batch_vector_metadata(self):
        batch_len = 3
        vector_ids = random.sample(range(10), batch_len)
        new_metadata = ['new_metadata_{}'.format(i) for i in range(batch_len)]
        response = public_vecto.update_batch_vector_metadata(vector_ids, new_metadata)

        logger.info(response.status_code)
        assert response.status_code == 403
    
@pytest.mark.analogy
class TestAnalogy:
    
    # Test getting an analogy from Vecto
    def test_get_analogy(self): # can be text or images
        query = 'vecto/api-tests/demo_dataset/navy.txt'
        analogy_from = 'vecto/api-tests/demo_dataset/blue.txt'
        analogy_to = 'vecto/api-tests/demo_dataset/orange.txt'
        top_k = 5
        response = public_vecto.get_analogy(query, analogy_from, analogy_to, top_k)
        results = response.json()['results']

        logger.info(response.status_code)
        assert response.status_code is 200
        logger.info("Checking if number of lookup results is equal to top_k: " + str(len(results) == top_k))
        assert len(results) is top_k
        logger.info("Checking if values in 'data' is string: " + str(isinstance(results[0]['data'], str)))
        assert isinstance(results[0]['data'], str)
        logger.info("Checking if values in 'id' is int: " + str(isinstance(results[round(len(results) / 2)]['id'], int)))
        assert isinstance(results[round(len(results) / 2)]['id'], int)
        logger.info("Checking if values in 'similarity' is float: " + str(isinstance(results[-1]['similarity'], float)))
        assert isinstance(results[-1]['similarity'], float)

    # Test creating an analogy on Vecto
    def test_create_analogy(self):
        analogy_from = 'vecto/api-tests/demo_dataset/blue.txt'
        analogy_to = 'vecto/api-tests/demo_dataset/orange.txt'
        analogy_id = 1
        response = public_vecto.create_analogy(analogy_id, analogy_from, analogy_to)

        logger.info(response.status_code)
        assert response.status_code == 403

    # Test deleting an analogy from Vecto
    def test_delete_analogy(self):
        analogy_id = 1
        response = public_vecto.delete_analogy(analogy_id)
        
        logger.info(response.status_code)
        assert response.status_code == 403

@pytest.mark.delete
class TestDelete:

    # Test deleting a single vector embedding from Vecto
    def test_delete_single_vector_embedding(self):
        vector_id = random.randrange(0, 10)
        response = public_vecto.delete_batch_vector_embeddings([vector_id])
       
        logger.info(response.status_code)
        assert response.status_code == 403

    # Test deleting multiple vector embeddings from Vecto
    def test_delete_batch_vector_embedding(self):
        batch_len = 5
        vector_ids = []
        deleted_ids = public_db_twin.get_deleted_ids()
        while len(vector_ids) < batch_len:
            rand_id = random.randrange(0, 10)
            if rand_id not in deleted_ids and rand_id not in vector_ids:
                vector_ids.append(rand_id)
        response = public_vecto.delete_batch_vector_embeddings(vector_ids)
       
        logger.info(response.status_code)
        assert response.status_code == 403
    
    # Test clearing off vector space at the end
    @pytest.mark.clear
    def test_clear_vector_space_entries(self):
        response = public_vecto.delete_vector_space_entries()
        
        logger.info(response.status_code)
        assert response.status_code == 403

# End of test
            
    
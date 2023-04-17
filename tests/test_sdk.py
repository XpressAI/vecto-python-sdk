# Copyright 2022 Xpress AI

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
from vecto import Vecto
from vecto.exceptions import VectoException, ForbiddenException, ServiceException
from test_util import DatabaseTwin, TestDataset
import random
import logging
import pytest
import json
import pathlib

'''
Please ensure that you have token, vecto_base_url and vector_space_id
either using vecto_config.env or exporting them before 
running `pytest test_sdk.py` in terminal

You can choose a different seed in test_util.py

If you run into any errors, you can use pdb.set_trace() to set pytest debugger checkpoint.
See https://docs.pytest.org/en/6.2.x/usage.html#setting-breakpoints for more info.
'''

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
token = os.environ['management_token']
vector_space_id = int(os.environ['vector_space_id'])
vecto_base_url = os.environ['vecto_base_url']


user_vecto = Vecto(token, vector_space_id, vecto_base_url=vecto_base_url)
user_db_twin = DatabaseTwin()

# Clear off vector space before start
@pytest.mark.clear
def test_clear_vector_space_entries():
    user_vecto.delete_vector_space_entries()
    
    f = io.StringIO('blue')
    lookup_response = user_vecto.lookup(f, modality='TEXT', top_k=100)
    
    logger.info("Checking if there's 0 lookup results: " + str(len(lookup_response) == 0))
    assert len(lookup_response) is 0


@pytest.mark.ingest
class TestIngesting:
    
    # Test ingesting one image into Vecto
    def test_ingest_single_image(self):
        image = TestDataset.get_random_image()
        attribute = TestDataset.get_image_attribute(image)
        response = user_vecto.ingest_image(image, attribute['data'])

        assert response is not None

        results = response.ids
        user_db_twin.update_database(results, attribute['data'])
        ref_db = user_db_twin.get_database()


        logger.info('Number of ingested input: ' + str(len(results)))
        assert len(results) == 1 # ingested only 1 input so it should return only list of length 1 
                                    #this should be len(batch)
        
        logger.info(f'Check if ID of last ingested input is {ref_db["id"].iloc[-1]}: ' + 
                        str(results[-1] == ref_db["id"].iloc[-1]))
        assert results[-1] == ref_db["id"].iloc[-1] # last ingested input of vector space so it should be the last entry in ref_db

    # Test ingesting multiple images into Vecto
    def test_ingest_image(self):
        
        batch = TestDataset.get_image_dataset()[:5]
        attribute = TestDataset.get_image_attribute(batch)
        response = user_vecto.ingest_image(batch, attribute['data'])
        results = response.ids
        user_db_twin.update_database(results, attribute['data'])
        ref_db = user_db_twin.get_database()
        
        assert response is not None
        
        logger.info('Number of ingested input:' + str(len(results)))
        assert len(results) == 5 # ingested only 5 input so it should return only list of length 5
        # assert len(results) == len(ref_db)


        logger.info(f'Check if ID of last ingested input is {ref_db["id"].iloc[-1]}: ' + 
                        str(results[-1] == ref_db["id"].iloc[-1]))
        assert results[-1] == ref_db["id"].iloc[-1] # last ingested batch input of vector space so it should be the last entry in ref_db



    # Test ingesting multiple images with source attribute into Vecto
    def test_ingest_image_with_valid_source(self):
        batch = TestDataset.get_image_dataset()[:5]
        data = {'vector_space_id': user_vecto.vector_space_id, 'data': [], 'modality': 'IMAGE'}
    
        vecto_data = []
        files = []
        for path in batch:
    
            temp_data = {}

            relative = "%s/%s" % (path.parent.name, path.name)
            data['data'].append(json.dumps({'relative': relative, "_source": "file:/./%s" % relative}))
            temp_data.update({'attributes': json.dumps({'relative': relative, "_source": "file:/./%s" % relative})})
            temp_data.update({'data': open(path, 'rb')})

            vecto_data.append(temp_data)

        response = user_vecto.ingest(vecto_data, 'IMAGE')
        # for f in files:
        #     f.close()
        results = response.ids
        user_db_twin.update_database(results, data['data'])
        ref_db = user_db_twin.get_database()

        logger.info(response)
        # assert response.status_code is 200
        assert response is not None

        logger.info('Number of ingested input:' + str(len(results)))
        assert len(results) == 5  # ingested only 5 input so it should return only list of length 5

        logger.info(f'Check if ID of last ingested input is {ref_db["id"].iloc[-1]}: ' +
                    str(results[-1] == ref_db["id"].iloc[-1]))
        assert results[-1] == ref_db["id"].iloc[-1]  # last ingested batch input of vector space so it should be the last entry in ref_db

    # Test ingesting one text into Vecto
    def test_ingest_single_text(self):
        text = TestDataset.get_random_text(TestDataset.get_color_dataset)
        index = [0]
        attribute = TestDataset.get_text_attribute(index, text)
        response = user_vecto.ingest_text(text, attribute)
        results = response.ids

        user_db_twin.update_database(results, attribute)
        ref_db = user_db_twin.get_database()
        
        logger.info(response)
        # assert response.status_code is 200
        assert response is not None
        logger.info('Number of ingested input:' + str(len(results)))
        assert len(results) is 1 # ingested only 1 input so it should return only list of length 1
        logger.info(f'Check if ID of last ingested input is {ref_db["id"].iloc[-1]}: ' + 
                        str(results[-1] == ref_db["id"].iloc[-1]))
        assert results[-1] == ref_db["id"].iloc[-1] # last ingested input of vector space so it should be the last entry in ref_db

    # Test ingesting multiple texts into Vecto
    def test_ingest_text(self):
        batch = TestDataset.get_color_dataset()
        attribute = TestDataset.get_text_attribute(batch.index.tolist()[:5], batch.tolist()[:5])
        response = user_vecto.ingest_text(batch.tolist()[:5], attribute)
        results = response.ids
        user_db_twin.update_database(results, attribute)
        ref_db = user_db_twin.get_database()
        
        logger.info(response)
        # assert response.status_code is 200
        assert response is not None
        logger.info('Number of ingested input:' + str(len(results)))
        assert len(results) is 5 # ingested only 5 input so it should return only list of length 5
        logger.info(f'Check if ID of last ingested input is {ref_db["id"].iloc[-1]}: ' + 
                        str(results[-1] == ref_db["id"].iloc[-1]))
        assert results[-1] == ref_db["id"].iloc[-1] # last ingested batch input of vector space so it should be the last entry in ref_db
    
    # Check if the number of entries in Vecto aligns with DatabaseTwin
    def test_ingested(self):
        ref_db = user_db_twin.get_database()

        logger.info('Length of ref_df is :' + str(len(ref_db)))
        assert len(ref_db) is len(user_vecto.lookup(" ", modality='TEXT', top_k=100))

@pytest.mark.lookup
class TestLookup:
    
    # Test doing lookup / search using text on Vecto
    def test_lookup_on_text(self):
        f = io.StringIO('blue')
        response_k5 = user_vecto.lookup(f, modality='TEXT', top_k=5)
        results_k5 = response_k5

        # logger.info(response_k5)
        # assert response_k5.status_code is 200
        assert response_k5 is not None
        logger.info("Checking if there's 5 lookup results: " + str(len(results_k5) == 5))
        assert len(results_k5) is 5
        logger.info("Checking if values in 'data' is string: " + str(isinstance(results_k5[0].attributes, str)))
        assert isinstance(results_k5[0].attributes, str)
        logger.info("Checking if values in 'id' is not empty: " + str(results_k5[round(len(results_k5) / 2)].id is not None))
        assert results_k5[round(len(results_k5) / 2)].id is not None
        logger.info("Checking if values in 'similarity' is float: " + str(isinstance(results_k5[-1].similarity, float)))
        assert isinstance(results_k5[-1].similarity, float)

        # top_k=100 is to return everything in the vector space
        response_k100 = user_vecto.lookup(f, modality='TEXT', top_k=100)
        results_k100 = response_k100

        # logger.info(response_k100)
        # assert response_k100.status_code is 200
        assert response_k100 is not None
        logger.info("Checking if there's 17 lookup results: " + str(len(results_k100) == 17))
        assert len(results_k100) is 17
        logger.info("Checking if values in 'data' is string: " + str(isinstance(results_k100[0].attributes, str)))
        assert isinstance(results_k100[0].attributes, str)
        logger.info("Checking if values in 'id' is not empty: " + str(results_k100[round(len(results_k100) / 2)].id is not None))
        assert results_k100[round(len(results_k100) / 2)].id is not None
        logger.info("Checking if values in 'similarity' is float: " + str(isinstance(results_k100[-1].similarity, float)))
        assert isinstance(results_k100[-1].similarity, float)
    
    # Test doing lookup / search using image on Vecto
    def test_lookup_on_image(self):
        query = TestDataset.get_random_image()[0]
        with open(query, 'rb') as f:
            response_k5 = user_vecto.lookup(f, modality='IMAGE', top_k=5)
        results_k5 = response_k5

        assert response_k5 is not None
        logger.info("Checking if there's 5 lookup results: " + str(len(results_k5) == 5))
        assert len(results_k5) is 5
        logger.info("Checking if values in 'data' is string: " + str(isinstance(results_k5[0].attributes, str)))
        assert isinstance(results_k5[0].attributes, str)
        logger.info("Checking if values in 'id' is not empty: " + str(results_k5[round(len(results_k5) / 2)].id is not None))
        assert results_k5[round(len(results_k5) / 2)].id is not None
        logger.info("Checking if values in 'similarity' is float: " + str(isinstance(results_k5[-1].similarity, float)))
        assert isinstance(results_k5[-1].similarity, float)

        with open(query, 'rb') as f:
            response_k100 = user_vecto.lookup(f, modality='IMAGE', top_k=100)
        results_k100 = response_k100

        assert response_k100 is not None
        logger.info("Checking if there's 17 lookup results: " + str(len(results_k100) == 17))
        assert len(results_k100) is 17
        logger.info("Checking if values in 'data' is string: " + str(isinstance(results_k100[0].attributes, str)))
        assert isinstance(results_k100[0].attributes, str)
        logger.info("Checking if values in 'id' is not empty: " + str(results_k100[round(len(results_k100) / 2)].id is not None))
        assert results_k100[round(len(results_k100) / 2)].id is not None
        logger.info("Checking if values in 'similarity' is float: " + str(isinstance(results_k100[-1].similarity, float)))
        assert isinstance(results_k100[-1].similarity, float) 


    # Test using lookup_image and lookup_text on Vecto
    def test_lookup_image_from_filepath(self):

        query = TestDataset.get_random_image()[0]
        logger.info("Checking that lookup_text_from_filepath can handle file paths")
        assert user_vecto.lookup_image_from_filepath(query, 5) is not None

        invalid_path = "/path/to/nonexistent/image.jpg"
        logger.info("Checking that lookup_text_from_filepath correctly detects an incorrect file path")

        with pytest.raises(FileNotFoundError, match="The file was not found."):
            user_vecto.lookup_image_from_filepath(invalid_path, 5)


    def test_lookup_image_from_url(self):

        logger.info("Checking that the method returns results when given a valid image URL")
        url = 'https://picsum.photos/300/200'
        response = user_vecto.lookup_image_from_url(url, 5)
        assert response is not None

        logger.info("Checking that lookup_image_from_url correctly detects an invalid URL")
        from urllib.error import URLError

        invalid_url = "http://invalid-url.example.com!/image.jpg"
        try:
            user_vecto.lookup_image_from_url(invalid_url, 5)
        
        except URLError:
            logger.info("URLError raised as expected")
        else:
            logger.error("Expected URLError not raised")

    def test_lookup_image_from_binary(self):

        logger.info("Checking that the method returns results when given text data as a file-like object")
        query = TestDataset.get_random_image()[0]
        with open(query, 'rb') as f:
            assert user_vecto.lookup_image_from_binary(f, 5) is not None

    def test_lookup_text_from_path(self):

        logger.info("Checking that the method returns results when given a valid file path")
        query = os.path.join("tests", "demo_dataset", "blue.txt")
        assert user_vecto.lookup_text_from_filepath(query, 5) is not None

        logger.info("Checking that an exception is raised when the file path is invalid")
        non_existing_path = pathlib.Path("non_existing_file.txt")
        
        with pytest.raises(FileNotFoundError):
            user_vecto.lookup_text_from_filepath(non_existing_path, top_k=5)

    def test_lookup_text_from_str(self):

        logger.info("Checking that the method returns results when given text data as a string")
        assert user_vecto.lookup_text_from_str('blue', 5) is not None


    def test_lookup_text_from_url(self):

        logger.info("Checking that the method returns results when given a valid image URL")
        url = 'https://raw.githubusercontent.com/XpressAI/vecto-python-sdk/main/tests/demo_dataset/blue.txt'
        response = user_vecto.lookup_text_from_url(url, 5)
        assert response is not None

        logger.info("Checking that lookup_text_from_url correctly detects an invalid URL")
        from urllib.error import URLError

        invalid_url = "http://invalid-url.example.com/text.txt"
        try:
            user_vecto.lookup_text_from_url(invalid_url, 5)
        
        except URLError:
            logger.info("URLError raised as expected")
        else:
            logger.error("Expected URLError not raised")


    def test_lookup_text_from_binary(self):

        logger.info("Checking that the method returns results when given text data as a file-like object")
        f = io.StringIO('blue')
        assert user_vecto.lookup_text_from_binary(f, 5) is not None


@pytest.mark.update
class TestUpdating:
    
    # Test updating a vector embedding using text on Vecto
    def test_update_single_text_vector_embedding(self):
        text = TestDataset.get_random_text(TestDataset.get_color_dataset)
        vector_ids = random.sample(range(len(text)), len(text))
        
        updated_vector = []
        
        for file, vector_id in zip(text, vector_ids):
            updated_vector.append({
            'id': vector_id,
            'data': io.StringIO(file),
        })

        user_vecto.update_vector_embeddings(updated_vector, modality='TEXT')

    # Test updating a vector embedding using image on Vecto
    def test_update_single_image_vector_embedding(self):
        image = TestDataset.get_random_image()
        vector_ids = random.sample(range(len(image)), len(image))

        updated_vector = []
        
        for file, vector_id in zip(image, vector_ids):
            updated_vector.append({
            'id': vector_id,
            'data': open(file, 'rb')
        })

        user_vecto.update_vector_embeddings(updated_vector, modality='IMAGE')
        
        for file in updated_vector:
            file['data'].close()

    # Test updating multiple vector embeddings using text on Vecto
    def test_update_batch_text_vector_embedding(self):
        text = TestDataset.get_color_dataset()[:5]
        vector_ids = random.sample(range(len(text)), len(text))

        updated_vector = []
        
        for file, vector_id in zip(text, vector_ids):
            updated_vector.append({
            'id': vector_id,
            'data': io.StringIO(file)
        })

        user_vecto.update_vector_embeddings(updated_vector, modality='TEXT')


    # Test updating multiple vector embeddings using image on Vecto
    def test_update_batch_image_vector_embedding(self):
        image = TestDataset.get_image_dataset()[:5]
        vector_ids = random.sample(range(len(image)), len(image))

        updated_vector = []
        
        for file, vector_id in zip(image, vector_ids):
            updated_vector.append({
            'id': vector_id,
            'data': open(file, 'rb')
        })

        user_vecto.update_vector_embeddings(updated_vector, modality='IMAGE')
        
        for file in updated_vector:
            file['data'].close()

    # Test updating attribute of a vector embedding on Vecto
    def test_update_single_vector_attribute(self):
        vector_id = random.randrange(0, 10)
        new_attribute = 'new_attribute'

        updated_attribute = [{
            'id': vector_id,
            'attributes': json.dumps(new_attribute),
        }]

        user_vecto.update_vector_attribute(updated_attribute)
        ref_db = user_db_twin.get_database()

        # Just a dummy lookup to return the specified ID - check specific entry
        f = io.StringIO('blue')
        lookup_response = user_vecto.lookup(f, modality='TEXT', top_k=1, ids=vector_id)
        results = lookup_response[0]

        logger.info("Checking if attribute is updated: " + str(results.attributes == new_attribute))
        assert results.attributes == new_attribute

        # Just a dummy lookup to return all the data in the vector space - check other entries
        f = io.StringIO('blue')
        lookup_response = user_vecto.lookup(f, modality='TEXT', top_k=100)
        lookup_attribute = []

        #need to iterate though this object
        for result in lookup_response:
            if result.id != vector_id:
                lookup_attribute.append([result.id, result.attributes])
        logger.info("Checking if other attribute is not updated...")
        for result in lookup_attribute:
            id = result[0]
            attribute = result[1]
            assert attribute == ref_db.iloc[id]['attribute']
        logger.info("All other attribute unchanged.")

    # Test updating attribute of multiple vector embeddings on Vecto
    def test_update_vector_attribute(self):
        batch_len = 3
        vector_ids = random.sample(range(10), batch_len)
        new_attribute = ['new_attribute_{}'.format(i) for i in range(batch_len)]

        updated_attribute = []
        
        for attribute, vector_id in zip(new_attribute, vector_ids):
            updated_attribute.append({
            'id': vector_id,
            'attributes': json.dumps(attribute)
        })

        user_vecto.update_vector_attribute(updated_attribute)
        ref_db = user_db_twin.get_database()
        
        # Just a dummy lookup to return all the data in the vector space - check other entries
        f = io.StringIO('blue')
        lookup_response = user_vecto.lookup(f, modality='TEXT', top_k=batch_len, ids=vector_ids)
        lookup_attribute = []
        for result in lookup_response:
            if result.id in vector_ids:
                lookup_attribute.append(result.attributes)
        lookup_attribute.sort()

        logger.info("Checking if attribute is updated: " + str(lookup_attribute == new_attribute))
        assert lookup_attribute == new_attribute

        # Just a dummy lookup to return all the data in the vector space - check other entries
        f = io.StringIO('blue')
        lookup_response = user_vecto.lookup(f, modality='TEXT', top_k=100)
        lookup_attribute = []
        for result in lookup_response:
            if result.id != vector_ids:
                lookup_attribute.append([result.id, result.attributes])

        logger.info("Checking if other attribute is not updated...")
        for result in lookup_attribute:
            id = result[0]
            if id not in vector_ids:
                attribute = result[1]
                assert attribute == ref_db.iloc[id].attribute
        logger.info("All other attribute unchanged.")
    
@pytest.mark.analogy
class TestAnalogy:
    
    # Test getting an analogy from Vecto
    def test_compute_analogy(self): # can be text or images
        query = io.StringIO('navy')
        analogy_start_end = {
            'start': io.StringIO('blue'),
            'end': io.StringIO('orange')
        }
        top_k = 10
        modality = 'TEXT'
        response = user_vecto.compute_analogy(query, analogy_start_end, top_k, modality)
        results = response

        assert response is not None
        logger.info("Checking if number of lookup results is equal to top_k: " + str(len(results) == top_k))
        assert len(results) is top_k
        logger.info("Checking if values in 'data' is string: " + str(isinstance(results[0].attributes, str)))
        assert isinstance(results[0].attributes, str)
        logger.info("Checking if values in 'id' is int: " + str(isinstance(results[round(len(results) / 2)].id, int)))
        assert isinstance(results[round(len(results) / 2)].id, int)
        logger.info("Checking if values in 'similarity' is float: " + str(isinstance(results[-1].similarity, float)))
        assert isinstance(results[-1].similarity, float)



    # Test creating an analogy on Vecto
    # Create and delete analogy checks against each other - you need to create one first before you can delete
    
    # As the API for creating analogies will change, these two will temporarily be commented out
    # def test_create_analogy(self):
    #     analogy_start = 'tests/demo_dataset/blue.txt'
    #     analogy_end = 'tests/demo_dataset/orange.txt'
    #     analogy_id = 1
    #     user_vecto.create_analogy(analogy_id, analogy_start, analogy_end)

    # Test deleting an analogy from Vecto
    # def test_delete_analogy(self):
    #     analogy_id = 1
    #     user_vecto.delete_analogy(analogy_id)

@pytest.mark.delete
class TestDelete:

    # Test deleting a single vector embedding from Vecto
    def test_delete_single_vector_embedding(self):
        vector_id = random.randrange(0, 10)
        user_vecto.delete_vector_embeddings([vector_id])
        ref_db = user_db_twin.get_database()
        user_db_twin.update_deleted_ids([vector_id])

        f = io.StringIO('blue')
        lookup_response = user_vecto.lookup(f, modality='TEXT', top_k=100)
        results = lookup_response
        deleted_ids = user_db_twin.get_deleted_ids()
       
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
        user_vecto.delete_vector_embeddings(vector_ids)
        ref_db = user_db_twin.get_database()
        user_db_twin.update_deleted_ids(vector_ids)

        f = io.StringIO('blue')
        lookup_response = user_vecto.lookup(f, modality='TEXT', top_k=100)
        results = lookup_response
       
        logger.info("Checking if the length of result is 6: " + str(len(results) == (len(ref_db) - len(deleted_ids))))
        assert len(results) is (len(ref_db) - len(deleted_ids))


@pytest.mark.exception
class TestExceptions:

    # Test compute analogy from Vecto
    def test_compute_analogy_from_list(self): 

        user_vecto.delete_vector_space_entries()
        batch = TestDataset.get_profession_dataset()
        response = user_vecto.ingest_text(batch, batch)
        results = response.ids
        user_db_twin.update_database(results, batch)

        query = 'King'
        analogy_start = ['Male', 'Husband']
        analogy_end = ['Female', 'Wife']
        
        analogy_start_end = []
        for start, end in zip(analogy_start, analogy_end):
            analogy_start_end.append({
            'start': start,
            'end': end
        })

        top_k = 20
        modality = 'TEXT'
        response = user_vecto.compute_analogy(query, analogy_start_end, top_k, modality)
        results = response

        logger.info("Checking if values in 'data' is queen: " + str(isinstance(results[0].attributes, str)))
        
        assert "Queen" in results[1].attributes # TODO: once the ingest is fixed, it should return the first result


    def test_invalid_vector_space(self):

        token = 0
        vector_space_id = 0

        invalid_user_vecto = Vecto(token, vector_space_id, vecto_base_url=vecto_base_url)

        with pytest.raises(VectoException) as e:

            invalid_user_vecto.lookup("BLUE", modality='TEXT', top_k=100)

    #Test ingesting multiple images with invalid source attribute into Vecto
    def test_ingest_image_with_invalid_source(self):

        batch = TestDataset.get_image_dataset()[:5]
        data = {'vector_space_id': user_vecto.vector_space_id, 'data': [], 'modality': 'IMAGE'}
    
        vecto_data = []
        files = []
        for path in batch:
    
            temp_data = {}

            relative = "%s/%s" % (path.parent.name, path.name)
            data['data'].append(json.dumps({'relative': relative, "_source": "file:/./%s" % relative}))
            temp_data.update({'attributes': json.dumps({'relative': relative, "_source": "file:/./%s" % relative})})
            temp_data.update({'data': open(path, 'rb')})

        with pytest.raises(VectoException) as e:
            logger.info(e)
            response = user_vecto.ingest(vecto_data, 'IMAGE')
            logger.info(response)
            logger.info(e)

            for f in files:
                f.close()
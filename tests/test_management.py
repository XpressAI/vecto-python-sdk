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
import random
import logging
import pytest
import json
import pathlib

from vecto.schema import (VectoIngestData, VectoEmbeddingData, VectoAttribute, VectoAnalogyStartEnd,
                    IngestResponse, LookupResult, VectoModel, VectoVectorSpace, VectoUser,
                    VectoToken, VectoNewTokenResponse, DataPage, DataEntry)

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

models = ""
vector_spaces = ""
test_vector_space = ""
test_vs_name = "management_sdk_test"
test_vs_token = ""
test_vs_data = ""

# Currently test disabled as a bug prevents certain VS to be deleted.
# @pytest.mark.management
# def test_clear_all_vector_spaces():
#     user_vecto.delete_all_vector_spaces()
#     vector_spaces = user_vecto.list_vector_spaces()
#     logger.info("Checking if there are 0 vector spaces: " + str(len(vector_spaces) == 0))
#     assert len(vector_spaces) is 0

@pytest.mark.management
def test_list_models():
    global models
    models = user_vecto.list_models()

    logger.info("Check if each element in the list is an instance of VectoModel")
    for model in models:
        assert isinstance(model, VectoModel)


@pytest.mark.management
def test_list_vector_spaces():
    global vector_spaces
    vector_spaces = user_vecto.list_vector_spaces()

    logger.info("Check if each element in the list is an instance of VectoVectorSpace")

    for vector_space in vector_spaces:
        assert isinstance(vector_space, VectoVectorSpace)

@pytest.mark.management
def test_create_vector_space():

    global test_vector_space

    model_id = 1  # CLIP
    test_vector_space = user_vecto.create_vector_space(test_vs_name, model_id)
    logger.info("Check if each new vector space is created.")
    assert len(user_vecto.list_vector_spaces()) == len(vector_spaces) + 1
    assert test_vector_space.name == test_vs_name
    assert test_vector_space.model.id == model_id

    # TODO: Make test that passes model string and invalid models that raises ModelNotFoundException

@pytest.mark.management
def test_get_vector_space():
    logger.info("Checking if a vector space can be searched by id")
    vector_space = user_vecto.get_vector_space(test_vector_space.id)
    assert test_vector_space == vector_space

@pytest.mark.management
def test_get_vector_space_by_name():
    logger.info("Checking if a vector space can be searched by name")
    vector_spaces = user_vecto.get_vector_space_by_name(test_vs_name)
    assert len(vector_spaces) >= 1
    for vector_space in vector_spaces:
        assert vector_space.name == test_vs_name


@pytest.mark.management
def test_rename_vector_space():
    logger.info("Check if vector space can be renamed.")
    updated_name = "updated_test_vector_space"
    updated_vector_space = user_vecto.rename_vector_space(test_vector_space.id, updated_name)
    assert updated_vector_space.name == updated_name


@pytest.mark.management
def test_get_user_information():
    logger.info("Check if user info can be fetched.")

    response = user_vecto.get_user_information()
    assert isinstance(response, VectoUser)

@pytest.mark.management
def test_get_tokens():
    tokens = user_vecto.list_tokens()

    logger.info("Check if each element in the list is an instance of VectoToken")

    for token in tokens:
        assert isinstance(token, VectoToken)


@pytest.mark.management
def test_create_token():
    global test_vs_token
    logger.info("Check if token is deleted")
    test_vs_token = user_vecto.create_token("Test SDK Token", "USAGE", test_vector_space.id, False)
    assert isinstance(test_vs_token, VectoNewTokenResponse)

    # TODO: Make tests that checks token access level 


@pytest.mark.management
def test_delete_token():
    logger.info("Check if the vector space token is deleted.")
    user_vecto.delete_token(test_vs_token.id)

    token_list = user_vecto.list_tokens()
    assert not any(token.id == test_vs_token.id for token in token_list)

@pytest.mark.management
def test_listing():
    global test_vs_data
    logger.info("Check if vector space data can be listed and the return types are correct.")
    test_vs_data = user_vecto.list_vector_space_data(vector_space_id, 10, 0)
    assert isinstance(test_vs_data, DataPage)
    assert isinstance(test_vs_data.elements[0], DataEntry)

@pytest.mark.management
def test_delete_data():
    global test_vs_data
    logger.info("Check if vector space data can be deleted.")
    dataEntry = test_vs_data.elements[0]
    user_vecto.delete_vector_space_entry(vector_space_id, dataEntry.id)
    updated_vs_data = user_vecto.list_vector_space_data(vector_space_id, 10, 0)
    updatedDataEntry = updated_vs_data.elements[0]
    assert dataEntry != updatedDataEntry

@pytest.mark.management
def test_delete_vector_space():

    user_vecto.delete_vector_space(test_vector_space.id)
    updated_vs_list = user_vecto.list_vector_spaces()
    id_exists = any(vecto_vector_space.id == test_vector_space.id for vecto_vector_space in updated_vs_list)
    logger.info("Check if the vector space is deleted.")
    assert not id_exists


@pytest.mark.metrics
def test_usage(): 
    from datetime import datetime
    today = datetime.now()
    usage_response = user_vecto.usage(today.year, today.month)
    logger.info("Checking that usage returns a valid response.")
    assert usage_response is not None
    logger.info("Checking that usage for today is not empty.")
    assert usage_response.usage.lookups.dailyMetrics[today.day-1].count > 0
    assert usage_response.usage.indexing.dailyMetrics[today.day-1].count > 0


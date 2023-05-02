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

from vecto.schema import LookupResult, IngestResponse
import logging
import pytest

import os 
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

import vecto

@pytest.mark.vectorspace
def test_create_vector_space_if_not_exists():
    logger.info("Checking if a vector space can be created if it doesn't exist")
    vs = vecto.VectorSpace("test_space")
    initial_exists = vs.exists()
    
    if not initial_exists:
        vs.create(model="CLIP", modality="IMAGE")
        assert vs.vector_space_id is not None
        assert vs.model == "CLIP"
        assert vs.modality == "IMAGE"
    else:
        assert vs.vector_space_id is not None

@pytest.mark.vectorspace
def test_vectorspace_clear_entries():
    logger.info("Checking if vector space entries can be cleared")
    vs = vecto.VectorSpace("test_space")
    vs.clear_entries()
    result = vs.lookup_text("test text", 5)
    assert len(result) is 0

@pytest.mark.vectorspace
def test_vectorspace_ingest_image():
    logger.info("Checking if an image can be ingested into a vector space")
    vs = vecto.VectorSpace("test_space", modality="IMAGE")
    result = vs.ingest_image("tests/demo_dataset/blue1.png", "test_attribute")
    assert isinstance(result, IngestResponse)

@pytest.mark.vectorspace
def test_vectorspace_ingest_text():
    logger.info("Checking if text can be ingested into a vector space")
    vs = vecto.VectorSpace("test_space", modality="TEXT")
    result = vs.ingest_text("test text", "test_attribute")
    assert isinstance(result, IngestResponse)

@pytest.mark.vectorspace
def test_vectorspace_lookup_image():
    logger.info("Checking if an image can be looked up in a vector space")
    vs = vecto.VectorSpace("test_space", modality="IMAGE")
    result = vs.lookup_image("tests/demo_dataset/blue1.png", 5)
    assert isinstance(result[0], LookupResult)

@pytest.mark.vectorspace
def test_vectorspace_lookup_text():
    logger.info("Checking if text can be looked up in a vector space")
    vs = vecto.VectorSpace("test_space", modality="TEXT")
    result = vs.lookup_text("test text", 5)
    assert isinstance(result[0], LookupResult)

@pytest.mark.vectorspace
def test_vectorspace_compute_text_analogy():
    logger.info("Checking if a text analogy can be computed in a vector space")
    vs = vecto.VectorSpace("test_space", modality="TEXT")
    result = vs.compute_text_analogy("dummy_query", {"start":"start_text", "end":"end_text"}, 5)
    assert isinstance(result[0], LookupResult)

@pytest.mark.vectorspace
def test_vectorspace_compute_image_analogy():
    logger.info("Checking if an image analogy can be computed in a vector space")
    vs = vecto.VectorSpace("test_space", modality="IMAGE")
    image1 = "tests/demo_dataset/blue1.png"
    image2 = "tests/demo_dataset/blue2.png"
    image3 = "tests/demo_dataset/green1.png"
    result = vs.compute_image_analogy(open(image3, 'rb'),
                                      {"start":open(image1, 'rb'), "end":open(image2, 'rb')},
                                      5)
    assert isinstance(result[0], LookupResult)
<p align="center">
<a href="https://www.vecto.ai/">
<img src="https://user-images.githubusercontent.com/68586800/192857099-499146bb-5570-4702-a88f-bb4582e940c0.png" width="300"/>
</a>
</p>

# Vecto Python SDK

## Building the Wheel and Installation
Run `python setup.py bdist_wheel --universal` to create a .whl file in the dist folder.

You can install that wheel file with `pip install dist/vecto-*.whl` into your current environment (if the file is in the current working directory).


    

## Sample Usage

```
import Vecto from vecto
vs = Vecto(token="YOUR_TOKEN", vector_space_id=YOUR_ID)

vs.lookup("Blue", "TEXT", top_k=5)
```

## Available Functions

```
    ingest
        ingest a batch of data into Vecto. Use batch of 1 for single entry.

    ingest_image
        ingest a batch of images into Vecto. Use batch of 1 for single image.
    
    ingest_text
        ingest a batch of text into Vecto.  Use batch of 1 for single text.
    
    lookup
        search on Vecto, based on the lookup item.
    
    update_vector_embeddings
        update current vector embeddings with new one.

    update_vector_metadata
        update current vector metadata with new one.

    get_analogy
        get an analogy from Vecto.
        It is also possible to do multiple analogies in one request body.
    
    create_analogy
        create an analogy and store in Vecto.
        It is also possible to do multiple analogies in one request body.
    
    delete_analogy
        delete an analogy that is stored in Vecto.

    delete_vector_embeddings
        delete vector embeddings that is stored in Vecto.

    delete_vector_space_entries
        delete the current vector space in Vecto. All ingested entries will be deleted as well.
```

## Running the Tests
We've setup an [action](https://github.com/XpressAI/vecto-python-sdk/actions/workflows/run-tests.yml) to automate the API tests. If you'd like to run the tests locally, export a valid `user_token`, `public_token`, and `vector_space_id`, then run:
```
pytest tests/test_user.py
pytest tests/test_public.py
```
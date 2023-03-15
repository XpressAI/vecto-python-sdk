<p align="center">
<a href="https://www.vecto.ai/">
<img src="https://user-images.githubusercontent.com/68586800/192857099-499146bb-5570-4702-a88f-bb4582e940c0.png" width="300"/>
</a>
</p>
<p align="center">
  <a href="https://docs.vecto.ai/docs/">Docs</a> •
  <a href="https://www.xpress.ai/blog/">Blog</a> •
  <a href="https://discord.com/invite/vgEg2ZtxCw">Discord</a>
<br>

# Vecto Python SDK
Official Python SDK for [Vecto](https://www.vecto.ai/), the database software that puts intelligent search and powerful models at your fingertips, allowing you to leverage the full potential of AI in mere minutes. 

## Installation

```
pip install vecto-sdk
```


## Building the Wheel
If you would like to build your own wheel, run `python setup.py bdist_wheel --universal` which creates a .whl file in the dist folder. You can install that wheel file with `pip install dist/vecto-*.whl` into your current environment (if the file is in the current working directory).
    
## Sample Usage

```
import Vecto from vecto
vs = Vecto(token="YOUR_TOKEN", vector_space_id=YOUR_ID)

vs.lookup("Blue", "TEXT", top_k=5)
```

Sign up for your access [here](https://www.vecto.ai/contactus). 


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

    update_vector_attribute
        update current vector attribute with new one.

    compute_analogy
        compute an analogy from Vecto.

    delete_vector_embeddings
        delete vector embeddings that is stored in Vecto.

    delete_vector_space_entries
        delete the current vector space in Vecto. All ingested entries will be deleted as well.
```

## Running the Tests
We've setup an [action](https://github.com/XpressAI/vecto-python-sdk/actions/workflows/run-tests.yml) to automate the API tests. If you'd like to run the tests locally, export a valid `user_token`, `public_token`, and `vector_space_id`, then run:

```
pytest tests/test_sdk.py
```

## Developers Discord
Have any questions? Feel free to chat with the devs at our [Discord](https://discord.com/invite/vgEg2ZtxCw)!

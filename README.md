# vecto-python-sdk

## Building the Wheel and Installation
Run `python setup.py bdist_wheel --universal` to create a .whl file in the dist folder.

You can install that wheel file with pip install vecto-*.whl into your current environment (if the file is in the current working directory).

## Running the Tests
Currently the tests are ported from the [vecto api tests repository](https://github.com/XpressAI/vecto-api-test). To run them, export `user_token`, `public_token`, and `vector_space_id` as variables from the [.env](https://github.com/XpressAI/vecto-api-test/blob/main/vecto_config.env), then run:
```
pytest vecto/api-tests/test_user.py
pytest vecto/api-tests/test_public.py
```

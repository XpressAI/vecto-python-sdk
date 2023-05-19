<p align="center">
<a href="https://www.vecto.ai/">
<img src="https://user-images.githubusercontent.com/68586800/192857099-499146bb-5570-4702-a88f-bb4582e940c0.png" width="300"/>
</a>
</p>
<p align="center">
  <a href="https://docs.vecto.ai/">Docs</a> •
  <a href="https://www.xpress.ai/blog/">Blog</a> •
  <a href="https://discord.com/invite/wtYbXvPPfD">Discord</a>
<br>

# Vecto Python SDK
Official Python SDK for [Vecto](https://www.vecto.ai/), the database software that puts intelligent search and powerful models at your fingertips, allowing you to leverage the full potential of AI in mere minutes. 

## Installation

```
pip install vecto-sdk
```


For the token, sign up for your access [here](https://www.vecto.ai/contactus).


## Building the Wheel
If you would like to build your own wheel, run `python setup.py bdist_wheel --universal` which creates a .whl file in the dist folder. You can install that wheel file with `pip install dist/vecto-*.whl` into your current environment (if the file is in the current working directory).

## Sample Usage

For first time users, we recommend using our `VectorSpace` interface.

### Find Nearest Neighbors

```
import vecto
vecto.api_key = os.getenv("VECTO_API_KEY", "")
vector_space = vecto.VectorSpace("my-cool-ai")

for animal in ["lion", "wolf", "cheetah", "giraffe", "elephant", "rhinoceros", "hyena", "zebrah"]:
    vector_space.ingest_text(animal, { 'text': animal, 'region': 'Africa' })

similar_animals = vector_space.lookup_text("cat", top_k=3)
                        
for animal in similar_animals:
    print(f"{animal.attributes['text']} similarity: {animal.similarity:.2%}")

# Prints: "lion similarity: 84.91%"
```

### Ingest Text or Images
```
import vecto
from pathlib import Path
vecto.api_key = os.getenv("VECTO_API_KEY", "")
vector_space = vecto.VectorSpace("my-cool-image-ai")

if not vector_space.exists():
    vector_space.create(model='CLIP', modality='IMAGE') 

for animal in ["lion.png", "wolf.png", "cheetah.png", "giraffe.png", "elephant.png", "rhinoceros.png", "hyena.png", "zebra.png"]:
    vector_space.ingest_image(Path(animal), { 'text': animal.replace('.png', ''), 'region': 'Africa' })

similar_animals = vector_space.lookup_image(Path("cat.png"), top_k=1)

for animal in similar_animals:
    print(f"{animal.attributes['text']}")

# Prints: lion
```

### Looking up by Analogy


```
import vecto
vecto.api_key = os.getenv("VECTO_API_KEY", "")
vector_space = vecto.VectorSpace("word_space")

if not vector_space.exists():
    vector_space.create(model='SBERT', modality='TEXT') 

for word in ["man", "woman", "child", "mother", "father", "boy", "girl", "king", "queen"]:
    vector_space.ingest_text(word, { 'text': word })

analogy = vector_space.compute_text_analogy("king", { 'start': 'man', 'end': 'woman' }, top_k=3)

for word in analogy:
    print(f"{word.attributes['text']} similarity: {word.similarity:.2%}")

# Prints: "queen similarity: 93.41%"
```

For more advanced capabilities including management access, we recommend using the core Vecto class.


## Developers Discord
Have any questions? Feel free to chat with the devs at our [Discord](https://discord.com/invite/wtYbXvPPfD)!

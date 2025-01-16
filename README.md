# StableHLO Model Exports

_Exports of popular models in [StableHLO](https://openxla.org/stablehlo)._

## Available Exports

Exports with large literals (weights) elided are available in the
[`exports/`](exports) folder.

**Currently this includes:**
- [GDM AlphaFold](https://github.com/google-deepmind/alphafold)
- [GDM Searchless Chess](https://github.com/google-deepmind/searchless_chess)
- [FlaxResNet50](https://huggingface.co/docs/transformers/en/model_doc/resnet#transformers.FlaxResNetModel)
- [hf_BERT](https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertModel)
- _More to come_

## Building and Exporting

### Setup Dependencies

I recommend a fresh venv for exporting, this has been tested only using py3.11
but py3.10 is likely to work as well.

```sh
python3 -m venv venv
source venv/bin/activate
(cd models/ && ./setup.sh)
```

### Export Models

To export all models use the `all` flag, this is the default so it can be omit.

```sh
export PYTHONPATH="$(pwd)/models:$(pwd)/models/gdm_searchless_chess"
python models/export.py --models=all
```

A list of models can be specified using the `models` flag if needed:

```py
python models/export.py --models=searchless_chess_9m
```

## View Exports

All exports can be viewed in the `exports/` folder, the `.mlir` files have
large constants elided for readability. All `.mlir.bc` files have the constants
embedded.

_Note:_ Some of the files with many large constants do not have `.mlir.bc` files
since they don't fit in git well. For these files you will need to run the
export scripts, or open a ticket with the request end we can figure out where
to host these.

To view the `.mlir.bc` file, there are two options:

### Using `stablehlo-translate` or `stablehlo-opt`

If you have access to `stablehlo-opt` and `stablehlo-translate` from building
the StableHLO repository:

```sh
stablehlo-translate --deserialize alphafold.mlir.bc
```

or

```sh
stablehlo-opt --pass-pipeline='builtin.module(stablehlo-deserialize)' alphafold.mlir.bc
```

### Using the StableHLO python bindings

On Linux we have ready-to-use python bindings for StableHLO published nightly:

```sh
pip install stablehlo -f https://github.com/openxla/stablehlo/releases/expanded_assets/dev-wheels
```

Once installed the following script can be used to deserialize and print the IR:

```python
from mlir.dialects import stablehlo
from mlir.ir import Context, Location

filepath="/path/to/stablehlo-exports/exports/alphafold.mlir.bc"
with open(filepath, 'rb') as file:
  bytecode = file.read()

with Context() as ctx:
  module = stablehlo.deserialize_portable_artifact(ctx, bytecode)

print(module)
```


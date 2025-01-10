# StableHLO Model Exports

_Exports of popular models in [StableHLO](https://openxla.org/stablehlo)._

## Available Exports

Exports with large literals (weights) elided are available in the
[`exports/`](exports) folder.

**Currently this includes:**
- [AlphaFold](https://github.com/google-deepmind/alphafold)
- [Searchless Chess](https://github.com/google-deepmind/searchless_chess)
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
export PYTHONPATH=$(pwd)/models/gdm_searchless_chess
python models/export.py --models=all
```

A list of models can be specified using the `models` flag if needed:

```py
python models/export.py --models=searchless_chess_9m
```

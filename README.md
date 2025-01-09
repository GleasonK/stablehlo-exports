# StableHLO Exports

_Exports of popular models in StableHLO._

## View Existing Dumps

Dumps with large literals (weights) elided are available in the `dumps/` folder.
StableHLO bytecode files with the weights included will be uploaded somewhere
with large file support shortly.

## Building and Exporting

### Setup Dependencies

I recommend a fresh venv for exporting, this has been tested only using py3.11
but py3.10 is likely to work as well.

```sh
python3 -m venv venv
source venv/bin/activate
./setup.sh
```

### Export Models

To export all models use the `all` flag, this is the default so it can be omit.

```sh
python dump.py --models=all
```

A list of models can be specified using the `models` flag if needed:

```py
python dump.py --models=searchless_chess_9m
```

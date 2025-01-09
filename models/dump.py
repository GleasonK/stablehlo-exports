from absl import app
from absl import flags
from collections.abc import Sequence
import os

import model

import jax
import jax.export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo as stablehlo
from jax._src.lib import xla_client

import searchless_chess_loader

from enum import Enum

class Models(Enum):
  ALL = "all"
  SEARCHLESS_CHESS_9M = "searchless_chess_9m"
  SEARCHLESS_CHESS_136M = "searchless_chess_136m"
  SEARCHLESS_CHESS_270M = "searchless_chess_270m"

flags.DEFINE_list(
    name = "models", 
    default = [Models.ALL.value],
    help="A comma-separated list of models to use. "
    f"Valid models are: {', '.join([model.value for model in Models])}. "
    "You can specify multiple models by separating them with commas (no spaces)."
)
FLAGS = flags.FLAGS

###
# Model loader definitions

def load_model(model):
  print("Loading model:", model)
  if model == Models.SEARCHLESS_CHESS_9M:
    return searchless_chess_loader.load("9M")
  if model == Models.SEARCHLESS_CHESS_136M:
    return searchless_chess_loader.load("136M")
  if model == Models.SEARCHLESS_CHESS_270M:
    return searchless_chess_loader.load("270M")
  raise ValueError(f"Unknown model {model}")

###
# MLIR file writer helpers

def write_bytecode(filename, stablehlo):
  """Write with weights using bytecode to keep data compressed"""
  print("Writing bytecode to:", filename)
  target_version = "1.0.0"
  module_serialized = xla_client._xla.mlir.serialize_portable_artifact(
    stablehlo, target_version)
  with open(filename, 'wb') as f:
    f.write(module_serialized)

def write_readable(filename, stablehlo):
  """Elide large constants when writing human-readable versions"""
  print("Writing to:", filename)
  with jax_mlir.make_ir_context():
    stablehlo_module = ir.Module.parse(stablehlo, context=jax_mlir.make_ir_context())
    stablehlo_pretty = stablehlo_module.operation.get_asm(large_elements_limit=20)
  with open(filename, 'w') as f:
    f.write(stablehlo_pretty)

def write_dump(model, stablehlo):
  """Write output to <repo_root>/dumps"""
  current_file_path = os.path.abspath(__file__)
  models_dir = os.path.dirname(current_file_path)
  dumps_dir = os.path.join(models_dir, "..", "dumps")
  filename = os.path.join(dumps_dir, f"{model.name}.mlir")
  write_readable(filename, stablehlo)
  write_bytecode(filename+".bc", stablehlo)

def dump_stablehlo(model : model.Model):
  print("Dumping model:", model.name)
  exported = jax.export.export(model.main)(*model.inputs)
  write_dump(model, exported.mlir_module())

def dump_models(models):
  for model in models:
    loaded = load_model(model)
    dump_stablehlo(loaded)

###
# Main dump function

def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  
  print(FLAGS.models)
  models = FLAGS.models
  if 'all' in models:
    models = [model for model in Models if model.value != 'all']

  dump_models(models)


if __name__ == '__main__':
  app.run(main)


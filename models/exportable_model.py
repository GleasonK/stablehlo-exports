from dataclasses import dataclass
from enum import Enum


class SourceFramework(Enum):
  JAX = "JAX"
  PYTORCH = "PyTorch"

@dataclass
class ExportableModel:
  """
  A simple struct-like class to hold model data and its associated function.
  """
  source_framework: SourceFramework
  name: str
  main: callable 
  inputs: any
  kwargs: any
  weights_embedded: bool = False

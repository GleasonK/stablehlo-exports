from dataclasses import dataclass

@dataclass
class ExportableModel:
  """
  A simple struct-like class to hold model data and its associated function.
  """
  name: str
  main: callable 
  inputs: any
  kwargs: any

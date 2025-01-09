from dataclasses import dataclass

@dataclass
class Model:
  """
  A simple struct-like class to hold model data and its associated function.
  """
  name: str
  main: callable 
  inputs: any

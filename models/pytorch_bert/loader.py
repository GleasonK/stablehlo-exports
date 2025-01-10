from transformers import AutoTokenizer, BertModel
import torch
import torch_xla2.export

from exportable_model import ExportableModel, SourceFramework

def load():
  tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
  model = BertModel.from_pretrained("google-bert/bert-base-uncased")
  inputs = tokenizer("Hi, my name is", return_tensors="pt")
  return ExportableModel(
    source_framework=SourceFramework.PYTORCH,
    name="pt_bert",
    main=model,
    inputs=(),
    kwargs=inputs,
    weights_embedded=True,
  )

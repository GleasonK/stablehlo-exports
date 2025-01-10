from transformers import AutoImageProcessor, FlaxResNetModel
from PIL import Image
import requests

import jax
from exportable_model import ExportableModel, SourceFramework


def load():
  url = "http://images.cocodataset.org/val2017/000000039769.jpg"
  image = Image.open(requests.get(url, stream=True).raw)
  image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
  resnet = FlaxResNetModel.from_pretrained("microsoft/resnet-50")
  input_dict = image_processor(images=image, return_tensors="np")
  return ExportableModel(
    source_framework=SourceFramework.JAX,
    name="jax_resnet_50",
    main=jax.jit(resnet),
    inputs=(),
    kwargs=input_dict,
    weights_embedded=True,
  )

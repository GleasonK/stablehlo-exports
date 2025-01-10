# GDM AlphaFold

To keep this setup / maintenance trivial, this export hacks on top of the
AlphaFold Colab shared in the
[AlphaFold GitHub README](https://github.com/google-deepmind/alphafold)

## Exporting the Model

The following core functions of the model are fully graph captured via JIT ([model][model]):

```python
self.apply = jax.jit(hk.transform(_forward_fn).apply)
self.init = jax.jit(hk.transform(_forward_fn).init)
```

We can aim to export these functions by creating a function with mimics
[`predict`][predict] in a fully graph capturable way:

[model]:https://github.com/google-deepmind/alphafold/blob/6350ddd63b3e3f993c7f23b5ce89eb4726fa49e8/alphafold/model/model.py#L89-L90
[predict]:https://github.com/google-deepmind/alphafold/blob/6350ddd63b3e3f993c7f23b5ce89eb4726fa49e8/alphafold/model/model.py#L167
[colab]: https://colab.research.google.com/github/deepmind/alphafold/blob/main/notebooks/AlphaFold.ipynb

### Export Using the Colab

1. Navigate to the AlphaFold GitHub Colab: **[AlphaFold Colab][colab]**.
2. Run the Colab (this will take some time, see note below).
3. Add the following code block to the end to export:

```python
import jax
import jax.export

@jax.jit
def alphafold_predict(feat, random_seed):
  model_runner.init_params(feat)
  return model_runner.apply(model_runner.params, random_seed, feat)

random_seed = jax.random.PRNGKey(random.randrange(sys.maxsize))
jax.export.export(alphafold_predict)(processed_feature_dict, random_seed)
```

### Minor Optimization to Export

Most of the notebook needs to be run, as most of it is initialization of the
alphafold model and its parameters.

The only exception if you wish to expedite the export is that the actual AF prediction does not need to run, can edit the prediction cell to:

```python
...
params = data.get_model_haiku_params(model_name, './alphafold/data')
model_runner = model.RunModel(cfg, params)
processed_feature_dict = model_runner.process_features(np_example, random_seed=0)
raise Exception("Exiting early since we don't need to predict / plot.")
# prediction = model_runner.predict(processed_feature_dict, random_seed=random.randrange(sys.maxsize))
```
from absl import app
from absl import flags
from collections.abc import Sequence
import functools
import io
import numpy as np
import os
import pandas as pd

import chess
import chess
import chess.engine
import chess.engine
import chess.pgn
import chess.pgn

from jax import random as jrandom
import jax
import jax.export

from searchless_chess.src import tokenizer
from searchless_chess.src import training_utils
from searchless_chess.src import transformer
from searchless_chess.src import utils
from searchless_chess.src.engines import engine as engine_lib
from searchless_chess.src.engines import neural_engines

import model

def _build_neural_engine_predictor(
    model_name: str,
    checkpoint_step: int = -1,
) -> neural_engines.NeuralEngine:
  """Returns a neural engine."""

  match model_name:
    case '9M':
      policy = 'action_value'
      num_layers = 8
      embedding_dim = 256
      num_heads = 8
    case '136M':
      policy = 'action_value'
      num_layers = 8
      embedding_dim = 1024
      num_heads = 8
    case '270M':
      policy = 'action_value'
      num_layers = 16
      embedding_dim = 1024
      num_heads = 8
    case 'local':
      policy = 'action_value'
      num_layers = 4
      embedding_dim = 64
      num_heads = 4
    case _:
      raise ValueError(f'Unknown model: {model_name}')

  num_return_buckets = 128

  match policy:
    case 'action_value':
      output_size = num_return_buckets
    case 'behavioral_cloning':
      output_size = utils.NUM_ACTIONS
    case 'state_value':
      output_size = num_return_buckets

  predictor_config = transformer.TransformerConfig(
      vocab_size=utils.NUM_ACTIONS,
      output_size=output_size,
      pos_encodings=transformer.PositionalEncodings.LEARNED,
      max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
      num_heads=num_heads,
      num_layers=num_layers,
      embedding_dim=embedding_dim,
      apply_post_ln=True,
      apply_qk_layernorm=False,
      use_causal_mask=False,
  )

  predictor = transformer.build_transformer_predictor(config=predictor_config)
  checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    "searchless_chess", "checkpoints", model_name
  )
  params = training_utils.load_parameters(
      checkpoint_dir=checkpoint_dir,
      params=predictor.initial_params(
          rng=jrandom.PRNGKey(1),
          targets=np.ones((1, 1), dtype=np.uint32),
      ),
      step=checkpoint_step,
  )
  _, return_buckets_values = utils.get_uniform_buckets_edges_values(
      num_return_buckets
  )
  return (predictor, params, return_buckets_values)

PREDICTOR_BUILDERS = {
    'local': functools.partial(_build_neural_engine_predictor, model_name='local'),
    '9M': functools.partial(
        _build_neural_engine_predictor, model_name='9M', checkpoint_step=6_400_000
    ),
    '136M': functools.partial(
        _build_neural_engine_predictor, model_name='136M', checkpoint_step=6_400_000
    ),
    '270M': functools.partial(
        _build_neural_engine_predictor, model_name='270M', checkpoint_step=6_400_000
    )
}

# --------

def get_analysis_sequences(board: chess.Board):
  """Returns buckets log-probs for each action, and FEN."""
  # Tokenize the legal actions.
  sorted_legal_moves = engine_lib.get_ordered_legal_moves(board)
  legal_actions = [utils.MOVE_TO_ACTION[x.uci()] for x in sorted_legal_moves]
  legal_actions = np.array(legal_actions, dtype=np.int32)
  legal_actions = np.expand_dims(legal_actions, axis=-1)
  # Tokenize the return buckets.
  dummy_return_buckets = np.zeros((len(legal_actions), 1), dtype=np.int32)
  # Tokenize the board.
  tokenized_fen = tokenizer.tokenize(board.fen()).astype(np.int32)
  sequences = np.stack([tokenized_fen] * len(legal_actions))
  # Create the sequences.
  sequences = np.concatenate(
      [sequences, legal_actions, dummy_return_buckets],
      axis=1,
  )
  return sequences

def load_sample_board():
  """Load a single sample puzzle board"""
  pgn = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. Nc3 Be7 5. O-O O-O 6. h3 h6 7. d4 exd4 8. Nxd4 Nxd4 9. Qxd4 d6 10. f4 Be6 11. Nd5 Nxd5 12. exd5 Bf5 13. g4 Bh7 14. Qd2 Qd7 15. Qg2 Rae8 16. a4 a6 17. Ra3 Bh4 18. b4 b5 19. axb5 axb5 20. Bd3 Bxd3 21. Rxd3 Re7 22. g5 hxg5 23. fxg5 Rfe8 24. Bd2 Re2 25. Qf3 Qe7 26. Qh5"
  game = chess.pgn.read_game(io.StringIO(pgn))
  if game is None:
    raise ValueError(f'Failed to read game from PGN {puzzle["PGN"]}.')
  board = game.end().board()
  return board

def sample_sequence():
  board = load_sample_board()
  sequences = get_analysis_sequences(board)
  return sequences

def load(agent):
  if not agent in PREDICTOR_BUILDERS:
    raise InputError(f"Invalid searchless chess agent: {agent}")

  predictor, params, return_buckets_values = PREDICTOR_BUILDERS[agent]()
  jitted_predict_fn = jax.jit(predictor.predict)

  ## Note: Weights can be closed over to capture in the StableHLO output
  #  if desired, just remove the `params` argument and update the sample inputs
  @jax.jit
  def predict_sequence(params, sequences):
    return jitted_predict_fn(params=params, targets=sequences, rng=None)

  inputs = (params, sample_sequence())
  return model.Model(
    name=f"searchless_chess_{agent.lower()}",
    main=predict_sequence,
    inputs=inputs
  )

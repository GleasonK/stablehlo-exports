#!/bin/bash

set -ex

setup_searchless_chess() {
  cd gdm_searchless_chess/searchless_chess
  pip install -r requirements.txt
  export PYTHONPATH=$(pwd)/..
  
  # Download data (test only, not train)
  cd data
  wget https://storage.googleapis.com/searchless_chess/data/eco_openings.pgn
  wget https://storage.googleapis.com/searchless_chess/data/puzzles.csv
  mkdir -p test
  cd test
  wget https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag
  wget https://storage.googleapis.com/searchless_chess/data/test/behavioral_cloning_data.bag
  wget https://storage.googleapis.com/searchless_chess/data/test/state_value_data.bag
  cd ../..

  ## Download checkpoints
  cd checkpoints
  ./download.sh

  cd ../..
}

setup_searchless_chess


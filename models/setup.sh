#!/bin/bash

set -ex

SETUP_DIR=$(dirname "$0")
cd $SETUP_DIR

setup_searchless_chess() {
  cd gdm_searchless_chess/searchless_chess
  pip install -r requirements.txt
  
  # Download data (test only, not train)
  cd data
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


#! /bin/bash

if ! [[ -d  ~/Private ]]; then
  echo "The encrypted ~/Private folder is not available, run fhsecTmpFolder"
fi

mkdir -p ~/Private/retain/
time python3 retain_evaluation_sec.py --path_model ~/Private/retain/Model/weights.01.hdf5 --path_data data/data_test.pkl --path_target data/target_test.pkl

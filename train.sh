#! /bin/bash

if ! [[ -d  ~/Private ]]; then
  echo "The encrypted ~/Private folder is not available, run fhsecTmpFolder"
fi

mkdir -p ~/Private/retain/Model
time python3 retain_train_sec.py --directory ~/Private/retain/Model --num_codes=942


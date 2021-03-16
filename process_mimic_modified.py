# This script processes MIMIC-III dataset and builds longitudinal diagnosis records for patients with at 2+ visits.
# The output data are 4 pickled pandas dataframes suitable for training RETAIN-Keras
# Originally Written by Edward Choi (mp2893@gatech.edu) https://github.com/mp2893/retain
# Modified by Timothy Rosenflanz (timothy.rosenflanz@optum.com) to work with RETAIN-Keras
# Usage: Put this script to the folder where MIMIC-III CSV files are located. Then execute the below command.
# python process_mimic_modified.py ADMISSIONS.csv DIAGNOSES_ICD.csv PATIENTS.csv <output directory> <train data proportion>

# Output files
# data_train.pkl: Pickled dataframe used for training containing the codes and to_event sequences as specified in the README
# data_test.pkl: Pickled dataframe used for testing containing the codes and to_event sequences as specified in the README
# data_train_3digit.pkl: Pickled dataframe used for training containing the 3 digit codes and to_event sequences as specified in the README
# data_test_3digit.pkl: Pickled dataframe used for testing containing the 3 digit codes and to_event sequences as specified in the README
# target_train.pkl: Pickled dataframe containing target labels for training as specified in the README
# target_test.pkl: Pickled dataframe containing target labels for testing as specified in the README
# dictionary.pkl: Python dictionary that maps string diagnosis codes to integer diagnosis codes.
# dictionary_3digit.pkl: Python dictionary that maps string diagnosis codes to integer 3 digit diagnosis codes.

import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import os

def convert_to_icd9(dx_str):
    if dx_str.startswith('E'):
        if len(dx_str) > 4: return dx_str[:4] + '.' + dx_str[4:]
        else: return dx_str
    else:
        if len(dx_str) > 3: return dx_str[:3] + '.' + dx_str[3:]
        else: return dx_str

def convert_to_3digit_icd9(dx_str):
    if dx_str.startswith('E'):
        if len(dx_str) > 4: return dx_str[:4]
        else: return dx_str
    else:
        if len(dx_str) > 3: return dx_str[:3]
        else: return dx_str

if __name__ == '__main__':
    admission_file = sys.argv[1]
    diagnosis_file = sys.argv[2]
    patients_file = sys.argv[3]
    out_directory = sys.argv[4]
    train_proportion = float(sys.argv[5])

    print('Collecting mortality information')
    pid_dod_map = {}
    infd = open(patients_file, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[1])
        dod_hosp = tokens[5]
        if len(dod_hosp) > 0:
            pid_dod_map[pid] = 1
        else:
            pid_dod_map[pid] = 0
    infd.close()

    print('Building pid-admission mapping, admission-date mapping')
    pid_adm_map = {}
    adm_date_map = {}
    infd = open(admission_file, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[1])
        adm_id = int(tokens[2])
        adm_time = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
        adm_date_map[adm_id] = adm_time
        if pid in pid_adm_map: pid_adm_map[pid].append(adm_id)
        else: pid_adm_map[pid] = [adm_id]
    infd.close()

    print('Building admission-dxList mapping')
    adm_dx_map = {}
    adm_dx_map_3digit = {}
    infd = open(diagnosis_file, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        adm_id = int(tokens[2])
        dx_str = 'D_' + convert_to_icd9(tokens[4][1:-1])
        dx_str_3digit = 'D_' + convert_to_3digit_icd9(tokens[4][1:-1])

        if adm_id in adm_dx_map:
            adm_dx_map[adm_id].append(dx_str)
        else:
            adm_dx_map[adm_id] = [dx_str]

        if adm_id in adm_dx_map_3digit:
            adm_dx_map_3digit[adm_id].append(dx_str_3digit)
        else:
            adm_dx_map_3digit[adm_id] = [dx_str_3digit]
    infd.close()

    print('Building pid-sortedVisits mapping')
    pid_seq_map = {}
    pid_seq_map_3digit = {}
    for pid, adm_id_list in pid_adm_map.items():
        if len(adm_id_list) < 2: continue

        sorted_list = sorted([(adm_date_map[adm_id], adm_dx_map[adm_id]) for adm_id in adm_id_list])
        pid_seq_map[pid] = sorted_list

        sorted_list_3digit = sorted([(adm_date_map[adm_id], adm_dx_map_3digit[adm_id]) for adm_id in adm_id_list])
        pid_seq_map_3digit[pid] = sorted_list_3digit

    print('Building pids, dates, mortality_labels, strSeqs')
    pids = []
    dates = []
    seqs = []
    morts = []
    for pid, visits in pid_seq_map.items():
        pids.append(pid)
        morts.append(pid_dod_map[pid])
        seq = []
        date = []
        for visit in visits:
            date.append(visit[0])
            seq.append(visit[1])
        dates.append(date)
        seqs.append(seq)

    print('Building pids, dates, strSeqs for 3digit ICD9 code')
    seqs_3digit = []
    for pid, visits in pid_seq_map_3digit.items():
        seq = []
        for visit in visits:
            seq.append(visit[1])
        seqs_3digit.append(seq)

    print('Converting strSeqs to intSeqs, and making types')
    types = {}
    new_seqs = []
    for patient in seqs:
        new_patient = []
        for visit in patient:
            new_visit = []
            for code in visit:
                if code in types:
                    new_visit.append(types[code])
                else:
                    types[code] = len(types)
                    new_visit.append(types[code])
            new_patient.append(new_visit)
        new_seqs.append(new_patient)

    print('Converting strSeqs to intSeqs, and making types for 3digit ICD9 code')
    types_3digit = {}
    new_seqs_3digit = []
    for patient in seqs_3digit:
        new_patient = []
        for visit in patient:
            new_visit = []
            for code in set(visit):
                if code in types_3digit:
                    new_visit.append(types_3digit[code])
                else:
                    types_3digit[code] = len(types_3digit)
                    new_visit.append(types_3digit[code])
            new_patient.append(new_visit)
        new_seqs_3digit.append(new_patient)

    print('Making additional modifications to the data')
    #Compute time to today as to_event column
    today = datetime.strptime('2025-01-01', '%Y-%m-%d')
    to_event = [[(today-date).days for date in patient] for patient in dates]
    #Compute time of the day when the person was admitted as the numeric column of size 1
    numerics = [[[date.hour * 60 + date.minute - 720] for date in patient] for patient in dates]
    #Add this feature to dictionary but leave 1 index empty for PADDING
    types['Time of visit'] = len(types)+1
    types_3digit['Time of visit'] = len(types_3digit)+1
    #Compute sorting indicies
    sort_indicies = np.argsort(list(map(len, to_event)))
    #Create the dataframes of data and sort them according to number of visits per patient
    all_data = pd.DataFrame(data={'codes': new_seqs,
                                  'to_event': to_event,
                                  'numerics': numerics}
                           ,columns=['codes', 'to_event', 'numerics'])\
                          .iloc[sort_indicies].reset_index()
    all_data_3digit = pd.DataFrame(data={'codes': new_seqs_3digit,
                                         'to_event': to_event,
                                         'numerics': numerics}
                                  ,columns=['codes', 'to_event', 'numerics'])\
                                 .iloc[sort_indicies].reset_index()
    all_targets = pd.DataFrame(data={'target': morts}
                               ,columns=['target'])\
                              .iloc[sort_indicies].reset_index()
    #Train test split
    data_train,data_test = train_test_split(all_data, train_size=train_proportion, random_state=12345)
    data_train_3digit,data_test_3digit = train_test_split(all_data_3digit, train_size=train_proportion, random_state=12345)
    target_train,target_test = train_test_split(all_targets, train_size=train_proportion, random_state=12345)
    #Reverse Dictionary into index:code format
    types = dict((v,k) for k,v in types.items())
    types_3digit = dict((v,k) for k,v in types_3digit.items())
    #Write out the data
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    data_train.sort_index().to_pickle(out_directory+'/data_train.pkl')
    data_test.sort_index().to_pickle(out_directory+'/data_test.pkl')
    data_train_3digit.sort_index().to_pickle(out_directory+'/data_train_3digit.pkl')
    data_test_3digit.sort_index().to_pickle(out_directory+'/data_test_3digit.pkl')
    target_train.sort_index().to_pickle(out_directory+'/target_train.pkl')
    target_test.sort_index().to_pickle(out_directory+'/target_test.pkl')
    pickle.dump(types, open(out_directory+'/dictionary.pkl', 'wb'), -1)
    pickle.dump(types_3digit, open(out_directory+'/dictionary_3digit.pkl', 'wb'), -1)

"""
Generates a realistic synthetic UCI Chronic Kidney Disease dataset (400 rows).
This is used by get_preprocessor() to fit the StandardScaler and LabelEncoders
that preprocess user input before the Random Forest model runs.

Run once locally, then commit the resulting CSV so Render has it on deploy.
"""
import csv
import os
import random

random.seed(42)

os.makedirs('static/models/chronic_kidney_disease/data', exist_ok=True)

HEADER = [
    'classification', 'age', 'bp', 'sg', 'al', 'su',
    'rbc', 'pc', 'pcc', 'ba',
    'bgr', 'bu', 'sc', 'sod', 'pot',
    'hemo', 'pcv', 'wc', 'rc',
    'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
]

def rand(lo, hi, decimals=1):
    return round(random.uniform(lo, hi), decimals)

def choice(*args):
    return random.choice(args)

def ckd_row():
    """Generates one CKD-positive synthetic patient."""
    return [
        'ckd',
        rand(20, 90, 0),           # age
        rand(60, 180, 0),          # bp  (often elevated)
        rand(1.005, 1.020, 3),     # sg  (low-normal for CKD)
        random.randint(1, 5),      # al  (albumin, often elevated)
        random.randint(0, 5),      # su
        choice('normal', 'abnormal'),   # rbc
        choice('normal', 'abnormal'),   # pc
        choice('present', 'notpresent'),# pcc
        choice('present', 'notpresent'),# ba
        rand(80, 490, 0),          # bgr
        rand(20, 390, 0),          # bu
        rand(0.5, 15.0, 1),        # sc  (often high)
        rand(110, 150, 0),         # sod
        rand(2.5, 8.0, 1),         # pot
        rand(3.5, 17.5, 1),        # hemo (often low)
        random.randint(15, 54),    # pcv
        rand(2200, 26400, 0),      # wc
        rand(1.5, 8.0, 1),         # rc
        choice('yes', 'no'),       # htn  (mostly yes in CKD)
        choice('yes', 'no'),       # dm
        choice('yes', 'no'),       # cad
        choice('good', 'poor'),    # appet (often poor)
        choice('yes', 'no'),       # pe
        choice('yes', 'no'),       # ane
    ]

def notckd_row():
    """Generates one CKD-negative synthetic patient."""
    return [
        'notckd',
        rand(20, 80, 0),           # age
        rand(60, 100, 0),          # bp  (normal range)
        rand(1.015, 1.025, 3),     # sg  (healthier range)
        random.randint(0, 1),      # al
        random.randint(0, 1),      # su
        choice('normal', 'abnormal'),   # rbc
        choice('normal', 'abnormal'),   # pc
        choice('present', 'notpresent'),# pcc
        choice('present', 'notpresent'),# ba
        rand(70, 160, 0),          # bgr  (normal-ish)
        rand(10, 40, 0),           # bu
        rand(0.4, 1.5, 1),         # sc  (normal)
        rand(135, 145, 0),         # sod
        rand(3.5, 5.5, 1),         # pot
        rand(12.0, 18.0, 1),       # hemo (healthy range)
        random.randint(38, 55),    # pcv
        rand(4000, 11000, 0),      # wc
        rand(3.5, 6.5, 1),         # rc
        choice('yes', 'no'),       # htn
        choice('yes', 'no'),       # dm
        'no',                      # cad  (rare in healthy)
        'good',                    # appet
        'no',                      # pe
        'no',                      # ane
    ]

rows = [HEADER]
for _ in range(250):   # 250 CKD  (~62.5 %)
    rows.append(ckd_row())
for _ in range(150):   # 150 not-CKD (~37.5 %)
    rows.append(notckd_row())

# Shuffle data rows (not header)
data_rows = rows[1:]
random.shuffle(data_rows)
rows = [HEADER] + data_rows

path = 'static/models/chronic_kidney_disease/data/processed_kidney_disease.csv'
with open(path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"Written {len(rows) - 1} data rows to {path}")

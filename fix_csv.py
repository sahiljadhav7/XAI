import csv, os

os.makedirs('static/models/chronic_kidney_disease/data', exist_ok=True)

rows = [
    ['classification','age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane'],
    ['ckd',    48, 80, 1.020, 1, 0, 'normal',   'normal',   'notpresent', 'notpresent', 121, 36, 1.2, 140, 4.0, 15.4, 44, 7800, 5.2, 'yes', 'yes', 'no',  'good', 'no',  'no'],
    ['ckd',    60, 70, 1.010, 3, 2, 'abnormal',  'abnormal',  'present',    'present',    200, 50, 2.5, 130, 5.0, 10.0, 25, 4000, 3.0, 'yes', 'yes', 'yes', 'poor', 'yes', 'yes'],
    ['ckd',    55, 80, 1.015, 2, 1, 'normal',   'abnormal',  'present',    'notpresent', 170, 45, 2.0, 135, 4.5, 12.0, 35, 6000, 4.2, 'yes', 'no',  'no',  'poor', 'yes', 'no'],
    ['notckd', 40, 80, 1.025, 0, 0, 'normal',   'normal',   'notpresent', 'notpresent', 90,  20, 0.8, 140, 4.0, 15.0, 46, 8000, 5.5, 'no',  'no',  'no',  'good', 'no',  'no'],
    ['notckd', 52, 90, 1.015, 1, 1, 'abnormal',  'abnormal',  'present',    'present',    150, 35, 1.8, 135, 4.0, 13.0, 38, 6500, 4.8, 'no',  'no',  'no',  'poor', 'no',  'no'],
    ['notckd', 35, 70, 1.020, 0, 0, 'normal',   'normal',   'notpresent', 'notpresent', 80,  15, 0.7, 142, 3.8, 16.0, 48, 9000, 5.8, 'no',  'no',  'no',  'good', 'no',  'no'],
]

path = 'static/models/chronic_kidney_disease/data/processed_kidney_disease.csv'
with open(path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"Written {len(rows)-1} data rows to {path}")

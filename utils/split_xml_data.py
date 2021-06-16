import os
from sklearn.model_selection import train_test_split

base_dir = '/mnt/Data2/Preethi/XRay_Report_Generation/Data/xml_data/'
base_op_dir = '/mnt/Data2/Preethi/XRay_Report_Generation/Data/xml_split_data/'


xml_files = os.listdir(base_dir)
print(len(xml_files))

train_files, test_files = train_test_split(xml_files, test_size = 250, shuffle = True, random_state = 666)

train_files, val_files = train_test_split(train_files, test_size = 250, shuffle = True, random_state = 666)

with open(base_op_dir+'train_list.txt', 'w') as f:
    f.write(train_files[0])
    for i in range(1,len(train_files)):
        f.write('\n')
        f.write(train_files[i])

with open(base_op_dir+'val_list.txt', 'w') as f:
    f.write(val_files[0])
    for i in range(1,len(val_files)):
        f.write('\n')
        f.write(val_files[i])

with open(base_op_dir+'test_list.txt', 'w') as f:
    f.write(test_files[0])
    for i in range(1,len(test_files)):
        f.write('\n')
        f.write(test_files[i])
import xml.etree.ElementTree as ET 
import os
import numpy as np

tag_list=[]
tag_count = []
base_dir = 'xml_data_orig/'
for file in os.listdir(base_dir):
    tree = ET.parse(base_dir+file)
    # print(tree) 
    root = tree.getroot() 
    # print(root)
    
    for item in root.findall('./MeSH'): 
        for tags in item.findall('./automatic'):
            current_tag = tags.text
            # current_tag = current_tag.split('/')
            # for tt in current_tag:
                # tt1 = tt.lower()
            current_tag = current_tag.lower()
            # if current_tag == 'gastric dilatation':
                # print(file)
                # exit()
            if current_tag not in tag_list:
                tag_list.append(current_tag)
                tag_count.append(1)
            else:
                current_index_2 = tag_list.index(current_tag)
                tag_count[current_index_2] += 1
                # print(current_tag)
print(len(tag_list))
count = 0
remove_indices = []
for i in range(len(tag_list)):
    if tag_count[i] < 3:
        count += 1
        remove_indices.append(i)
for index in sorted(remove_indices, reverse = True):
    del tag_list[index]
print(count)
print(len(tag_list))
# exit()

tag_list.sort()
# tag_count = np.zeros((len(tag_list),),dtype = np.int32)
# for file in os.listdir(base_dir):
#     tree = ET.parse(base_dir+file)
#     root = tree.getroot()
#     for images in root.findall('./parentImage'):
#         image_name = images.get('id')
#         for tags in root.findall('./MeSH/automatic'):
#             current_tag = tags.text
#             current_tag = current_tag.lower()
#             current_index = tag_list.index(current_tag)
#             tag_count[current_index] += 1

# count = 0
# for i in range(len(tag_list)):
#     if tag_count[i] <= 10:
#         count += 1
#         print(tag_list[i])
# print(count)
# exit()


# print(tag_list)
# print(len(tag_list))
# exit()
with open('tags_automatic_trim.txt','w') as f:
    goodbye = 1
tag_count = np.zeros((len(tag_list)+1,),dtype = np.int32)
print(len(tag_list))
count = 0
for file in os.listdir(base_dir):
    tree = ET.parse(base_dir+file)
    # print(tree) 
    # exit()
    root = tree.getroot()
    for images in root.findall('./parentImage'):
        image_name = images.get('id')
        current_tags = np.zeros((len(tag_list)+1,),dtype=np.int16)
        # print(current_tags.shape)
        # exit()
        temp_list = []
        for tags in root.findall('./MeSH/automatic'):
            # print('here')
            # exit()
            current_tag = tags.text
            # current_tag = current_tag.split('/')
            # for tt in current_tag:
                # tt1 = tt.lower()
                # current_index = tag_list.index(tt1)
                # temp_list.append(current_index)
                # current_tags[current_index] = 1
                # tag_count[current_index]+=1
            current_tag = current_tag.lower()
            if current_tag in tag_list:
                current_index = tag_list.index(current_tag)
                tag_count[current_index]+=1
                temp_list.append(current_index)
                current_tags[current_index] = 1
        if len(temp_list)==0:
            # print(image_name)
            current_tags[-1] = 1
            tag_count[-1]+=1
            # count +=1
        with open('tags_automatic_trim.txt','a') as f:
            f.write(image_name)
            for i in range(len(tag_list)+1):
                f.write(' '+str(current_tags[i]))
            f.write('\n')
# # print(count)
# tag_list.append('normal')
# top_tage = np.argsort(tag_count)
# print(top_tage.dtype)
# sorted_tag_list = []
# for i in range(top_tage.shape[0]):
#     sorted_tag_list.append(tag_list[top_tage[i]])
# # sorted_tag_list = tag_list[indices]
# sorted_tag_count = tag_count[top_tage]

# sum_count = float(np.sum(sorted_tag_count))
# norm_sorted_tag_count = sorted_tag_count/sum_count

# # for i in range(len(tag_list)):
# #     print(sorted_tag_list[i]+'\t'+str(sorted_tag_count[i]))

# i = len(tag_list)-1
# cummu = 0.0
# with open('top_tag_list_automatic.txt','w') as f:
#     while i >= 0:
#         cummu = cummu + norm_sorted_tag_count[i]
#         f.write(str(top_tage[i]))
#         f.write('\t')
#         f.write(sorted_tag_list[i])
#         f.write('\t')
#         f.write(str(sorted_tag_count[i]))

#         f.write('\t')
#         f.write(str(cummu))
#         f.write('\n')
        
#         i = i-1

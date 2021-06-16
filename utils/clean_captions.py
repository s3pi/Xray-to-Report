import os
import json
import xml.etree.ElementTree as ET

base_dir = '/mnt/Data2/Preethi/XRay_Report_Generation/Data/xml_data/'
a = str()
b = str()
special = []
caption_dict = {}
for file in os.listdir(base_dir):
    count = 0
    tree = ET.parse(os.path.join(base_dir,file))
    root = tree.getroot()
    # print(file)
    for item in root.findall('./MedlineCitation/Article/Abstract'): 
        for items in item.findall('./AbstractText'):
            if items.get('Label') == 'FINDINGS':
                a = items.text
            if items.get('Label') == 'IMPRESSION':
                b = items.text
    # # print(a)
    # if (a is not None) and (b is not None):
    #     a = a.lower()
    #     b = b.lower()
    #     c = a+' '+b
    #     count = 1
    # elif (a is not None) and (b is None):
    #     a = a.lower()
    #     c = a
    #     count = 1
    # elif (a is None) and (b is not None):
    #     b = b.lower()
    #     c = b
    #     count = 1
    # else:
    #     count = 0


    if (b is not None) and (a is not None):
        b = a.lower()
        c = b
        count = 1
    else:
        count = 0
    
    if count == 1:
        c = c.replace('..','.').replace('. .','.').replace('.  ','. ')
        c = c.replace('-',' ')
        c = c.replace('/',' / ')
        c = c.replace('[&lt;t','t')
        c = c.replace('.&gt;]','.')
        c = c.replace('[<h','h')
        c = c.replace('[p','p')
        c = c.replace('>].','.')
        c = c.replace('<br>',' ')
        c = c.replace('<',' ')
        c = c.replace('>',' ')
        c = c.replace(',',' ,')
        c = c.replace(':',' :')
        c = c.replace(';', ' ;')
        c = c.replace('?',' ?')
        c = c.replace('body(lateral','body (lateral')
        c = c.replace('(','( ')
        c = c.replace(')','')
        # fstop tokens . , : ; ?
        c = c.replace('.',' .')
        c = c.replace('.','<fstop>')
        c = c.replace(',','<fstop>')
        c = c.replace(':','<fstop>')
        c = c.replace(';','<fstop>')
        c = c.replace('?','<fstop>')
        # alt token ( ) /
        c = c.replace('(','<alt>')
        c = c.replace('/','<alt>')
        # num token
        c = c.replace('0','<num>')
        c = c.replace('1','<num>')
        c = c.replace('2','<num>')
        c = c.replace('3','<num>')
        c = c.replace('4','<num>')
        c = c.replace('5','<num>')
        c = c.replace('6','<num>')
        c = c.replace('7','<num>')
        c = c.replace('8','<num>')
        c = c.replace('9','<num>')
        c = c.replace('%','<num>')
        c = c.replace('<fstop> <num> <fstop>','<fstop>') # removing lists indexes
        c = c.replace('<num> <fstop><num>','<num>') # removing decimal point numbers
        c = c.replace('cm ','<num> ')
        c = c.replace('mm ','<num> ')
        c = c.replace('<num> <num>', '<num>')
        c = c.replace('<num> x <num>','<num>')
        c = c.replace('<num><num><num>','<num>')
        c = c.replace('<num><num>','<num>')
        c = c.replace('<num>th and <num>th','<num>')
        c = c.replace('<num>th <fstop> <num>th <fstop> and <num>th','<num>')
        c = c.replace('<num>th','<num>')
        c = c.replace('<num>st','<num>')
        c = c.replace('<num>nd','<num>')
        c = c.replace('<num>rd','<num>')
        c = c.replace('<num>x','<num> x')
        c = c.replace('<num> <num>','<num>')
        # unk token for xxxx
        c = c.replace('xxxx','<unk>')
        c = c.replace('x <unk>','<unk>')
        c = c.replace('<unk> <unk>', '<unk>')
        c = c.replace('<unk> <unk>', '<unk>')
        c = c.replace('<unk> <unk>', '<unk>')
        c = c.replace('<fstop> <unk> <fstop>','<fstop>')
        # pos token l<num> and t<num>
        c = c.replace('l<num>','<pos>')
        c = c.replace('t<num>','<pos>')
        c = c.replace('<pos> <pos>','<pos>')
        c = c.replace('<pos> <fstop> <pos> <fstop> <pos> and <pos> <fstop>','<pos> <fstop>')
        for i in range(len(c)):
            if (ord(c[i]) < 97 or ord(c[i]) > 122) and (ord(c[i])!=32) and (ord(c[i])!=44) and (ord(c[i])!=46):
                # if c[i] == ';':
                #     print(file)
                #     # exit()
                if c[i] not in special:
                    special.append(c[i])
            # exit()
        caption_dict[file] = c
# print(caption_dict)

with open('findings_dict.json','w') as fp:
    json.dump(caption_dict, fp)
print(special)
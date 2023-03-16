# ---
# jupyter:
#   jupytext:
#     formats: notebooks///ipynb,notebooks/py///py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python [conda env:pytorch]
#     language: python
#     name: pytorch
# ---

# +
import time
import os
from tqdm import tqdm

import pandas as pd
import numpy as np
import cv2
import seaborn as sns

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
# %matplotlib inline

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('device', device)

# +
#from wordcloud import WordCloud
from collections import defaultdict
from collections import Counter

import itertools

import re

from wordcloud import WordCloud
# -

import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
matplotlib.rc("font",family='SimHei') # 中文字体
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

from torchvision import transforms

import xml.etree.ElementTree as ET

# 数据集文件夹路径
directory = '../data/NLMCXR/ecgen-radiology'
image_dir='../data/NLMCXR/NLMCXR_png/'

# extracting data from the xml documents
img = []
img_impression = []
img_finding = []
for filename in tqdm(os.listdir(directory)):
    if filename.endswith(".xml"):
        f = directory + '/' + filename
        tree = ET.parse(f)
        root = tree.getroot()
        for child in root:
            if child.tag == 'MedlineCitation':
                for attr in child:
                    if attr.tag == 'Article':
                        for i in attr:
                            if i.tag == 'Abstract':
                                for name in i:
                                    if name.get('Label') == 'FINDINGS':
                                        finding=name.text
                                        
        for p_image in root.findall('parentImage'):
            
            img.append(p_image.get('id'))
            img_finding.append(finding)

dataset = pd.DataFrame()
dataset['Image_path'] = img
dataset['Finding'] = img_finding
dataset.head(10)

print('Dataset Shape:', dataset.shape)


def absolute_path(x):
    '''Makes the path absolute '''
    x =image_dir+ x + '.png'
    return x
dataset['Image_path'] = dataset['Image_path'].apply(lambda x : absolute_path(x)) # making the paths absolute

dataset.head(10)


def image_desc_plotter(data, n, rep):  
    count = 1  
    fig = plt.figure(figsize=(10,20))

    if rep == 'finding':
        
        for filename in data['Image_path'].values[95:100]:   
        
            findings = list(data["Finding"].loc[data["Image_path"] == filename].values) 
            img = cv2.imread(filename)    
            ax = fig.add_subplot(n, 2 , count , xticks=[], yticks=[])  
            ax.imshow(img)     
            count += 1            
            ax = fig.add_subplot(n ,2 ,count)   
            plt.axis('off')     
            ax.plot()     
            ax.set_xlim(0,1)    
            ax.set_ylim(0, len(findings))  
            for i, f in enumerate(findings):   
                ax.text(0,i,f,fontsize=20)   
            count += 1 
        plt.show()
        
    else:
        print("Enter a valid String")


image_desc_plotter(dataset, 5, 'finding')

# loading the heights and widths of each image
h = []
w = []
for i in tqdm(np.unique(dataset['Image_path'].values)):
    img = cv2.imread(i)
    h.append(img.shape[0])
    w.append(img.shape[0])

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.title('Height Plot')
plt.ylabel('Heights')
plt.xlabel('--Images--')
sns.scatterplot(x=range(len(h)), y=h)#此处不添加xy会报错：scatterplot() takes from 0 to 1 positional arguments but 2 were given
plt.subplot(122)
plt.title('Width Plot')
plt.ylabel('Widths')
plt.xlabel('--Images--')
sns.scatterplot(x=range(len(w)), y=h)

# Images have different heights and widths, they will be resized into a common shape

# +
print('Number of Images:', dataset['Image_path'].nunique())

# number of missing values
print(dataset.isnull().sum())
print("There are a total of  ",dataset.isnull().sum()[1]," rows where 'findings' column has no value")
# -

dataset = dataset.dropna(axis=0) # drop all missing value rows

dataset.isnull().sum()

print('New Shape of the Data:', dataset.shape)

dataset.head(12)

plt.figure(figsize=(14,7))
plt.subplot(131)
img1 = cv2.imread(dataset['Image_path'].values[6])
plt.imshow(img1)
plt.title(dataset['Image_path'].values[6])
plt.subplot(132)
img2 = cv2.imread(dataset['Image_path'].values[7])
plt.title(dataset['Image_path'].values[7])
plt.imshow(img2)
plt.subplot(133)
img3 = cv2.imread(dataset['Image_path'].values[8])
plt.title(dataset['Image_path'].values[8])
plt.imshow(img3)

dataset['Finding'].values[6], dataset['Finding'].values[7], dataset['Finding'].values[8]

# The dataset consists of multiple chest shots of the same person. The images of a person has the same file name except the last 4 digits. Therefore that can be taken as the person ID.
# 该数据集由同一个人的多个胸部照片组成。一个人的图像具有相同的文件名，除了最后4位数字。因此，这可以作为人的ID。

# This creates 2 dictionaries with keys as the person id and the number of images and findings for that person. 
images = {}
findings = {}
for img, fin in dataset.values:
    a = img.split('-')
    a.pop(len(a)-1)
    a = '-'.join(e for e in a)
    if a not in images.keys():
        images[a] = 1
        findings[a] = fin
    else:
        images[a] += 1
        findings[a] = fin

images[image_dir+'CXR1001_IM-0004'], findings[image_dir+'CXR1001_IM-0004']

images

print('Total Number of Unique_IDs :', len(images.keys()))

plt.figure(figsize=(17,5))
plt.bar(range(len(images.keys())), images.values())
plt.ylabel('Total Images per individual')
plt.xlabel('Number of Individuals in the Data')

one = 0
two = 0
three = 0
four = 0
for v in images.values():
    if v == 1:
        one +=1
    elif v == 2:
        two += 1
    elif v == 3:
        three += 1
    elif v == 4:
        four += 1
    else:
        print('Error')
one, two, three, four


# The above variables one, two, three, four contains the total number of IDs with 1,2,3,4 number of images respectively.
#
# As we can see there are multiple images corresponding to a single person. These are different chest scans at different views. Most of the individuals have only 2 scans while the highest being 4.

#len(images)
def train_test_split(data):
    persons = list(data.keys())
    persons_train = persons[:2500]
    persons_cv = persons[2500:3000]
    persons_test = persons[3000:3350]
    return persons_train, persons_cv, persons_test


images_train, images_cv, images_test = train_test_split(images)


def combining_images(image_set):
    
    image_per_person = defaultdict(list)  # creating a list of dictionary to store all the image paths
                                            #corresponding to a person_id
    for pid in image_set:
        for img in dataset['Image_path'].values:
            if pid in img:
                image_per_person[pid].append(img)
            else:
                continue
    return image_per_person


img_per_person_train = combining_images(images_train)
img_per_person_cv = combining_images(images_cv)
img_per_person_test = combining_images(images_test)

len(img_per_person_train), len(images_train)

img_per_person_train[image_dir+'CXR1001_IM-0004']

# +
# def load_image(file):
#     img = tf.io.read_file(file)
#     img = tf.image.decode_png(img, channels=3)
#     img = tf.image.convert_image_dtype(img, tf.float32)
#     return img
# -

# just checking the ID which has 4 images
for k,v in images.items():
    if v == 4:
        print(k)
        break


# +
# plt.figure(figsize=(9,9))
# plt.subplot(221)
# plt.imshow(load_image('Scanned Images/CXR1102_IM-0069-12012.png'))
# plt.title('Scanned Images/CXR1102_IM-0069-12012.png')
# plt.subplot(222)
# plt.imshow(load_image('Scanned Images/CXR1102_IM-0069-2001.png'))
# plt.title('Scanned Images/CXR1102_IM-0069-2001.png')
# plt.subplot(223)
# plt.imshow(load_image('Scanned Images/CXR1102_IM-0069-3001.png'))
# plt.title('Scanned Images/CXR1102_IM-0069-3001.png')
# plt.subplot(224)
# plt.imshow(load_image('Scanned Images/CXR1102_IM-0069-4004.png'))
# plt.title('Scanned Images/CXR1102_IM-0069-4004.png')
# -

# 2 side view and 2 front view images for the same ID
#
# Sample chest scans of a person(4 images)
#
# Now, we have multiple chest scans to produce a single report. Some person_ids have 1, some have 2 and the highest is 4. So we can take pairs of those images as input. If it has only one image, then it can be replicated.

def create_data(image_per_person):
    # new dataset
    person_id, image1, image2, report = [],[],[],[]
    for pid, imgs in image_per_person.items():   #contains pid and the images associated with that pid

        if len(imgs) == 1:
            image1.append(imgs[0])
            image2.append(imgs[0])
            person_id.append(pid)
            report.append(findings[pid])
        else:
            num = 0
            a = itertools.combinations(imgs, 2)
            for i in a:
                image1.append(i[0])
                image2.append(i[1])
                person_id.append(pid + '_' + str(num))
                report.append(findings[pid])
                num += 1
    data = pd.DataFrame()
    data['Person_id'] = person_id
    data['Image1'] = image1
    data['Image2'] = image2
    data['Report'] = report
    
    return data


train = create_data(img_per_person_train)
test = create_data(img_per_person_test)
cv = create_data(img_per_person_cv)

train.head()


# # Text Cleaning

# +
def lowercase(text):
    '''Converts to lowercase'''
    new_text = []
    for line in text:
        new_text.append(line.lower())
    return new_text

def decontractions(text):
    '''Performs decontractions in the doc'''
    new_text = []
    for phrase in text:
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"couldn\'t", "could not", phrase)
        phrase = re.sub(r"shouldn\'t", "should not", phrase)
        phrase = re.sub(r"wouldn\'t", "would not", phrase)
        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        phrase = re.sub(r"\*+", "abuse", phrase)
        new_text.append(phrase)

    return new_text

def rem_punctuations(text):
    '''Removes punctuations'''
    punctuations = '''!()-[]{};:'"\,<>/?@#$%^&*~''' # full stop is not removed
    new_text = []
    for line in text:
        for char in line:
            if char in punctuations: 
                line = line.replace(char, "")
        new_text.append(' '.join(e for e in line.split()))
    return new_text

def rem_numbers(text):
    '''Removes numbers and irrelevant text like xxxx*'''
    new_text = []
    for line in text:
        temp = re.sub(r'x*','',line)
        new_text.append(re.sub(r'\d','',temp))
    return new_text

def words_filter(text):
    '''Removes words less than 2 characters except no and ct'''
    new_text = []
    for line in text:
        temp = line.split()
        temp2 = []
        for word in temp:
            if  len(word) <=2 and word != 'no' and word != 'ct':
                continue
            else:
                temp2.append(word)
        new_text.append(' '.join(e for e in temp2))
    return new_text

def multiple_fullstops(text):
    ''' Removes multiple full stops from the text'''
    new_text = []
    for line in text:
        new_text.append(re.sub(r'\.\.+', '.', line))
    return new_text

def fullstops(text):
    new_text = []
    for line in text:
        new_text.append(re.sub('\.', ' .', line))
    return new_text

def multiple_spaces(text):
    new_text = []
    for line in text:
        new_text.append(' '.join(e for e in line.split()))
    return new_text

def separting_startg_words(text):
    new_text = []
    for line in text:
        temp = []
        words = line.split()
        for i in words:
            if i.startswith('.') == False:
                temp.append(i)
            else:
                w = i.replace('.','. ')
                temp.append(w)
        new_text.append(' '.join(e for e in temp))
    return new_text

def rem_apostrophes(text):
    new_text = []
    for line in text:
        new_text.append(re.sub("'",'',line))
    return new_text


# -

def text_preprocessing(text):
    '''Combines all the preprocess functions'''
    new_text = lowercase(text)
    new_text = decontractions(new_text)
    new_text = rem_punctuations(new_text)
    new_text = rem_numbers(new_text)
    new_text = words_filter(new_text)
    new_text = multiple_fullstops(new_text)
    new_text = fullstops(new_text)
    new_text = multiple_spaces(new_text)
    new_text = separting_startg_words(new_text)
    new_text = rem_apostrophes(new_text)
    return new_text


train['Report'] = text_preprocessing(train['Report'])
test['Report'] = text_preprocessing(test['Report'])
cv['Report'] = text_preprocessing(cv['Report'])

length = [len(e.split()) for e in train['Report'].values]# Number of words in each report
max(length)

plt.title('Number of Words per Report')
sns.scatterplot(x=range(train.shape[0]), y=length)
plt.ylabel('Number of words')

# +
l = []
for i in train['Report'].values:
    l.extend(i.split())

c = Counter(l)
# -

words = []
count = []
for k,v in c.items():
    words.append(k)
    count.append(v)
words_count = list(zip(count, words))

top_50_words = sorted(words_count)[::-1][:50]
bottom_50_words = sorted(words_count)[:50]

plt.figure(figsize=(15,5))
plt.bar(range(50), [c for c,w in top_50_words])
plt.title('Top 50 Most Occuring Words')
plt.xlabel('Words')
plt.ylabel('Count')
plt.xticks(ticks=range(50), labels=[w for c,w in top_50_words], rotation=90)

plt.figure(figsize=(15,5))
plt.bar(range(50), [c for c,w in bottom_50_words])
plt.title('Top 50 Least Occuring Words')
plt.xlabel('Words')
plt.ylabel('Count')
plt.xticks(ticks=range(50), labels=[w for c,w in bottom_50_words], rotation=90)

# +
w = WordCloud(height=1500, width=1500).generate(str(l))

plt.figure(figsize=(12,12))
plt.title('WordCloud of Reports')
plt.imshow(w)


# -

def remodelling(x):
    '''adds start and end tokens to a sentence '''
    return 'startseq' + ' ' + x + ' ' + 'endseq'

# 暂时去除了加入的前后终止符
# train['Report'] = train['Report'].apply(lambda x : remodelling(x))
# test['Report'] = test['Report'].apply(lambda x : remodelling(x))
# cv['Report'] = cv['Report'].apply(lambda x : remodelling(x))

# save the cleaned data(STRUCTURED DATA)
train.to_csv('Train_Data.csv', index=False)
test.to_csv('Test_Data.csv', index=False)
cv.to_csv('CV_Data.csv', index=False)





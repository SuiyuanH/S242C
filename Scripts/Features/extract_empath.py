# HN 用以提取通感（empath）特征，这类特征主要是根据特定词表，计算
import os, json, sys
import numpy as np
import time
import regex as re

import _load

# feature: Empath word categories

from empath import Empath
lexicon = Empath()
# HN 这是一个空表

# add new tailored categories to empath

lexicon.create_category('spatial',['short','long','circular','small','big','large','huge','gigantic','tiny','rectangular','rectangle','massive',
                                  'giant','enormous','smallish','rounded','middle','oval','sized','size','miniature','circle',
                                  'colossal','center','triangular','shape','boxy','round','shaped','dimensioned'])
# 这个词表包含了与空间或几何形状相关的词汇。它可以用于识别文本中描述的物体或概念的形状、大小、长度等特征。


lexicon.create_category('sentiment',['good','bad','nice','annoying','formidable','superb','abysmal','mean','wonderful',
                                    'great','terrible','horrendous','awful','dreadful','neat','fantastic','terrific'])
# 这个词表包含了与情感或情感倾向相关的词汇。它可以用于识别文本的整体情感倾向，例如积极、消极或中立等。

lexicon.create_category('opinion',['think','feel','mean','argue','reason','say','state','mind','believe','suggest','proves',
                                   'although','find','opinion','guess','deem','consider'])
# 这个词表包含了与意见或观点相关的词汇。它可以用于识别文本中表达的个人观点、看法或意见。

lexicon.create_category('logic',['logical','rational','reasonable','justified','reasoned','obvious','coherent','consistent',
                                'coherent','legitimate','valid','plausible'])
# 这个词表包含了与逻辑或推理相关的词汇。它可以用于识别文本中的逻辑关系、推理过程或论证结构。

lexicon.create_category('ethic',['ethical','right','morally','decent','indecent','unethical','honorable','honourable',
                                'value','virtue','honest','legal','legitimate','illegitimate','right','wrong','vice',
                                'norm','immoral'])
# 这个词表包含了与道德或伦理相关的词汇。它可以用于识别文本中的道德准则、价值观或伦理观念。

# take arguments
if len(sys.argv) != 4:
    print('Expecting 3 arguments: model, dataset, split')
    sys.exit(1)

model = sys.argv[1] # model: HC3
size = sys.argv[2] # domain: medicine
split = sys.argv[3] # split: train

################################################################################################################################################

def extract_empath(text):
    # 
    wordlist = [str(x).lower() for x in re.findall(r'\'*[a-zA-Z]\w*',text)]
    empath_ = list(lexicon.analyze(text).values())
    
    return empath_

data_path = os.path.join('.','Data','data','full_text')
save_path = os.path.join('.','Data','features','full_text')

texts = _load._load_split(os.path.join(data_path,model,size),size,split)

empath = []
print('Starting extraction of empath stats...')
        
for i, text in enumerate(texts):
    
    empath.append(extract_empath(text))
            

print('... empath extracted!')
if not os.path.exists(os.path.join(save_path,model,size)):
    os.makedirs(os.path.join(save_path,model,size))
    
np.save(os.path.join(save_path,model,size,'{}_empath_{}.npy'.format(size,split)),empath)
# [None, 5]
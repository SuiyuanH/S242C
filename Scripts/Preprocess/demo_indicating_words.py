import re
import os
import nltk
#nltk.download('punkt')
from nltk import sent_tokenize # for spliting English sentences

src_path = os.path.dirname(os.path.abspath(__file__))

# for spliting Chinese sentences
def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")

def replace_ni(string):
    return string.replace('您','你')

# human indicating words (both english and chinese)
with open(f'{src_path}/indicating_words_en_human.txt', encoding='gbk') as f:
    indicating_words_human_en = [l.rstrip() for l in f]
    
with open(f'{src_path}/indicating_words_zh_human.txt', encoding='gbk') as f:
    indicating_words_human_zh = [l.rstrip() for l in f]

# chatgpt indicating words (both english and chinese)
with open(f'{src_path}/indicating_words_en_chatgpt.txt', encoding='gbk') as f:
    indicating_words_chatgpt_en = [l.rstrip() for l in f]
    
with open(f'{src_path}/indicating_words_zh_chatgpt.txt', encoding='gbk') as f:
    indicating_words_chatgpt_zh = [l.rstrip() for l in f]

def filtering(text, indicating_words, language, verbose=False):
    '''removing sentence(s) that includes indicating words'''
    assert isinstance(text, str)
    assert isinstance(indicating_words, list)
    if language == 'en':
        sents = sent_tokenize(text)
    elif language == 'zh':
        sents = cut_sent(text)
    else:
        raise NotImplementedError
  
    filtered_sents = []
    for s in sents:
        if language == 'zh':
            # replace"您"to"你" for Chinese corpus
            s = replace_ni(s)
        has = False
        for k in indicating_words:
            if k in s:
                has = True
                break
        if not has:
            filtered_sents.append(s)
            
    filtered_sents = ' '.join(filtered_sents)
    
    if verbose:
        print(f'Original answers: {text} \nFiltered answers: {filtered_sents}\n')

    return filtered_sents

def auto_filter (text, is_machine, language='en', verbose=False):
    if language == 'en':
        if is_machine:
            filtered_answers = filtering(text, indicating_words=indicating_words_chatgpt_en, language='en', verbose=verbose)
        else:
            filtered_answers = filtering(text, indicating_words=indicating_words_human_en, language='en', verbose=verbose)
    else:
        if is_machine:
            filtered_answers = filtering(text, indicating_words=indicating_words_chatgpt_zh, language='zh', verbose=verbose)
        else:
            filtered_answers = filtering(text, indicating_words=indicating_words_human_zh, language='zh', verbose=verbose)
    return filtered_answers

if __name__ == '__main__':
    # human answers, en
    answers = "The best answer to this is: Read the fine print on your credit card agreement. What is common, at least in the US, is that you can make any charges you want during a time window."
    filtered_answers = filtering(answers, indicating_words=indicating_words_human_en, language='en', verbose=True)

    # human answers, zh
    answers = "易方达易理财是中国的一家资产管理公司。它是由中国证券投资基金业协会监管的一家合法公司，并且是中国证监会批准的基金管理公司。如果您担心关于易方达易理财的安全性，建议您咨询相关监管机构，或者咨询专业的金融顾问，以确保您的投资是安全的。所在城市若有招商银行，也可以了解下招行发售的理财产品，您可以进入招行主页，点击理财产品-个人理财产品页面查看，也可通过搜索分类您需要的理财产品。温馨提示：购买之前请详细阅读产品说明书。"
    filtered_answers = filtering(answers, indicating_words=indicating_words_human_zh, language='zh', verbose=True)

    # chatgpt answers, en
    answers = "Fundamental analysis is a method of evaluating the intrinsic value of an asset by examining its underlying economic and financial factors. It involves analyzing a company's financial statements, such as its balance sheet, income statement, and cash flow statement, as well as other factors such as its management, competitive advantage, and industry trends.If you're interested in learning fundamental analysis, here are a few steps you can follow:Start by learning the basic concepts and terminology of financial analysis, such as profitability ratios, liquidity ratios, and debt ratios.Learn how to read and interpret financial statements.I hope this helps! Let me know if you have any other questions."
    filtered_answers = filtering(answers, indicating_words=indicating_words_chatgpt_en, language='en', verbose=True)

    # chatgpt answers, zh
    answers = "我不能为你提供任何基于个人经验的神秘主义规律，因为我是一个人工智能助手，没有自己的个人经验。我只能根据我所获得的信息来回答问题，并且我的知识基于 2021 年前的信息，并没有浏览的功能。神秘主义是一种不科学的信仰，它基于超自然的力量或神秘的规律，并且往往忽略了科学知识和方法。尽管有些人喜欢探索神秘主义的理论，但是没有科学依据的主张不能被视为可靠或可信的。"
    filtered_answers = filtering(answers, indicating_words=indicating_words_chatgpt_zh, language='zh', verbose=True)
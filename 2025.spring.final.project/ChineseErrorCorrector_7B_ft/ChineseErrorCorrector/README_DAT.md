#  一键语法错误增强工具

-----------------

欢迎使用一键语法错误增强工具，该工具可以进行14种语法错误的增强，不同行业可以根据自己的数据进行错误替换，来训练自己的语法和拼写模型，荣获2024 CCL 冠军。



## 使用
- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:
``` sh
conda create -n zh_correct -y python=3.9
conda activate zh_correct
pip install ChineseErrorCorrector
# If you are in mainland China, you can set the mirror as follows:
pip install ChineseErrorCorrector -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```



## 介绍

一键语法错误增强工具，支持：
- [1.缺字漏字](#1缺字漏字)
- [2.错别字错误](#2错别字错误)
- [3.缺少标点](#3缺少标点)
- [4.错用标点](#4错用标点)
- [5.主语不明](#5主语不明)
- [6.谓语残缺](#6谓语残缺)
- [7.宾语残缺](#7宾语残缺)
- [8.其他成分残缺](#8其他成分残缺)
- [9.虚词多余](#9虚词多余)
- [10.其他成分多余](#10其他成分多余)
- [11.主语多余](#11主语多余)
- [12.语序不当](#12语序不当)
- [13.动宾搭配不当](#13动宾搭配不当)
- [14.其他搭配不当](#14其他搭配不当)




## 注意

如果没有进行数据增强，则返回None

---
## API



### 1.缺字漏字


```python
from ChineseErrorCorrector.utils.dat import GrammarErrorDat

cged_tool = GrammarErrorDat()
print(cged_tool.lack_word("小明住在北京"))

# 输出：小明在北京

```
### 2.错别字错误


```python
from ChineseErrorCorrector.utils.dat import GrammarErrorDat

cged_tool = GrammarErrorDat()
print(cged_tool.wrong_word("小明住在北京"))
# 输出：小明住在北鲸

```
### 3.缺少标点


```python
from ChineseErrorCorrector.utils.dat import GrammarErrorDat

cged_tool = GrammarErrorDat()
print(cged_tool.lack_char("小明住在北京，热爱NLP。"))
# 输出：小明住在北京热爱NLP。

```
### 4.错用标点


```python
from ChineseErrorCorrector.utils.dat import GrammarErrorDat

cged_tool = GrammarErrorDat()
print(cged_tool.wrong_char("小明住在北京，热爱NLP。"))
# 输出：小明住在北京。热爱NLP。

```
### 5.主语不明


```python
from ChineseErrorCorrector.utils.dat import GrammarErrorDat

cged_tool = GrammarErrorDat()
print(cged_tool.unknow_sub("小明住在北京"))
# 输出：住在北京

```
### 6.谓语残缺


```python
from ChineseErrorCorrector.utils.dat import GrammarErrorDat

cged_tool = GrammarErrorDat()
print(cged_tool.unknow_pred("小明住在北京"))
# 输出：小明在北京
```
### 7.宾语残缺


```python
from ChineseErrorCorrector.utils.dat import GrammarErrorDat

cged_tool = GrammarErrorDat()
print(cged_tool.lack_obj("小明住在北京，热爱NLP。"))
# 输出：小明住在北京，热爱。
```
### 8.其他成分残缺


```python
from ChineseErrorCorrector.utils.dat import GrammarErrorDat

cged_tool = GrammarErrorDat()
print(cged_tool.lack_others("小明住在北京，热爱NLP。"))
# 输出：小明住北京，热爱NLP。
```
### 9.虚词多余

```python
from ChineseErrorCorrector.utils.dat import GrammarErrorDat

cged_tool = GrammarErrorDat()
print(cged_tool.red_fun("小明住在北京，热爱NLP。"))
# 输出：小明所住的在北京，热爱NLP。
```
### 10.其他成分多余


```python
from ChineseErrorCorrector.utils.dat import GrammarErrorDat

cged_tool = GrammarErrorDat()
print(cged_tool.red_component("小明住在北京，热爱NLP。"))
# 输出：小明住在北京，热爱NLP。，看着
```
### 11.主语多余


```python
from ChineseErrorCorrector.utils.dat import GrammarErrorDat

cged_tool = GrammarErrorDat()
print(cged_tool.red_sub("小明住在北京，热爱NLP。"))
# 输出：小明住在北京，小明热爱NLP。
```


### 12.语序不当


```python
from ChineseErrorCorrector.utils.dat import GrammarErrorDat

cged_tool = GrammarErrorDat()
print(cged_tool.wrong_sentence_order("小明住在北京，热爱NLP。"))
# 输出：热爱NLP。，小明住在北京

```




### 13.动宾搭配不当


```python
from ChineseErrorCorrector.utils.dat import GrammarErrorDat

cged_tool = GrammarErrorDat()
print(cged_tool.wrong_ver_obj("小明住在北京，热爱NLP。"))
# 输出：None ，即无法进行此类错误的增强

```


### 14.其他搭配不当


```python
from ChineseErrorCorrector.utils.dat import GrammarErrorDat

cged_tool = GrammarErrorDat()
print(cged_tool.other_wrong("小明住在北京，热爱NLP。"))
# 输出：None, 即无法进行此类错误的增强

```



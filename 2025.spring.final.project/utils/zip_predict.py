# 将预测结果压缩后再提交
import os
path=os.getcwd()
newpath=path+"/output/"
os.chdir(newpath)
os.system('zip prediction.zip predict.json')
os.chdir(path)
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:24:14 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
import os
import re
print(os.getcwd())
os.chdir('D://algorithm/20190630LR')
from LRCARD import *

modeldata = pd.read_excel(r'20200224_modeldata.xlsx')
A1 = modeldata.dtypes
A2 = modeldata.columns.tolist()
data = modeldata.rename(columns={'give_time':'time','order_id':'id'})#21914
data = data[data['target'] != -1]#16592


#
Varlist = data.columns.tolist()[23:]
len(Varlist)
lc = LRCARD(df=data.copy(), varlist=Varlist, target='target', 
                 missingvalue=[-9999,-1], specialvalue=[-9999], maxBin=6, Monotonous=1, minBinPcnt=0.02)

spdata, Setdis = lc.dfSplit(validtime='2018-12-18', testPcnt=0.25, trainBadNum=720) #validset:>=validtime trainBadNum：训练集坏数量

VarTrait, VarTypeAt = lc.varCategory(funcVars= Varlist)
vmany = list(VarTrait[VarTrait['varType'] == 'numMany']['var'])
v3 = list(VarTrait[VarTrait['varType'] == 'greater3']['var'])
v2 = list(VarTrait[VarTrait['varType'] == 'numLess3']['var'])
#vstring = list(VarTrait[(VarTrait['varType'] == 'string') & (VarTrait['varUnique'] > 1)]['var'])

#importance
rf_df = lc.rfImportance(funcVars= Varlist)
vim = list(rf_df[rf_df['rf_importance'] > 0.0001]['var'])[:1000]

#Result
#vchi = [i for i in vim if i in vmany]
#vchi.remove('age')
#vchi = vmany.copy()
vchi = Varlist[:-1]
print(len(vchi))
R1, W1, B1, Failvar1 = lc.getResult(funcVars=vchi, binway='chi')
print(R1['var'].unique().shape, W1.shape, B1.shape, len(Failvar1))

vchi3 = [i for i in vim if i in v3]
#vchi3 = v3.copy()
print(len(vchi3))
lc.maxBin = 3
R3, W3, B3, Failvar3 = lc.getResult(funcVars=vchi3, binway='chi')
print(R3['var'].unique().shape, W3.shape, B3.shape, len(Failvar3))
#lc.maxBin = 5

vstay = [i for i in vim if i in v2]
#vstay = v2.copy()
print(len(vstay))
R2, W2, B2, Failvar2 = lc.getResult(funcVars=vstay, binway='nocut')

#不卡方分箱
vnom = ['age']
lc.maxBin = 4
lc.mono = 0
R4, W4, B4, Failvar4 = lc.getResult(funcVars=vnom, binway='chi')
print(R4['var'].unique().shape, W4.shape, B4.shape, len(Failvar4))

#psi
#P, Pdf = lc.cal_Psi(funcdf=B2, funcVars=vstay)
#bigpsi = Pdf[Pdf['var'] == '最近12个月内贷款查询次数']
#concat Result, dfbin, dfwoe
#R = lc.concatResult(Rlist=[R1], Wlist=[W1], Blist=[B1], minimumIV=0.02, maximumPsi=0.1)
#R = lc.concatResult(Rlist=[R1, R3], Wlist=[W1, W3], Blist=[B1, B3], Psi=P)
#R = lc.concatResult(Rlist=[R1, R2, R3], Wlist=[W1, W2, W3], Blist=[B1, B2, B3])
#R = lc.concatResult(Rlist=[R1, R2, R3, R4], Wlist=[W1, W2, W3, W4], Blist=[B1, B2, B3, B4], minimumIV=0.01, maximumPsi=0.1)
#R = lc.concatResult(Rlist=[R1, R2, R4], Wlist=[W1, W2, W4], Blist=[B1, B2, B4], minimumIV=0.01, maximumPsi=0.1)
#R, P, Pdf = lc.concatResult(Rlist=[R1, R2, R3, R4], 
#                                   Wlist=[W1, W2, W3, W4], 
#                                   Blist=[B1, B2, B3, B4], 
#                                   minimumIV=0.02, maximumPsi=10)
R, P, Pdf = lc.concatResult(Rlist=[R1, R4], Wlist=[W1, W4], Blist=[B1, B4], minimumIV=0.02, maximumPsi=10)

#W = lc.dfwoe

#B = lc.dfbin
R.to_excel(r'20200220_R.xlsx')

#pearson
vcorr = list(R['var'].drop_duplicates())
Vstay = lc.correlation(funcVars=vcorr, threshold=0.65)
Rstay = R[R['var'].isin(Vstay)]
Vstayiv = Rstay[['var', 'IV']].drop_duplicates()
#Rstay.to_excel(r'D:\瓜子\2019 项目\20191205 B卡\Rstaypboc2.xlsx')
VstayCorr = lc.dfwoe[Vstay].corr()
#VstayCorr.to_excel(r'D:\瓜子\2019 项目\20191205 B卡\VstayCorr.xlsx')

#Rstay.to_excel(r'Rstay.xlsx')
#train
modelV = Vstay[:10]
modelV = Varlist.copy()
modelV.remove('未结清非银笔数')
modelV.remove('贷款(包含已结清的)过去24个月逾期次数')

Coef, Djdis, Dpdis= lc.modelTraining(modelvar=modelV, seed=12)

modelVc = lc.dfwoe[Varlist].corr()


#lc.scorePicture(Dpdis
#vless0 = list(Coef[Coef['coef'] < 0]['var'])
#Vstay = [i for i in Vstay if i not in vless0]
#print(len(Vstay))


#vless0list = []
#vless0 = list(Coef[Coef['coef'] < 0]['var'])
#vless0list = vless0list + vless0
#vcorr2 = [i for i in vcorr if i not in vless0list]
#Vstay2 = lc.correlation(funcVars=vcorr2, threshold=0.7)
#Mdf, Coef, Djdis, Dpdis= lc.modelTraining(modelvar=Vstay2)

#vmore0 = list(Coef[Coef['coef'] > 0]['var'])
#Coef, Djdis, Dpdis= lc.modelTraining(modelvar=vmore0)

Rselected = lc.Result[lc.Result['var'].isin(modelV)]
Rselected.to_excel(r'20200220_Rstay.xlsx')
#changeset
lc.df = data.rename(columns={'time':'give_time','保险到期日期':'time'})
spdata,Setdis2 = lc.changeSet(newVaTime='2018-12-18', testPcnt=0.25, trainBadNum=725)

#手动调整训练测试验证集的划分,把训练测试集的好样本挪600到验证集
#备份原始数据集
cdata = lc.df.copy()
#
s = lc.df[(lc.df['set'] != 'valid') & (lc.df['target'] == 0) & 
                   (lc.df['time'] > '2019-11-01')]
s1 = s.sample(n=600)
s1['set'] = 'valid'
s2 = lc.df[~lc.df['id'].isin(s1['id'])]
lc.df = s1.append(s2)

setdis3 = lc.df.pivot_table('id', index='set', columns='target', aggfunc='count', margins=True, fill_value=0)
setdis3['badpcnt'] = setdis3[1]/setdis3['All']

del lc.dfwoe['set']
del lc.dfbin['set']
lc.dfwoe = lc.df[['id','set']].merge(lc.dfwoe, how='inner', on='id')
lc.dfbin = lc.df[['id','set']].merge(lc.dfbin, how='inner', on='id')

#vif
finalvar = vmore0.copy()
V = lc.cal_VIF(funcVars=finalvar)

#

#change bin
var = ''
vardf = data[['id', 'target', var]]

result,resultdf = lc.productWoeIV(var,vardf)

lc.maxBin = 3
#lc.mono = 0
result,resultdf = lc.getChiBin(var,vardf) 

binlist=[-1.1, -1, 0, 2, 19]
result,resultdf = lc.AppointBin(var=var, newBin=binlist)


#update Result\result\dfbin\dfwoe
updatedR, updatedr = lc.updatefile(var, result, resultdf, way='change')
#updatedR, updatedr = lc.updatefile(var, result, resultdf, way='add')


#save file
os.chdir(r'20200225')
MResult, Mvarcorr, ScorePsi, VarPsi, Scoredis = lc.finalFile(modelvar=modelV, version='20200225_v1')






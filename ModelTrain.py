# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:52:21 2019

@author: Administrator
"""
import pandas as pd
import numpy as np
import random
import math
import re
import os
import time
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick  
from matplotlib.font_manager import FontProperties
from CutVariables import *
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection  import train_test_split
 
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

class LRCARD:
    def __init__(self, df, varlist, target, missingvalue, specialvalue, maxBin, Monotonous, minBinPcnt):
        self.df = df
        self.dfSize = df.shape[0]
        self.varlist = varlist
        self.target = target
        self.mvalue = missingvalue
        self.svalue = specialvalue
        self.maxBin = maxBin
        self.mono = Monotonous
        self.minBinPcnt = minBinPcnt
    
     
    def dfSplit(self, validtime, testPcnt, trainBadNum):
        validset = self.df[self.df['time'] >= validtime]
        ttset = self.df[self.df['time'] < validtime]
        validset['set'] = 'valid'
        if not trainBadNum:
            trainset,testset = train_test_split(ttset, test_size=testPcnt)
        else:
            ttgood = ttset[ttset[self.target] == 0]
            traingood,testgood = train_test_split(ttgood, test_size=testPcnt)
            ttbad = ttset[ttset[self.target] == 1]
            trainbad = ttbad.sample(n=trainBadNum)
            testbad = ttbad[~ttbad['id'].isin(trainbad['id'])]
            trainset = pd.concat([traingood, trainbad])
            testset = pd.concat([testgood, testbad])
        trainset['set'] = 'train'
        testset['set'] = 'test'
        self.df = pd.concat([trainset, testset, validset])
        setdis = self.df.pivot_table('id', index='set', columns=self.target, aggfunc='count', margins=True, fill_value=0)
        setdis['badpcnt'] = setdis[1]/setdis['All']
        self.df = self.df.sample(frac=1)               
        return self.df, setdis
       
    #变量
    def varCategory(self, funcVars):
        varType = []
        varUnique = []
        varExample = []
        for var in funcVars:
            unique = list(set(list(self.df[var])))            
            varUnique.append(len(unique))
            varExample.append(unique[:10])
            if pd.api.types.is_numeric_dtype(self.df[var]):
                uExs = len([i for i in unique if i not in self.svalue])
                if uExs > self.maxBin:
                    varType.append('numMany')
                elif uExs >= 3:
                    varType.append('greater3')
                elif uExs == 0 or len(unique) == 1:
                    varType.append('equal0')
                else:
                    varType.append('numLess3')
            elif pd.api.types.is_string_dtype(self.df[var]):
                varType.append('string')
            elif pd.api.types.is_timedelta64_dtype(self.df[var]):
                varType.append('time')
            else:
                varType.append(self.df[var].dtypes)
        varTrait = pd.DataFrame({'var':funcVars, 'varType':varType, 'varUnique':varUnique, 'varExample':varExample})
        varTypeAt =  varTrait.groupby('varType')['var'].count()
        print(varTypeAt)   
        return varTrait, varTypeAt
    
    #日期型变量       
    def TransDate(self, funcVars, timeflag):
        self.df[timeflag] = pd.to_datetime(self.df[timeflag])
        for var in funcVars:
            name = var + '_timeflag'
            self.df[name] = (self.df[timeflag] - self.df[var]).days
            del self.df[var]
        return None
    
    #importance
    def rfImportance(self, funcVars):
        rf_cols = []
        norf_cols = []
        for var in funcVars:
            if pd.api.types.is_numeric_dtype(self.df[var]):
                rf_cols.append(var)
            else:
                norf_cols.append(var)
        rf_cols.append('target')
        train = self.df[rf_cols].values
        x = train[:,:-1]
        y = train[:,-1]  
        rf = RandomForestRegressor()
        rf.fit(x,y)    
        rf_importance = rf.feature_importances_
        rf_df = pd.DataFrame({"var":rf_cols[:-1],"rf_importance":rf_importance},columns = ["var","rf_importance"])
        rf_df = rf_df.sort_values("rf_importance",ascending = False)
        return rf_df
    
    #IV
    def productWoeIV(self, var, vardf):
        result = vardf.groupby(var,as_index=False)[self.target].agg({'total':'count','bad':'sum','bad/total':'mean'})
        result['good'] = result['total'] - result['bad']
        T = sum(result['total'])
        B = sum(result['bad'])
        G = T - B
        result['total_pcnt'] = result['total']/T
        result['bad/totalBad'] = result['bad']/B
        result['good/totalGood'] = result['good']/G
        result['good/total'] = result['good']/result['total']  
        result['woe'] = np.log((result['bad/totalBad']+1e-10)/(result['good/totalGood']+1e-10))
        result['iv'] = result['woe'] * (result['bad/totalBad']-result['good/totalGood'])
        result['IV'] = result['iv'].sum()
        result['var'] = var    
        rawvar = re.sub(r'_delta|_merging', '', var)
        result['missingRate'] = (vardf[vardf[rawvar].isin(self.mvalue)].shape[0])*1.0/self.dfSize        
        vardf = pd.merge(vardf,result[[var,'woe']].rename(columns={'woe':var+'_woe'}),how='left', on=var)   
        result.rename(columns = {var : 'var_bin'}, inplace=True)   
        result = result.ix[:,['var','missingRate','var_bin','total','total_pcnt','bad','bad/total','woe','IV']]
        result = result.sort_values('bad/total', ascending=False)
        result = result.reset_index(drop=True)
        result.loc['合计'] = result.loc[:,['total','bad']].apply(lambda x : x.sum())
        result.loc['合计','bad/total'] = result.loc['合计','bad']/result.loc['合计','total']
        result.loc['合计','var'] = var
        result.loc['合计','IV'] = result.ix[0,'IV']
        return result, vardf
    
    #单调性
    def BinMonoJudge(self, var, fo):
        badrate = fo.groupby(var,as_index=False)[self.target].agg({'total':'count','bad':'sum','bad/total':'mean'})
        badrate = badrate.sort_values('bad/total')
        badrate = badrate.reset_index(drop = True)
        bin_left = []
        s = badrate.shape[0]
        badrate[var] = badrate[var].astype(str)
        for i in range(s):
            le = float(np.array(badrate.iloc[[i],[0]]).tolist()[0][0][1:-1].split(',')[0])
            bin_left.append(le)
        if sorted(bin_left) == bin_left or sorted(bin_left,reverse=True) == bin_left: 
            return 1#单调
        else:
            return 0#非单调
    #卡方分箱
    def getChipoint(self, var, fo, max_interval):
            Chipoint = ChiMerge(fo, var, self.target, max_interval, [], self.minBinPcnt)    
            Chipoint.insert(0, fo[var].min()-0.1)
            Chipoint.append(fo[var].max()) 
            return Chipoint
        
    def getChiBin(self, var, vardf):
        name0 = var + '_delta'
        if len(self.svalue) != 0:
            fo = vardf[~vardf[var].isin(self.svalue)] 
            fsvalue = vardf[vardf[var].isin(self.svalue)]
            fsvalue[name0] = fsvalue[var]
        else:
            fo = vardf.copy()
            fsvalue = pd.DataFrame()
        fo = fo.sort_values(by=var) 
        varBinNum = self.maxBin
        if self.mono == 1:#要求单调
            Chipoint = self.getChipoint(var, fo, varBinNum)
            fo[name0] = pd.cut(fo[var], Chipoint)
            varMono = self.BinMonoJudge(name0, fo)
            while varMono == 0:#不单调
                varBinNum -= 1  
                Chipoint = self.getChipoint(var, fo, varBinNum)
                fo[name0] = pd.cut(fo[var], Chipoint)
                varMono = self.BinMonoJudge(name0, fo)
            else:
                pass
        else:
            Chipoint = self.getChipoint(var, fo, varBinNum)
            fo[name0] = pd.cut(fo[var], Chipoint)
        fo[name0] = fo[name0].astype(str)
        vardf = pd.concat([fo,fsvalue]) 
        result, vardf = self.productWoeIV(name0, vardf)            
        return result, vardf
    
    #
    def getResult(self, funcVars, binway):
        Result = pd.DataFrame(columns = ['var','missingRate','var_bin','total','total_pcnt','bad','bad/total','woe','IV']) 
        dfbin = self.df[['id','time','set',self.target]]
        dfwoe = self.df[['id','time','set',self.target]]
        failvar = []
        n = 0
        for var in funcVars:
            n += 1
            print(n, '/'+str(len(funcVars)))
            name0 = str(var) + '_delta'
            name1 = name0 + '_woe'      
            vardf = self.df[['id', self.target, var]]            
            try:
                if binway == 'chi':
                    result,resultdf = self.getChiBin(var, vardf)
                else:
                    vardf[name0] = vardf[var]
                    result,resultdf = self.productWoeIV(name0, vardf)
                Result = pd.concat([Result,result])
                dfbin = dfbin.merge(resultdf[['id', name0]], how='inner', on='id')
                dfwoe = dfwoe.merge(resultdf[['id', name1]], how='inner', on='id')
            except:
                failvar.append(var)
        Result['var'] = Result['var'].map(lambda x : re.sub(r'_delta', '', x))        
        Result = Result.sort_values(['IV', 'var'], ascending=False)
        dfbin.rename(columns = lambda x : re.sub(r'_delta', '', x), inplace=True)
        dfwoe.rename(columns = lambda x : re.sub(r'_delta_woe', '', x), inplace=True)    
        return Result, dfwoe, dfbin, failvar
    
    #指定分箱
    def AppointBin(self, var, newBin):
        name0 = var + '_delta'
        vardf = self.df[['id', self.target, var]]
        vardf = vardf.sort_values(by=var)
        vardf[name0] = pd.cut(vardf[var], newBin)
        vardf[name0] = vardf[name0].astype(str)
        result,resultdf = self.productWoeIV(name0, vardf)
        return result, resultdf
    
    #重新分箱或增加新变量后更新Result, dfbin, dfwoe
    def updatefile(self, var, result, resultdf, way):
        name0 = var + '_delta'
        name1 = name0 + '_woe'   
        if way == 'change':
            self.Result = self.Result[self.Result['var'] != var]
            del self.dfbin[var]
            del self.dfwoe[var]    
        self.Result = pd.concat([self.Result, result])
        self.dfbin = self.dfbin.merge(resultdf[['id', name0]], how='inner', on='id')
        self.dfwoe = self.dfwoe.merge(resultdf[['id', name1]], how='inner', on='id')    
        self.Result['var'] = self.Result['var'].map(lambda x : re.sub(r'_delta', '', x))
        self.Result = self.Result.sort_values(['IV', 'var'], ascending=False)
        self.result = self.Result[['var', 'IV']].drop_duplicates('var')
        self.dfbin.rename(columns = lambda x : re.sub(r'_delta', '', x), inplace=True)
        self.dfwoe.rename(columns = lambda x : re.sub(r'_delta_woe', '', x), inplace=True)
        return self.Result, self.result
    
    #组合
    def concatResult(self, Rlist, Wlist, Blist, minimumIV, maximumPsi):
        R = pd.concat(Rlist).sort_values(['IV', 'var'], ascending=False)
        Vhighiv = list(set(list(R[R['IV'] >= minimumIV]['var'])))   
        
        B = Blist[0]
        for b in Blist[1:]:
            b = b.drop(['time', 'set', 'target'], axis=1)
            B = B.merge(b, how='inner', on='id')
        B.iloc[:, 4:] = B.iloc[:, 4:].astype(str)
        psiAll, psidf = self.cal_Psi(B, Vhighiv)
        Vhl = list(set(psiAll[(psiAll['ttpsi'] <= maximumPsi) & (psiAll['tvpsi'] <= maximumPsi)]['var'].tolist()))
        print('highiv and lowpsi var/varlist:{}/{}'.format(len(Vhl), len(self.varlist)))
                
        self.Result = R[R['var'].isin(Vhl)]
        self.result = self.Result[['var', 'IV']].drop_duplicates('var')
        
        W = Wlist[0]
        for w in Wlist[1:]:
            w = w.drop(['time', 'set', 'target'], axis=1)
            W = W.merge(w, how='inner', on='id')

        self.dfwoe = W.loc[:, ['id', 'time', 'set', 'target']+Vhl]
        self.dfbin = B.loc[:, ['id', 'time', 'set', 'target']+Vhl]
        print('var个数:{}, dfwoeShape:{}, dfbinShape:{}'.format(self.result.shape, self.dfwoe.shape, self.dfbin.shape))
        return self.Result, psiAll, psidf

    #psi
    def cal_Psi(self, funcdf, funcVars):
        psiAll = pd.DataFrame(columns=['var','ttpsi', 'tvpsi'])
        psidf = pd.DataFrame(columns=['var','var_bin',
                                       ('count', 'train'),'train',('sum', 'train'),('mean', 'train'),
                                       ('count', 'test'),'test',('sum', 'test'),('mean', 'test'),
                                       'ttpsi',
                                       ('count', 'valid'),'valid',('sum', 'valid'),('mean', 'valid'),
                                       'tvpsi'])
        for var in funcVars:
            setdis = funcdf.pivot_table('target', index=var, columns='set', aggfunc=['count', 'sum', 'mean'], fill_value=0)
            
            setpcnt = setdis['count'].apply(lambda x : x/x.sum())
            setpcnt['ttpsi'] = (setpcnt['train'] - setpcnt['test'])*np.log((setpcnt['train']+1e-10)/(setpcnt['test']+1e-10))
            setpcnt['tvpsi'] = (setpcnt['train'] - setpcnt['valid'])*np.log((setpcnt['train']+1e-10)/(setpcnt['valid']+1e-10))
            
            setdis = setdis.merge(setpcnt, how='inner', left_index=True, right_index=True)
            setdis = setdis.sort_values(('mean', 'train'), ascending=False)
            setdis.index = setdis.index.astype(str)
            setdis.loc['sum'] = setdis.apply(lambda x : x.sum(), axis=0)
            for i in ['train','test','valid']:
                setdis.ix[-1, ('mean', i)] = setdis.ix['sum',('sum', i)]/setdis.ix['sum',('count', i)]            
            setdis.reset_index(inplace=True)
            setdis = setdis.rename(columns={var:'var_bin'})
            setdis.insert(0,'var',[var] * len(setdis))
                        
            varpsi = pd.DataFrame({'var':var, 'ttpsi':setpcnt['ttpsi'].sum(), 'tvpsi':setpcnt['tvpsi'].sum()}, index=[0])
            psiAll = pd.concat([psiAll, varpsi])
            psidf = pd.concat([psidf, setdis])
        psiAll = psiAll.sort_values('tvpsi', ascending = False)
        
        return psiAll, psidf
    
    #coef
    def coefTest(self, funcVars):
        varCoef = pd.DataFrame(columns = ['var', 'coef'])
        for var in funcVars:
            modeldf = self.dfwoe[[var, self.target]].values
            X_train,y_train = modeldf[:,0].reshape(-1, 1), modeldf[:,-1].astype('int')
            clf = LogisticRegression(penalty='l2',solver = 'liblinear',class_weight = 'balanced',tol=1e-4)
            clf.fit(X_train,y_train)
            varcoef = pd.DataFrame({'var':var, 'coef':clf.coef_.tolist()[0]})
            varCoef = pd.concat([varCoef, varcoef])
        varCoef1 = varCoef[varCoef['coef'] > 0]
        varCoef2 = varCoef[varCoef['coef'] < 0]
        print('大于0:', varCoef1.shape[0], '小于0:', varCoef2.shape[0])
        varCoef = varCoef.sort_values('coef', ascending = False)
        return varCoef
    
    #pearson
    def correlation(self, funcVars, threshold):
        ivcending = list(self.result[self.result['var'].isin(funcVars)]['var'])        
        corrdf = self.dfwoe.ix[:, ivcending]
        corr_matrix = corrdf.corr()
        colsStay = []
        while corr_matrix.empty is False:
            highivar = corr_matrix.columns.tolist()[0]
            colsStay.append(highivar)
            corr_matrix = corr_matrix[corr_matrix[highivar] < threshold]
            stay_col = corr_matrix.index.tolist()
            corr_matrix = corr_matrix[stay_col]
        else:
            pass
        print('stay:',len(colsStay))
        return colsStay
            
    #training
    def modelTraining(self, modelvar, seed):
        time_start = time.clock()
        cols = modelvar + ['id', 'set', self.target]
        modeldf = self.dfwoe[cols]    
        #模型训练
        mdfTrain = modeldf[modeldf['set'] == 'train'].values
        X_train, y_train = mdfTrain[:,:-3], mdfTrain[:,-1].astype('int')
        
        tuned_parameters = [{'C':np.arange(0.01,1,0.01)}]
        clf=GridSearchCV(LogisticRegression(penalty='l2',random_state=seed,
                                             #solver = 'sag',
                                             solver = 'liblinear',
#                                             class_weight = 'balanced',
                                             tol=1e-4),
                         tuned_parameters,scoring='roc_auc',cv=10) 
        clf.fit(X_train,y_train)
    #    print('模型最佳参数',clf.best_estimator_)
        clf = clf.best_estimator_
        intercept = clf.intercept_[0]#截距
        self.col_coef = pd.DataFrame({"var":modelvar,"coef":clf.coef_.tolist()[0]})
        
        ttv = modeldf.values[:,:-3]
        y_proba = clf.predict_proba(ttv)
        modeldf['prob_0'] = y_proba[:,0]
        modeldf['prob_1'] = y_proba[:,1]
        
        self.train_ks, self.train_auc,self.train_fpr,self.train_tpr = KS(df=modeldf[modeldf['set']=='train'], score='prob_0', target=self.target)
        self.test_ks, self.test_auc, self.test_fpr, self.test_tpr = KS(df=modeldf[modeldf['set']=='test'], score='prob_0', target=self.target)
        self.valid_ks, self.valid_auc, self.valid_fpr, self.valid_tpr = KS(df=modeldf[modeldf['set']=='valid'], score='prob_0', target=self.target)
        
        #打分
        pred1_rate = y_proba[:,1]
        oddsAll = pred1_rate/(1 - pred1_rate)    
        oddmax, oddmin = oddsAll.max(), oddsAll.min()
        scoremax, scoremin = 800,300
        
        self.B = (scoremax - scoremin)/np.log(oddmax/oddmin)
        self.A = scoremax + self.B * np.log(oddmin)
        self.PDO = self.B * np.log(2)
        self.base_score = self.A - self.B * intercept
        print('A=:{:.0f}, B={:.0f}, PDO={:.0f}, base_score={:.0f}'.format(self.A, self.B, self.PDO, self.base_score))
        
        modeldf['odds'] = modeldf.apply(lambda x : x['prob_1']/x['prob_0'],axis=1)
        modeldf['score'] = modeldf.apply(lambda x : self.A - self.B*np.log(x['odds']),axis=1).astype(int)
        
        cutpoint = [300+i*50 for i in range(11)]
        cutpoint[0] = cutpoint[0]-2    
        modeldf['score_delta_dj'] = pd.cut(modeldf['score'], cutpoint)
        modeldf['score_delta_dp'] = pd.qcut(modeldf['score'], 10, duplicates='drop')
        modeldf['score_delta_dj'], modeldf['score_delta_dp']= modeldf['score_delta_dj'].astype(str), modeldf['score_delta_dp'].astype(str)
        djdis = modeldf.groupby('score_delta_dj')[self.target].agg({'amount':'count', 'badrate':'mean'}).fillna(0)
        dpdis = modeldf.groupby('score_delta_dp')[self.target].agg({'amount':'count', 'badrate':'mean'}).fillna(0)
        djdis['pcnt'] = djdis['amount'].map(lambda x : '%.1f%%'%(x*100/self.dfSize))
        dpdis['pcnt'] = dpdis['amount'].map(lambda x : '%.1f%%'%(x*100/self.dfSize))

        self.scorePicture(djdis)
        self.scorePicture(dpdis)
        
        print('trainKS:{:.4f}, trainAUC:{:.4f}'.format(self.train_ks, self.train_auc))
        print('testKS:{:.4f}, testAUC:{:.4f}'.format(self.test_ks, self.test_auc))
        print('validKS:{:.4f}, validAUC:{:.4f}'.format(self.valid_ks, self.valid_auc))        
        self.modeldf = modeldf
        
        time_end = time.clock()
        print("运行时间：{0}分钟".format(int((time_end - time_start)/60)))#程序运行多少分钟

        return self.col_coef, djdis, dpdis

    def scorePicture(self, scoredis):
        a = list(scoredis['amount'])  
        b = [float('%.2f'%(i*100)) for i in list(scoredis['badrate'])]
        l=[i for i in range(len(scoredis))]   
                
        fmt='%.2f%%'
        yticks = mtick.FormatStrFormatter(fmt)  #设置百分比形式的坐标轴
        lx=[str(i)[-6:-3] for i in list(scoredis.index)]       
        
        fig = plt.figure(figsize=(7,4))
          
        ax1 = fig.add_subplot(111)  
        ax1.plot(l, b,'or-',label=u'坏账率');
        ax1.yaxis.set_major_formatter(yticks)
        for i,(_x,_y) in enumerate(zip(l,b)):  
            plt.text(_x,_y,b[i],color='black',fontsize=10,)  #将数值显示在图形上
        ax1.legend(loc=1)
        if max(b)+10 < 100:
            ax1.set_ylim([0, max(b)+10]);
        else:
            ax1.set_ylim([0, 100]);
        ax1.set_ylabel('坏账率');
        
        plt.legend(prop={'family':'SimHei','size':8})  #设置中文
        
        ax2 = ax1.twinx() # this is the important function  
        plt.bar(l, a, alpha=0.3,color='steelblue',label=u'样本量')  
        ax2.legend(loc=2)
        ax2.set_ylim([0, max(a)+500])  #设置y轴取值范围
        plt.legend(prop={'family':'SimHei','size':8},loc="upper left") 
        plt.xticks(l,lx)
        for i,(_x,_y) in enumerate(zip(l,a)):  
            plt.text(_x,_y,a[i],color='black',fontsize=10,)  #将数值显示在图形上
        
        plt.xlabel("箱右节点")
        plt.title("评分分布")
        plt.show()    
        
        return None
    
    def changeSet(self, newVaTime, testPcnt, trainBadNum):
        self.df, setdis = self.dfSplit(newVaTime, testPcnt, trainBadNum)        
        del self.dfwoe['set']
        del self.dfbin['set']
        self.dfwoe = self.df[['id','set']].merge(self.dfwoe, how='inner', on='id')
        self.dfbin = self.df[['id','set']].merge(self.dfbin, how='inner', on='id')
        return self.df, setdis
        
    #vif
    def cal_VIF(self, funcVars):
        vardf = self.dfwoe[funcVars]
        X = np.matrix(vardf)
        VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        vif_df = pd.DataFrame({"var":funcVars,"VIF":VIF_list})
        self.vif_df = vif_df.sort_values("VIF",ascending = False)
        return self.vif_df
    
    def picture(self, fpr,tpr,ks,auc, part):
        mpl.rcParams['font.sans-serif']=[u'simHei']
        mpl.rcParams['axes.unicode_minus']=False
        plt.figure(figsize=(8,6),facecolor="w")
        plt.plot(fpr,tpr,c="r",lw=2,label=u"LR算法,AUC=%.4f" %auc)
        plt.plot((0,1),(0,1),c='#a0a0a0',lw=2,ls='--')
        plt.xlim = ([-0.01,1.02])
        plt.ylim = ([-0.01,1.02])
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('False Positive Rate(FPR)', fontsize=16)
        plt.ylabel('True Positive Rate(TPR)', fontsize=16)
        plt.grid(b=True, ls=':')
        plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
        plt.title(u'{}集的ROC/AUC  KS={}'.format(part, '%.4f'%ks), fontsize=18)
        plt.savefig(part+'集LR算法的ROC和AUC.jpeg')#png
    #    plt.show()
        return None
    
    def scoreDistribution(self, funcdf):
        for i in['train','test','valid']:
            funcdf[i + '_good'] = funcdf[('count', i)] - funcdf[('sum', i)]
        funcdf['total_good'] = funcdf['train_good'] + funcdf['test_good'] + funcdf['valid_good']
        funcdf['total_bad'] = funcdf[('sum', 'train')] + funcdf[('sum', 'test')] + funcdf[('sum', 'valid')]
        funcdf['total_all'] = funcdf['total_good'] + funcdf['total_bad']
        funcdf['total_badpcnt'] = funcdf['total_bad']/(funcdf['total_all']+1e-10)        
        scoredis = funcdf[['var', 'var_bin',
                               'total_good','total_bad','total_all', 'total_badpcnt',
                               'train_good',('sum', 'train'),('count', 'train'),('mean', 'train'),
                               'test_good',('sum', 'test'),('count', 'test'),('mean', 'test'),
                               'valid_good',('sum', 'valid'),('count', 'valid'),('mean', 'valid')]]
        return scoredis
        
        
    #final file
    def finalFile(self, modelvar, version):
        
        mResult = self.Result[self.Result['var'].isin(modelvar)]
        self.vif_df = self.cal_VIF(modelvar)
        mResult = mResult.merge(self.vif_df, how='inner', on='var').merge(self.col_coef, how='inner', on='var')
        mResult['score'] = mResult.apply(lambda x : self.B * ((-1)*x['woe']) * x["coef"],axis=1)
        mvar = mResult[['var','IV','VIF','missingRate']].drop_duplicates('var')
        mvarDetail = mResult[mResult['var_bin'].notnull()][['var','var_bin','total_pcnt','bad/total','woe','coef','score']]
        
        mvarcorr = self.dfwoe[modelvar].corr()
        
        mparam = pd.DataFrame({'train':[self.train_ks, self.train_auc], 
                           'test':[self.test_ks, self.test_auc], 
                           'valid':[self.valid_ks, self.valid_auc],
                           'train-tast':[self.train_ks-self.test_ks, self.train_auc-self.test_auc],
                           'A & B':[self.A, self.B], 'PDO & base_score':[self.PDO, self.base_score]})
        

        self.modeldf = self.modeldf.sort_values('score')
        scorePsi, scorePsidf = self.cal_Psi(self.modeldf, ['score_delta_dp','score_delta_dj'])
        
        mdfbin = self.dfbin[['id', 'set', 'target'] + modelvar]
        varPsi,varPsidf = self.cal_Psi(mdfbin, modelvar)
        
        mdfbin['baseScore'] = round(self.base_score)

        for var in modelvar:
            name = 'score_' + var
            binscore = mResult[mResult['var'] == var]
            binscore['var_bin'] = binscore['var_bin'].astype(str)
            score_dict = dict(zip(binscore['var_bin'], binscore['score']))
#            mdfbin[var] = mdfbin[var].astype(str)
            mdfbin[name] = mdfbin[var].map(lambda x : score_dict[x])

        mdfbin = mdfbin.drop(['set', 'target'], axis=1)
        
        scoreDis = self.scoreDistribution(scorePsidf)
        
        scorePsidf = scorePsidf.iloc[:, :-7]
        
        mdfdata = self.df[['id', 'time'] + modelvar].merge(
                mdfbin, how='inner', on='id').merge(
                self.modeldf, how='inner', on='id')
        
        if version:
            writer = pd.ExcelWriter('RawResult_'+version+'.xlsx')
            mResult.to_excel(writer, sheet_name='MResult')
            mvar.to_excel(writer, sheet_name='Mvar')
            mvarcorr.to_excel(writer, sheet_name='Mvarcorr')
            mvarDetail.to_excel(writer, sheet_name='MvarDetail')  
            mparam.to_excel(writer, sheet_name='Mparam') 
            scorePsidf.to_excel(writer, sheet_name='ScorePsi')
            varPsidf.to_excel(writer, sheet_name='VarPsi')
            scoreDis.to_excel(writer, sheet_name='ScoreDis')
            mdfdata.to_excel(writer, sheet_name='mdfdata')
            writer.save()
            self.picture(self.train_fpr,self.train_tpr,self.train_ks, self.train_auc, part='训练')
            self.picture(self.test_fpr,self.test_tpr,self.test_ks,self.test_auc,part='测试')
            self.picture(self.valid_fpr,self.valid_tpr,self.valid_ks, self.valid_auc,part='验证')
        
        return mResult, mvarcorr, scorePsidf, varPsidf, scoreDis#, mdfdata
        


    

    
    
    
    
    
    
    
    
    
    
    
    






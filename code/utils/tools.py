# Module for basic tools
# Taku Ito
# 12/11/2019

import numpy as np
import h5py
import scipy.stats as stats

tasks = {'EMOTION':[0,1],
         'GAMBLING':[2,3],
         'LANGUAGE':[4,5],
         'MOTOR':[6,7,8,9,10,11],
         'RELATIONAL':[12,13],
         'SOCIAL':[14,15],
         'WM':[16,17,18,19,20,21,22,23]}
taskNames = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
TRsPerRun = [176,176,253,253,316,316,284,284,232,232,274,274,405,405]
restRuns = ['rfMRI_REST1_RL', 'rfMRI_REST1_LR','rfMRI_REST2_RL', 'rfMRI_REST2_LR']
taskRuns= ['tfMRI_EMOTION_RL','tfMRI_EMOTION_LR','tfMRI_GAMBLING_RL','tfMRI_GAMBLING_LR',
           'tfMRI_LANGUAGE_RL','tfMRI_LANGUAGE_LR','tfMRI_MOTOR_RL','tfMRI_MOTOR_LR',
           'tfMRI_RELATIONAL_RL','tfMRI_RELATIONAL_LR','tfMRI_SOCIAL_RL','tfMRI_SOCIAL_LR','tfMRI_WM_RL','tfMRI_WM_LR']

def loadRestResiduals(subj,model='24pXaCompCorXVolterra',zscore=False,FIR=False):
    datafile = '/projects/f_mc1689_1/HCP352Data/data/hcppreprocessedmsmall/parcellated_postproc/' + subj + '_glmOutput_data.h5' 
    h5f = h5py.File(datafile,'r')
    data = []
    if FIR:
        dataid = 'rfMRI_REST_' + model + '_taskReg_resid_FIR'
        data = h5f['taskRegression'][dataid][:]
        if zscore:
            # Zscore each run separately
            runstart = 0
            for run in range(4):
                runend = runstart + 1195
                data[:,runstart:runend] = stats.zscore(data[:,runstart:runend],axis=1)
                runstart += 1195
                
            # Now z-score rest time series as if it were task
            trcount = 0
            for ntrs in TRsPerRun:
                trstart = trcount
                trend = trcount + ntrs
                data[:,trstart:trend] = stats.zscore(data[:,trstart:trend],axis=1)

                trcount += ntrs

        data = data.T
    else:
        for run in restRuns:
            dataid = run + '/nuisanceReg_resid_' + model
            tmp = h5f[dataid][:]
            if zscore:
                tmp = stats.zscore(tmp,axis=1)
            data.extend(tmp.T)
    data = np.asarray(data).T
    h5f.close()
    return data

def loadTaskActivity64k(subj,task,model='24pXaCompCorXVolterra',zscore=False):
    datafile = '/projects/f_mc1689_1/HCP352Data/data/hcppreprocessedmsmall/vertexWise/' + subj + '_glmOutput_64k_data.h5' 
    h5f = h5py.File(datafile,'r')
    dataid = task + '_' + model + '_taskReg_betas_canonical'
    betas = h5f['taskRegression'][dataid][:]
    h5f.close()
    betas = betas[:,1:]
    return betas

def loadTaskActivityParcels(subj,task,model='24pXaCompCorXVolterra',zscore=False):
    datafile = '/projects/f_mc1689_1/HCP352Data/data/hcppreprocessedmsmall/parcellated_postproc/' + subj + '_glmOutput_data.h5' 
    h5f = h5py.File(datafile,'r')
    dataid = 'tfMRI_' + task + '_' + model + '_taskReg_betas_canonical'
    betas = h5f['taskRegression'][dataid][:]
    h5f.close()
    betas = betas[:,1:]
    return betas

def loadTaskResiduals(subj, model='24pXaCompCorXVolterra', taskModel='FIR', zscore=False):
    datafile = '/projects/f_mc1689_1/HCP352Data/data/hcppreprocessedmsmall/parcellated_postproc/' + subj + '_glmOutput_data.h5' 
    h5f = h5py.File(datafile,'r')
    resids = []
    for task in taskNames:
        dataid = 'tfMRI_' + task + '_' + model + '_taskReg_resid_' + taskModel
        tmp = h5f['taskRegression'][dataid][:]
        if zscore:
            nTRsPerRun = int(tmp.shape[1]/2)
            tmp[:,:nTRsPerRun] = stats.zscore(tmp[:,:nTRsPerRun],axis=1)
            tmp[:,nTRsPerRun:] = stats.zscore(tmp[:,nTRsPerRun:],axis=1)
        resids.extend(tmp.T)
    resids = np.asarray(resids).T
    h5f.close()
    return resids

def loadMultRegFC(subj):
    datafile = '/projects/f_mc1689_1/NetworkDiversity/data/multregfc/' + subj + '_restFC.h5'
    h5f = h5py.File(datafile,'r')
    fc = h5f['multreg_restfc'][:].copy()
    h5f.close()
    return fc


def loadTaskFullTS(subj, model='24pXaCompCorXVolterra', zscore=False):
    #datafile = '/projects3/TaskFCMech/data/hcppreprocessedmsmall/hcpPostProcCiric/' + subj + '_glmOutput_data.h5'         
    datafile = '/projects/f_mc1689_1/HCP352Data/data/hcppreprocessedmsmall/parcellated_postproc/' + subj + '_glmOutput_data.h5' 
    h5f = h5py.File(datafile,'r')
    resids = []
    for task in taskRuns:
        tmp = h5f[task]['nuisanceReg_resid_24pXaCompCorXVolterra'][:]
        if zscore:
            tmp = stats.zscore(tmp,axis=1)
        resids.extend(tmp.T)
    resids = np.asarray(resids).T
    h5f.close()
    return resids

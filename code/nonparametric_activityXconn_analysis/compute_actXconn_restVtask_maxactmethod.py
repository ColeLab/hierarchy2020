import numpy as np
import sys
sys.path.append('../')
import utils.taskGLMPipeline as tgp
import utils.tools as tools
import os
import multiprocessing as mp
import scipy.stats as stats
os.environ['OMP_NUM_THREADS'] = str(1)
import statsmodels.sandbox.stats.multicomp as mc
import sklearn
import nibabel as nib
import pandas as pd
import sys
import taskGLMPipeline as tgp
import h5py
from importlib import reload

#######################################################################################################
#### Construct global variables
## Exploratory subjects
#subjNums = ['178950','189450','199453','209228','220721','298455','356948','419239','499566','561444','618952','680452','757764','841349','908860',
#            '103818','113922','121618','130619','137229','151829','158035','171633','179346','190031','200008','210112','221319','299154','361234',
#            '424939','500222','570243','622236','687163','769064','845458','911849','104416','114217','122317','130720','137532','151930','159744',
#            '172029','180230','191235','200614','211316','228434','300618','361941','432332','513130','571144','623844','692964','773257','857263',
#            '926862','105014','114419','122822','130821','137633','152427','160123','172938','180432','192035','200917','211417','239944','303119',
#            '365343','436239','513736','579665','638049','702133','774663','865363','930449','106521','114823','123521','130922','137936','152831',
#            '160729','173334','180533','192136','201111','211619','249947','305830','366042','436845','516742','580650','645450','715041','782561',
#            '871762','942658','106824','117021','123925','131823','138332','153025','162026','173536','180735','192439','201414','211821','251833',
#            '310621','371843','445543','519950','580751','647858','720337','800941','871964','955465','107018','117122','125222','132017','138837',
#            '153227','162329','173637','180937','193239','201818','211922','257542','314225','378857','454140','523032','585862','654350','725751',
#            '803240','872562','959574','107422','117324','125424','133827','142828','153631','164030','173940','182739','194140','202719','212015',
#            '257845','316633','381543','459453','525541','586460','654754','727553','812746','873968','966975']

## Validation subjects
subjNums = ['100206','108020','117930','126325','133928','143224','153934','164636','174437','183034','194443','204521','212823','268749','322224',
             '385450','463040','529953','587664','656253','731140','814548','877269','978578','100408','108222','118124','126426','134021','144832',
             '154229','164939','175338','185139','194645','204622','213017','268850','329844','389357','467351','530635','588565','657659','737960',
             '816653','878877','987074','101006','110007','118225','127933','134324','146331','154532','165638','175742','185341','195445','205119',
             '213421','274542','341834','393247','479762','545345','597869','664757','742549','820745','887373','989987','102311','111009','118831',
             '128632','135528','146432','154936','167036','176441','186141','196144','205725','213522','285345','342129','394956','480141','552241',
             '598568','671855','744553','826454','896879','990366','102513','112516','118932','129028','135629','146533','156031','167440','176845',
             '187850','196346','205826','214423','285446','348545','395756','481042','553344','599671','675661','749058','832651','899885','991267',
             '102614','112920','119126','129129','135932','147636','157336','168745','177645','188145','198350','208226','214726','286347','349244',
             '406432','486759','555651','604537','679568','749361','835657','901442','992774','103111','113316','120212','130013','136227','148133',
             '157437','169545','178748','188549','198451','208327','217429','290136','352738','414229','497865','559457','615744','679770','753150',
             '837560','907656','993675','103414','113619','120414','130114','136833','150726','157942','171330']


## General parameters/variables
nParcels = 360
nSubjs = len(subjNums)

glasserfile2 = '../../resultdata/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
glasser2 = nib.load(glasserfile2).get_data()
glasser2 = np.squeeze(glasser2)

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

model = '24pXaCompCorXVolterra'
zscore = False
nTasks = 24
#######################################################################################################

#######################################################################################################
#### Load data
restTS = np.zeros((nParcels,4780,len(subjNums)))
taskTS = np.zeros((nParcels,3810,len(subjNums)))
task_timing = np.zeros((3810,nTasks,len(subjNums)))
scount = 0
for subj in subjNums:
    if scount%25==0: print('Loading in data for subject', scount+1, '/', len(subjNums))
    restTS[:,:,scount] = tools.loadRestResiduals(subj,model=model,zscore=zscore,FIR=False)
    taskTS[:,:,scount] = tools.loadTaskFullTS(subj, model=model,zscore=zscore) # We make this FIR since we only want to use FIR regressed task residuals for FC analysis
    task_timing[:,:,scount] = tgp.loadTaskTimingForAllTasks(subj,taskModel='canonical')['taskRegressors']>.5 # Only to obtain which time points to estimate FC with
    scount += 1
    
task_timing = task_timing.astype(bool)
#######################################################################################################


#######################################################################################################
#### Compute task fc connectivity
taskFC = np.zeros((nParcels,nParcels,24,len(subjNums)))
restFC = np.zeros((nParcels,nParcels,24,len(subjNums)))

restAct = np.zeros((nParcels,24,len(subjNums)))
taskAct = np.zeros((nParcels,24,len(subjNums)))

taskGBC = np.zeros((nParcels,24,len(subjNums)))
restGBC = np.zeros((nParcels,24,len(subjNums)))

taskFC_alltasks = np.zeros((nParcels,nParcels,len(subjNums)))
restFC_alltasks = np.zeros((nParcels,nParcels,len(subjNums)))

taskGVC = np.zeros((nParcels,len(subjNums)))
for s in range(len(subjNums)):
    if s%25==0: print('Subject ' + str(s))
    condcount = 0
    trcount = 0
    # Find the minimum number of TRs for each task
    min_tps = int(np.min(np.sum(task_timing[:,:,s],axis=0))/2.0)
    for task in taskNames:
        
        # Load regressors for data
        X = tgp.loadTaskTiming(subj, 'tfMRI_' + task, taskModel='canonical', nRegsFIR=25)

        taskRegs = X['taskRegressors'] # These include the two binary regressors
        runlength = int(taskRegs.shape[0])

        taskRegs = taskRegs > 0.5

        taskRegs_allCond = np.sum(taskRegs,axis=1)>0
        
        # Iterate through conditions        
        taskblock_mats = []
        restblock_mats = []
        for cond in range(taskRegs.shape[1]):
            # Identify specific tps during task blocks
            task_tp_ind = np.where(taskRegs[:,cond])[0]

            onsets = np.where(np.diff(np.asarray(taskRegs[:,cond],dtype=int))==1)[0]
            offsets = np.where(np.diff(np.asarray(taskRegs[:,cond],dtype=int))==-1)[0]
            baseline_iti = np.where(np.asarray(taskRegs_allCond,dtype=int)==0)[0]
            nblocks = len(onsets)
            
            taskTS_tmp = taskTS[:,trcount:(trcount+runlength),s]
            restTS_tmp = restTS[:,trcount:(trcount+runlength),s]

            baseline_activity = np.mean(taskTS_tmp[:,baseline_iti],axis=1)
            baseline_activity_rest = np.mean(restTS_tmp[:,baseline_iti],axis=1)

            tmp_task_mat = np.zeros((nParcels,nblocks))
            tmp_rest_mat = np.zeros((nParcels,nblocks))
            for block in range(nblocks):
                # Try catch block if offset is after run end (after accounting for HRF lag)
                for parcel in range(360):
                    try:
                        max_act_task = np.max(taskTS_tmp[parcel,onsets[block]:offsets[block]])
                        min_act_task = np.min(taskTS_tmp[parcel,onsets[block]:offsets[block]])
                        if np.abs(max_act_task)-baseline_activity[parcel]>np.abs(min_act_task)-baseline_activity[parcel]:
                            tmp_task_mat[parcel,block] = max_act_task-baseline_activity[parcel]
                        else:
                            tmp_task_mat[parcel,block] = min_act_task-baseline_activity[parcel]


                        max_act_rest = np.max(restTS_tmp[parcel,onsets[block]:offsets[block]])
                        min_act_rest = np.min(restTS_tmp[parcel,onsets[block]:offsets[block]])
                        if np.abs(max_act_rest)-baseline_activity_rest[parcel]>np.abs(min_act_rest)-baseline_activity_rest[parcel]:
                            tmp_rest_mat[parcel,block] = max_act_rest-baseline_activity_rest[parcel]
                        else:
                            tmp_rest_mat[parcel,block] = min_act_rest-baseline_activity_rest[parcel]
                    except:
                        max_act_task = np.max(taskTS_tmp[parcel,onsets[block]:])
                        min_act_task = np.min(taskTS_tmp[parcel,onsets[block]:])
                        if np.abs(max_act_task)-baseline_activity[parcel]>np.abs(min_act_task)-baseline_activity[parcel]:
                            tmp_task_mat[parcel,block] = max_act_task-baseline_activity[parcel]
                        else:
                            tmp_task_mat[parcel,block] = min_act_task-baseline_activity[parcel]


                        max_act_rest = np.max(restTS_tmp[parcel,onsets[block]:])
                        min_act_rest = np.min(restTS_tmp[parcel,onsets[block]:])
                        if np.abs(max_act_rest)-baseline_activity_rest[parcel]>np.abs(min_act_rest)-baseline_activity_rest[parcel]:
                            tmp_rest_mat[parcel,block] = max_act_rest-baseline_activity_rest[parcel]
                        else:
                            tmp_rest_mat[parcel,block] = min_act_rest-baseline_activity_rest[parcel]


            # Shorten this so that it matches number of TPs of all other tasks
            #task_tp_ind = task_tp_ind[:min_tps]
            # Isolate data for the task
            # Estimate task FC
            taskFC[:,:,condcount+cond,s] = np.corrcoef(tmp_task_mat)
            restFC[:,:,condcount+cond,s] = np.corrcoef(tmp_rest_mat)
            taskblock_mats.extend(tmp_task_mat.T)
            restblock_mats.extend(tmp_rest_mat.T)
            np.fill_diagonal(taskFC[:,:,condcount+cond,s],0)
            np.fill_diagonal(restFC[:,:,condcount+cond,s],0)

            taskAct[:,condcount+cond,s] = np.mean(tmp_task_mat,axis=1) - baseline_activity 


        
        # Go to next run
        trcount += runlength
        # Go to next condition
        condcount += taskRegs.shape[1]

    taskGBC[:,:,s] = np.nanmean(taskFC[:,:,:,s],axis=1)
    restGBC[:,:,s] = np.nanmean(restFC[:,:,:,s],axis=1)
    # Correlate blocks across all tasks
    taskFC_alltasks[:,:,s] = np.corrcoef(np.asarray(taskblock_mats).T)
    restFC_alltasks[:,:,s] = np.corrcoef(np.asarray(restblock_mats).T)

# Temporary
taskGBC = np.mean(taskFC_alltasks,axis=1)
restGBC = np.mean(restFC_alltasks,axis=1)

taskVrest_GBC = taskGBC - restGBC
# Save out activity
t_act = np.abs(stats.ttest_1samp(taskAct,0,axis=2)[0])

####
np.savetxt('../../resultdata/connectivity/conn_taskVrest_gbc_peakActivity.csv',taskVrest_GBC,delimiter=',')
np.savetxt('../../data/results/activity/activity_peakActivity.csv',t_act,delimiter=',')
####

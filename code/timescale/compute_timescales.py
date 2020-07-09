import numpy as np
import sys
sys.path.append('../')
import utils.taskGLMPipeline as tgp
import utils.tools as tools
import os
import multiprocessing as mp
import scipy.stats as stats
os.environ['OMP_NUM_THREADS'] = str(1)
import sklearn
import nibabel as nib
import pandas as pd
import h5py
from importlib import reload
import matplotlib.pyplot as plt
import seaborn as sns
import scipy


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

model = '24pXaCompCorXVolterra'
zscore = False
FIR = False
nTasks = 24

#######################################################################################################
#### Load data
restTS = np.zeros((nParcels,4780,len(subjNums)))
scount = 0
for subj in subjNums:
    if scount%25==0: print('Loading in data for subject', scount+1, '/', len(subjNums))
    restTS[:,:,scount] = tools.loadRestResiduals(subj,model=model,zscore=zscore,FIR=FIR)
    scount += 1

task_timing = task_timing.astype(bool)
#######################################################################################################

def autocorr_decay(dk,A,tau,B):
    return A*(np.exp(-(dk/tau))+B)

#######################################################################################################
# Compute timescales with 100 lags
maxlag = 100
taus = np.zeros((nParcels,len(subjNums)))
As = np.zeros((nParcels,len(subjNums)))
Bs = np.zeros((nParcels,len(subjNums)))
xdata = np.arange(maxlag)
variability = np.zeros((nParcels,len(subjNums)))
for scount in range(len(subjNums)):
    if scount%10==0: print(scount, '/', len(subjNums))
    for roi in range(nParcels):
        tscale = np.correlate(restTS[roi,:,scount],restTS[roi,:,scount],mode='full')
        start = len(tscale)//2 + 1
        # Normalize
        tscale = np.divide(tscale,np.max(tscale))
        stop = start + maxlag
        tscale = tscale[start:stop]
        # Sometimes curve_fit cannot find a good solution, so if it can't, enter a 'nan' for this subject's ROI
        try:
            A, tau, B = scipy.optimize.curve_fit(autocorr_decay,xdata,tscale,p0=[0,np.random.rand(1)[0]+0.01,0],bounds=(([0,0,-np.inf],[np.inf,np.inf,np.inf])),method='trf')[0]
            As[roi,scount] = A
            taus[roi,scount] = tau
            Bs[roi,scount] = B
        except:
            print('Subject', scount, 'ROI', roi, 'NaN')
            As[roi,scount] = np.nan
            taus[roi,scount] = np.nan
            Bs[roi,scount] = np.nan
            

        
np.savetxt('../../resultdata/timescales/Murray_As_regions_autocorrelation_rest100.txt',As,delimiter=',')
np.savetxt('../../resultdata/timescales/Murray_taus_regions_autocorrelation_rest100.txt',taus,delimiter=',')
np.savetxt('../../resultdata/timescales/Murray_Bs_regions_autocorrelation_rest100.txt',Bs,delimiter=',')
#######################################################################################################

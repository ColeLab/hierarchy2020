# hierarchy2020
code base for:
Ito T, Hearne LJ, Cole MW (in press). "A cortical hierarchy of localized and distributed processes revealed via dissociation of task activations, connectivity changes, and intrinsic timescale." NeuroImage.

bioRxiv link: https://www.biorxiv.org/content/10.1101/262626v2

This is the initial code release. 

Please note that almost all code was run on a compute cluster at Rutgers (https://oarc.rutgers.edu/). Much of the code, outside of data visualization, was run on this cluster. My intention of this code release is to enable others to adapt and use the code for their needs, and not as a 'plug-and-play' package.

Feel free to e-mail me with questions or comments. In some cases, it may be easier for me to point to the relevant code/lines of interest.

Email: taku [dot] ito1 [at] gmail [dot] com (Taku Ito)

---

#### General organization

There are two main directories, as described below: `resultdata/` and `code/`

* `resultdata/` - contains generated outputs used for visualization. Outputs were generated on a compute cluster at Rutgers University (https://oarc.rutgers.edu) 
  * In general, this directory only exists to generate figures from the manuscript using the jupyter notebooks provided in `code/`
* `code/` - contains the code/scripts used to run analyses. There are two main types of codes: jupyter notebooks for figure visualization and statistical calculation reported in the paper, and scripts used to generate output data (e.g., FC, activation, timescale, activity flow outputs)
  * jupyter notebooks have suffixes with `*.ipynb`. In general, notebooks are labeled such that `Fig2*` contains visualizations for Figure 2, and so on. Note that notebooks with `s`, e.g., `Fig2s*.ipynb` are results/analyses for the replication cohort of subjects. Some of these visualizations/graphs are not reported in the paper, but are provided here.
  * `activation_analyses/` - code to save out task GLM activations for all tasks. Assumes data has been already saved out as .h5 files.
  * `activityflow/` - code to run activity flow analysis on preprocessed HCP data
  * `connectivity_analysis/` - code to run connectivity analyses, assuming FIR task regression has already been performed.
  * `nonparametric_activityXconn_analysis/` - code to run analyses for the peak-magnitude approach to estimating task activations nand task FC. This assumes that that time series data are nuisance regressed **without any task GLM**.
  * `timescale/` - code to run timescale calculation on resting-state HCP data. Assumes nuisance regressed time series data.



All relevant code for preprocessing can be found in this repository: https://github.com/ito-takuya/corrQuench/tree/master/glmScripts
* Any questions about this code please feel free to contact me with questions.
 



#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1]:


#@title ## Base imports
import os
import cmd
import sys
import json
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

import sklearn.metrics

import skimage
import skimage.io
import PIL
import PIL.Image
import requests

import IPython.display
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots

# Display versions of python packages
pip_versions = get_ipython().run_line_magic('system', 'pip freeze  # uses colab magic to get list from shell')
pip_versions_organized = {
    "standard": [pip_version for pip_version in pip_versions if "==" in pip_version],
    "other": [pip_version for pip_version in pip_versions if "==" not in pip_version]
    }
print(f"Python version: {sys.version} \n")  # display version of python itself (i.e. 3.8.10)
cli = cmd.Cmd()
cli.columnize(pip_versions_organized["standard"], displaywidth=800)
cli.columnize(pip_versions_organized["other"], displaywidth=160)


# In[2]:


colab_ip = get_ipython().run_line_magic('system', 'hostname -I   # uses colab magic to get list from bash')
colab_ip = colab_ip[0].strip()   # returns "172.28.0.12"
colab_port = 9000                # could use 6000, 8080, or 9000

notebook_filename = filename = requests.get(f"http://{colab_ip}:{colab_port}/api/sessions").json()[0]["name"]

# Avoids scroll-in-the-scroll in the entire Notebook
def resize_colab_cell():
  display(IPython.display.Javascript('google.colab.output.setIframeHeight(0, true, {maxHeight: 10000})'))
get_ipython().events.register('pre_run_cell', resize_colab_cell)


#@markdown ### func `def get_path_to_save(...):`
def get_path_to_save(plot_props:dict=None, file_prefix="", save_filename:str=None, save_in_subfolder:str=None, extension="jpg", dot=".", create_folder_if_necessary=True):
    """
    Code created myself (Rahul Yerrabelli)
    """
    replace_characters = {
        "$": "",
        "\\frac":"",
        "\\mathrm":"",
        "\\left(":"(",
        "\\right)":")",
        "\\left[":"[",
        "\\right]":"]",
        "\\": "",
        "/":"-",
        "{": "(",
        "}": ")",
        "<":"",
        ">":"",
        "?":"",
        "_":"",
        "^":"",
        "*":"",
        "!":"",
        ":":"-",
        "|":"-",
        ".":"_",
    }

    # define save_filename based on plot_props
    if save_filename is None:
        save_filename = "unnamed"

    #save_path = f"../outputs/{notebook_filename.split('.',1)[0]}"
    save_path = [
                 "outputs",
                f"{notebook_filename.split('.',1)[0]}",
                ]
    if save_in_subfolder is not None:
        if isinstance(save_in_subfolder, (list, tuple, set, np.ndarray) ):
            save_path.append(**save_in_subfolder)
        else:  # should be a string then
            save_path.append(save_in_subfolder)
    save_path = os.path.join(*save_path)

    if not os.path.exists(save_path) and create_folder_if_necessary:
        os.makedirs(save_path)
    return os.path.join(save_path, file_prefix+save_filename+dot+extension)
    #plt.savefig(os.path.join(save_path, save_filename+dot+extension))


# In[3]:


#@title ## Mount google drive and import my code

mountpoint_folder_name = "drive"  # can be anything, doesn't have to be "drive"
project_path_within_drive = "PythonProjects/ECV-Analysis" #@param {type:"string"}
#project_path_within_drive = "UIUC ECs/Rahul_Ashkhan_Projects/SpeculumProjects_Shared/Analysis" #@param {type:"string"}
project_path_full = os.path.join("/content/",mountpoint_folder_name,
                        "MyDrive",project_path_within_drive)

get_ipython().run_line_magic('cd', '{project_path_full}')


# In[4]:


try:
    import google.colab.drive
    import os, sys
    # Need to move out of google drive directory if going to remount
    get_ipython().run_line_magic('cd', '')
    # drive.mount documentation can be accessed via: drive.mount?
    #Signature: drive.mount(mountpoint, force_remount=False, timeout_ms=120000, use_metadata_server=False)
    google.colab.drive.mount(os.path.join("/content/",mountpoint_folder_name), force_remount=True)  # mounts to a folder called mountpoint_folder_name

    if project_path_full not in sys.path:
        pass
        #sys.path.insert(0,project_path_full)
    get_ipython().run_line_magic('cd', '{project_path_full}')

except ModuleNotFoundError:  # in case not run in Google colab
    import traceback
    traceback.print_exc()


# In[5]:


def convert_wga_to_total_days(ga, return_separately=False):  # convert "37w 3d" to 262
    """
    "20w 2d" -> 142
    "20w2d" -> 142
    "20w" -> 140
    "5w" -> 35
    """
    if isinstance(ga, str):
        assert "w" in ga
        wks_str, days_str = ga.split("w",maxsplit=1)
        wks = int(wks_str)
        days_str = days_str.strip()
        if days_str.endswith("d"):
            days_str = days_str[:-1]
        if days_str == "":
            days = 0
        else:
            days = int(days_str)

        if return_separately:
            return (wks,days)
        else:
            return wks*7+days
    else:
        return None

def convert_total_days_to_wga(total_days):
    days = total_days % 7
    wks = round((total_days - days)/7)   # shouldn't need to round, but used to convert float to int
    wga = f"{wks:g}w{days}d"
    return wga


# ## Calculating EFW at a specific time using prior ultrasound
# 
# How to use coefficients
# $\ln(EFW)= b_0 + b_1 \cdot t +b_2 \cdot t^2 +b_3 \cdot t^3 +b_4 \cdot t^4$ where $t$ is GA in weeks.
# For percentiles not described, you can use interpolation. For example, take a percentile $\alpha$ surrounded by two defined percentiles ie $\alpha_0<\alpha<\alpha_1$. Then:  
# $\ln(EFW_\alpha)=\frac{\ln(EFW_{\alpha_1})-\ln(EFW_{\alpha_0})}{\alpha_1-\alpha_0} \cdot (\alpha-\alpha_0) + \ln(EFW_{\alpha_0})$

# In[6]:


def calculate_efw(ga_wks, sex, quantile):
    from numpy.polynomial import Polynomial
    from numpy.polynomial.polynomial import polyval

    # sex input can already be "M" and "F" or can be 1 vs 2 (this numeric system matches the other spreadsheet)
    if sex==1:
        sex="F"
    elif sex==2:
        sex="M"
    if ga_wks is None or sex is None or quantile is None or np.isnan(ga_wks) or np.isnan(quantile):  # will throw error if check if sex (str) is np.isnan
        return np.nan
    if ga_wks < 14 or ga_wks > 50:
        raise ValueError("ga_wks appears to be in days instead of weeks")
    unique_quantiles = efw_who_coeffs_df.index.levels[1]
    if quantile in unique_quantiles:
        #b0,b1,b2,b3,b4 = efw_who_coeffs_df.loc[(sex,quantile)]
        #ln_b0 + b1*t + b2*t**2 +b3*t**3 +b4*t**4
        ln_efw = np.polynomial.polynomial.polyval(ga_wks, efw_who_coeffs_df.loc[(sex,quantile)])
        efw =np.exp(ln_efw)
        return efw
    elif quantile < np.min(unique_quantiles) or quantile > np.max(unique_quantiles):
        return np.nan   # do not extrapolate if too extreme eg <0.001%
    else:
        closest_quantile_lower = unique_quantiles[unique_quantiles <= quantile].max()
        closest_quantile_upper = unique_quantiles[unique_quantiles > quantile].min()
        try:
            ln_efw_lower = np.polynomial.polynomial.polyval(ga_wks, efw_who_coeffs_df.loc[(sex,closest_quantile_lower)])
        except KeyError:
            print(f"KeyError for {(sex,closest_quantile_lower)}. (ga_wks, sex, quantile) = ({ga_wks}, {sex}, {quantile}).")
            return np.nan
        try:
            ln_efw_upper = np.polynomial.polynomial.polyval(ga_wks, efw_who_coeffs_df.loc[(sex,closest_quantile_upper)])
        except KeyError:
            print(f"KeyError for {(sex,closest_quantile_lower)}. (ga_wks, sex, quantile) = ({ga_wks}, {sex}, {quantile})")
            return np.nan
        ln_efw = (ln_efw_upper-ln_efw_lower)/(closest_quantile_upper-closest_quantile_lower) * (quantile-closest_quantile_lower) + ln_efw_lower
        efw= np.exp(ln_efw)
        return efw



# # Data processing

# ## Get raw data

# In[7]:


computerized_records_df = pd.read_csv("data/01_raw/Computerized_data_2024-03-13_1752.csv", index_col=0,
                                      converters={
                                          #"ecv_successful_1":bool  # has to be float (default) to allow NaN
                                      })
manual_records_df = pd.read_excel("data/01_raw/Manual_USData_v14.xlsx", index_col=0,
                                  converters={
                                      "Skip":str,
                                      "ECV to Delivery (days)":int,  # some are floats like 10.6, will get rounded down
                                      "Delivery GA": convert_wga_to_total_days,
                                      })
# Below file can also be found online at
## https://github.com/jcarvalho45/whoFetalGrowth/blob/main/coefficientsEFWbySexV3.csv
## https://srhr.org/fetalgrowthcalculator/#/
## Kiserud 2017. The World Health Organization Fetal Growth Charts: A Multinational Longitudinal Study of Ultrasound Biometric Measurements and Estimated Fetal Weight. https://doi.org/10.1371/journal.pmed.1002220
## Kiserud 2018. The World Health Organization fetal growth charts: concept, findings, interpretation, and application. https://doi.org/10.1016/j.ajog.2017.12.010

efw_who_coeffs_df = pd.read_csv("data/01_raw/WHOcoefficientsEFWbySexV3.csv", index_col=[0,1])

manual_records_df


# ## Combine raw data and process it

# In[8]:


all_records_by_pt_df=pd.merge(left=manual_records_df, right=computerized_records_df, on="subject_id")
# Filter out columns marked to be skipped (eg ECV wasn't actually done)
all_records_by_pt_df = all_records_by_pt_df[all_records_by_pt_df["Skip"]!="Yes"]

# These five columns were empty for every pt, so will just remove them
all_records_by_pt_df = all_records_by_pt_df.drop(columns=["marijuana_use","cocaine_use","amphetamines_use","opiates_use","substance_use_complete"])
# "Skip" and "include_pregnancy" mean the same, so can remove one of them
all_records_by_pt_df = all_records_by_pt_df.drop(columns=["include_pregnancy"])


# The computer only counted up to 3 ECVs in a patient. However, one patient (Pt #45) had four ECVs. The fourth one was a failure (unlike the first 3). Here I add another pair of columns for the 4th ECV and mark it as a ECV failure
all_records_by_pt_df.insert(all_records_by_pt_df.columns.get_loc("ecv_successful_3")+2, "ecv_successful_4", np.nan)
all_records_by_pt_df.insert(all_records_by_pt_df.columns.get_loc("ecv_fetal_abnormal_3")+2, "ecv_fetal_abnormal_4", np.nan)
all_records_by_pt_df.loc[45, "ecv_successful_4"] = 0

# For pt 102, after initial ECV, turned back within 2 min, and second ECV was done successfully. Few hours later, midwife noticed return to breech, and third ECV was done. No further turns or ECVs were documented. Vaginal delivery went smoothly.
# Computer records only counted the first ECV. Will add the remaining ECVs as successful
all_records_by_pt_df.loc[102, "ecv_successful_2"] = 1
all_records_by_pt_df.loc[102, "ecv_successful_3"] = 1


# Do same basic processing
proc_records_by_pt_df = all_records_by_pt_df.copy()

# For some patients, we have the presentation immediately before the ECV. Others, we only have the presentation on the last US. These are both in separate columns. We will combine them into one called "Presentation US"
proc_records_by_pt_df.insert(proc_records_by_pt_df.columns.get_loc("Presentation Pre-ECV"), "Presentation", proc_records_by_pt_df["Presentation Pre-ECV"])
proc_records_by_pt_df["Presentation"].fillna(proc_records_by_pt_df["Presentation US"], inplace=True)
# The presentation description will be specific eg "Breech, complete; vertex on maternal L". We will drop the specifics by splitting at the ";"
proc_records_by_pt_df["Presentation"] = proc_records_by_pt_df["Presentation"].str.split(pat = ";", expand=True)[0]
# We will no split up the brief and detail presentation info (eg "Breech" and "complete") into separate columns
#proc_records_by_pt_df[["Presentation Brief", "Presentation Detail"]] = proc_records_by_pt_df["Presentation"].str.split(pat = ",", expand=True)
proc_records_by_pt_df.insert(proc_records_by_pt_df.columns.get_loc("Presentation"), "Presentation Brief", proc_records_by_pt_df["Presentation"].str.split(pat = ",", expand=True)[0].str.strip())
proc_records_by_pt_df.insert(proc_records_by_pt_df.columns.get_loc("Presentation"), "Presentation Detail", proc_records_by_pt_df["Presentation"].str.split(pat = ",", expand=True)[1].str.strip())
proc_records_by_pt_df = proc_records_by_pt_df.drop(columns=["Presentation","Presentation Pre-ECV","Presentation US"])

# Use available information to calculate GA at ECV
proc_records_by_pt_df.insert(proc_records_by_pt_df.columns.get_loc("ECV to Delivery (days)"), "ECV GA", proc_records_by_pt_df["Delivery GA"]-proc_records_by_pt_df["ECV to Delivery (days)"])
# Use the EFW % from the last US to calculat the EFW at the time of ECV and delivery
# There are 5 data points that an "US EFW (g)" recorded, but not an "US EFW (%)". Ignore these as these are from old ultrasounds anyway (when EFW <1000g)
proc_records_by_pt_df.insert(proc_records_by_pt_df.columns.get_loc("US EFW (%)")+1, "ECV EFW (g)",
                             proc_records_by_pt_df.apply(lambda row: calculate_efw(row["ECV GA"]/7, row["baby_gender_1"], row["US EFW (%)"]), axis=1))
proc_records_by_pt_df.insert(proc_records_by_pt_df.columns.get_loc("US EFW (%)")+2, "Delivery EFW (g)",
                             proc_records_by_pt_df.apply(lambda row: calculate_efw(row["Delivery GA"]/7, row["baby_gender_1"], row["US EFW (%)"]), axis=1))
# Alternatively, do it by iterating through df
#for index, row in proc_records_by_pt_df.iterrows():
#    (calculate_efw(row["ECV GA"]/7, row["baby_gender_1"], row["US EFW (%)"]))

# Some rows do not have an AFI. Estimate it from SDP
proc_records_by_pt_df.insert(proc_records_by_pt_df.columns.get_loc("AFI"), "AFI equiv", proc_records_by_pt_df["AFI"])
# Most values of SDP are in format like "3.97 x 2.75", although some are just "3.97". Split will return na if the pattern "x" is not found
proc_records_by_pt_df.insert(proc_records_by_pt_df.columns.get_loc("SDP"), "SDP1", proc_records_by_pt_df["SDP"].str.split(pat="x", n=1, expand=True)[0].str.strip())
proc_records_by_pt_df["SDP1"].fillna(proc_records_by_pt_df["SDP"], inplace=True)
def SDP_to_AFI(sdp):
    return sdp * 3
proc_records_by_pt_df["AFI equiv"].fillna(SDP_to_AFI(pd.to_numeric(proc_records_by_pt_df["SDP1"])), inplace=True)
proc_records_by_pt_df.drop(columns=["SDP"])

# Some rows do not have a delivery BMI, only initial. Fortunately, all but pt 3 have an initial BMI (pt 3 has no BMIs or weights)
# Create a "last bmi" column. If no other BMI is available, this will be the first BMI.
proc_records_by_pt_df.insert(proc_records_by_pt_df.columns.get_loc("delivery_bmi"), "last_bmi", proc_records_by_pt_df["delivery_bmi"])
proc_records_by_pt_df["last_bmi"].fillna(proc_records_by_pt_df["first_bmi"], inplace=True)

# number of ECVs done/attempted
proc_records_by_pt_df["ecv_tries"] = proc_records_by_pt_df[["ecv_successful_1","ecv_successful_2","ecv_successful_3","ecv_successful_4"]].notna().sum(axis=1)


# ## Get df with data by ECV attempt

# In[9]:


ecv_num_options = "1234"

# make a dataframe where each row is an ECV, not just a patient
proc_records_by_ecv_df = pd.melt(proc_records_by_pt_df.reset_index(),  # need to reset index so that subject_id can be access
                                 id_vars=["subject_id"],
                                 value_vars=[f"ecv_successful_{ecv_num}" for ecv_num in ecv_num_options],
                                 var_name="ecv_num", value_name="ecv_successful"
                                 ).dropna(subset="ecv_successful")

# convert ecv_num values from "ecv_successful_x" to "ecv_successful_x" where x is in {1,2,3,4}
proc_records_by_ecv_df["ecv_num"] = proc_records_by_ecv_df["ecv_num"].str.split(pat = "_", expand=True)[2].str.strip().astype(int)
# Get all the other values from the original dataframe from the subject_id
proc_records_by_ecv_df = pd.merge(proc_records_by_ecv_df, proc_records_by_pt_df.reset_index(), on = "subject_id")
# Drop values that are specific to the ecv (except for ecv_num)
proc_records_by_ecv_df = proc_records_by_ecv_df.drop(columns = [col for col in proc_records_by_ecv_df.columns if col.lower().startswith("ecv_") and col.split("_")[-1] in ecv_num_options])
# Put back subject_id into the index as well as ecv_num
proc_records_by_ecv_df = proc_records_by_ecv_df.set_index(["subject_id","ecv_num"]).sort_index()


# ## Exportable ECV file

# In[ ]:


proc_records_by_pt_df.head()


# In[13]:


# Number of unique values, number of non-null values, and total size
col_cts=proc_records_by_pt_df.agg(["nunique","count","size"])
with pd.option_context("display.max_columns", None):
    display(col_cts)


# In[14]:


unique_value_ct_by_col=all_records_by_pt_df.nunique()
cols_without_unique_values = unique_value_ct_by_col[unique_value_ct_by_col<=1]
display(cols_without_unique_values)


# In[18]:


exportable_records_by_pt["age_delivery"] // 5


# In[24]:


max_prev_preg = "12345"

exportable_records_by_pt = proc_records_by_pt_df.drop(columns=[
    "Skip","ECV Done?",  # all ECVs not to be counted were already removed
    *[f"ecv_fetal_abnormal_{n}" for n in ecv_num_options],  # this value was not consistently reported enough to be useful
    "ecv_complete","pregnancy_episode_delivery_complete", # unsure what these columns meant, but all values are the same
    "baby_status_1", "baby_status_2", "baby_gender_2",  # twins excluded, baby status meant to be FHT abnormalities but wasn't reported enough to be useful
    "chronic_htn_before","chronic_htn_during","dm1_before","dm1_during","dm2_before","dm2_during","gestational_dm_during","preeclampsia_during", # HTN and diabetes values were not consistently reported enough to be useful
    "ultrasounds_complete", # indicates number of ultrasounds in pregnancy, not really useful
    "tobacco_use", "alcohol_use", "ill_drug_use", # this value was not consistently reported enough to be useful
    "SDP1", # redundant with SDP
    *[f"gravida_prev_preg_{n}" for n in max_prev_preg], # not useful as Gs and Ps for this pregnancy are already included
    *[f"para_prev_preg_{n}" for n in max_prev_preg], # not useful as Gs and Ps for this pregnancy are already included
    *[f"term_prev_preg_{n}" for n in max_prev_preg], # not useful as Gs and Ps for this pregnancy are already included
    *[f"preterm_prev_preg_{n}" for n in max_prev_preg], # not useful as Gs and Ps for this pregnancy are already included
    ])
exportable_records_by_pt = exportable_records_by_pt.rename(columns={
    "gravida_this_pregnancy":"gravida",
    "para_this_pregnancy":"para",
    "term_this_pregnancy":"term_preg_ct",
    "preterm_this_pregnancy":"preterm_preg_ct",
    "baby_gender_1":"baby_gender",
    "language_c":"language",
    })
# Convert age to 5 year intervals (eg 33 to 30-34)
exportable_records_by_pt.insert(exportable_records_by_pt.columns.get_loc("age_delivery"),
                                "mat_age_delivery",
                                (exportable_records_by_pt["age_delivery"] // 5)*5
                                )
exportable_records_by_pt["mat_age_delivery"] = exportable_records_by_pt["mat_age_delivery"].astype(str) + "-" + (exportable_records_by_pt["mat_age_delivery"]+4).astype(str)
exportable_records_by_pt = exportable_records_by_pt.drop(columns=["age_delivery"])


with pd.option_context("display.max_columns", None):
    display(exportable_records_by_pt)


# ### Save processed dfs for future running

# In[25]:


all_records_by_pt_df.to_csv(  "data/02_processed/all_records_by_pt_df"+".csv")
all_records_by_pt_df.to_excel("data/02_processed/all_records_by_pt_df"+".xlsx")
all_records_by_pt_df.to_pickle("data/02_processed/all_records_by_pt_df"+".pkl")

proc_records_by_pt_df.to_csv(  "data/02_processed/proc_records_by_pt_df"+".csv")
proc_records_by_pt_df.to_excel("data/02_processed/proc_records_by_pt_df"+".xlsx")
proc_records_by_pt_df.to_pickle("data/02_processed/proc_records_by_pt_df"+".pkl")

proc_records_by_ecv_df.to_csv(  "data/02_processed/proc_records_by_ecv_df"+".csv")
proc_records_by_ecv_df.to_excel("data/02_processed/proc_records_by_ecv_df"+".xlsx")
proc_records_by_ecv_df.to_pickle("data/02_processed/proc_records_by_ecv_df"+".pkl")

exportable_records_by_pt.to_csv(  "data/02_processed/exportable_records_by_pt"+".csv")
exportable_records_by_pt.to_excel("data/02_processed/exportable_records_by_pt"+".xlsx")
exportable_records_by_pt.to_pickle("data/02_processed/exportable_records_by_pt"+".pkl")


# In[ ]:





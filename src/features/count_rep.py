import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error
import math
pd.options.mode.chained_assignment=None
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"]=(20,5)
plt.rcParams["figure.dpi"]=100
plt.rcParams["lines.linewidth"]=2

df=pd.read_pickle("././data/interim/01_Data_Processed.pkl")
df=df[df["Label"]!="rest"]
acc_r=df["Accelerometer_x"]**2+df["Accelerometer_y"]**2+df["Accelerometer_z"]**2
gyro_r=df["Gyroscope_x"]**2+df["Gyroscope_y"]**2+df["Gyroscope_z"]**2
df["Accelerometer_r"]=np.sqrt(acc_r)
df["Gyroscope_r"]=np.sqrt(gyro_r)
bench_df=df[df["Label"]=="bench"]
squat_df=df[df["Label"]=="squat"]
row_df=df[df["Label"]=="row"]
ohp_df=df[df["Label"]=="ohp"]
dead_df=df[df["Label"]=="dead"]
fs=1000/200
LowPass=LowPassFilter()

bench_set=bench_df[bench_df["Set"]==bench_df["Set"].unique()[0]]
squat_set=squat_df[squat_df["Set"]==squat_df["Set"].unique()[0]]
row_set=row_df[row_df["Set"]==row_df["Set"].unique()[0]]
ohp_set=ohp_df[ohp_df["Set"]==ohp_df["Set"].unique()[0]]
dead_set=dead_df[dead_df["Set"]==dead_df["Set"].unique()[0]]

def count(df,cutoff=0.4,order=10,column="Accelerometer_r"):
    data=LowPass.low_pass_filter(df,column,fs,cutoff,order)
    indexes=argrelextrema(data[column+"_lowpass"].values,np.greater)
    peeks=data.iloc[indexes]
    fig,ax=plt.subplots()
    plt.plot(df[f"{column}_lowpass"])
    plt.plot(peeks[f"{column}_lowpass"],"o",color="red")
    ax.set_ylabel(f"{column}_lowpass")
    exercise=df["Label"].iloc[0].title()
    category=df["Category"].iloc[0].title()
    plt.title(f"{category} {exercise}: {len(peeks)} Reps")
    plt.savefig(f"./reports/figures/Count_rep_Figs/{category}_{exercise}_{len(peeks)}_Reps.png")
    return len(peeks)


count(bench_set,cutoff=0.4)
count(squat_set,cutoff=0.35)
count(row_set,cutoff=0.65,column="Gyroscope_x")
count(ohp_set,cutoff=0.35)
count(dead_set,cutoff=0.4)

df["Reps"]=df["Category"].apply(lambda x:5 if x == "heavy" else 10)
rep_df=df.groupby(["Label","Category","Set"])["Reps"].max().reset_index()
rep_df["Reps_pred"]=0

for s in df["Set"].unique():
    subset=df[df["Set"]==s]
    column="Accelerometer_r"
    cutoff=0.4
    if subset["Label"].iloc[0] =="squat":
        cutoff=0.35
    if subset["Label"].iloc[0] =="row":
        cutoff=0.65
    if subset["Label"].iloc[0] =="ohp":
        cutoff=0.35

    reps=count(subset,cutoff,10,column)
    rep_df.loc[rep_df["Set"]==s,"Reps_pred"]=reps

    
error=mean_absolute_error(rep_df["Reps"],rep_df["Reps_pred"]).round(2)
print(error)
rep_df.groupby(["Label","Category"])[["Reps","Reps_pred"]].mean().plot.bar().savefig(f"././reports/figures/Count_rep_Figs/Error_Reps.png")

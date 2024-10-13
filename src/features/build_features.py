import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from DataTransformation import LowPassFilter,PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"]=(20,5)
plt.rcParams["figure.dpi"]=100
plt.rcParams["lines.linewidth"]=2

def main(input_data_path,output_data_path):
    df=pd.read_pickle(input_data_path)
    pridictor_col=list(df.columns[:6])

    for col in pridictor_col:
        df[col] = df[col].interpolate()

    for s in df["Set"].unique():
        duration=df[df["Set"]==s].index[-1] - df[df["Set"]==s].index[0]
        df.loc[(df["Set"]==s),"Duration"]=duration.seconds

    df_lowpass=df.copy()

    LowPass=LowPassFilter()
    fs=1000/200
    cutoff=1.2


    # df_lowpass=LowPass.low_pass_filter(df_lowpass,"Accelerometer_y",fs,cutoff,order=5)
    # subset=df_lowpass[df_lowpass["Set"]==45]
    # fig,ax=plt.subplots(nrows=2,sharex=True,figsize=(20,10))
    # ax[0].plot(subset["Accelerometer_y"].reset_index(drop=True),label="raw data")
    # ax[1].plot(subset["Accelerometer_y_lowpass"].reset_index(drop=True),label="butterworth data") 
    # ax[0].legend(loc="upper center",bbox_to_anchor=(0.5,1.15),fancybox=True,shadow=True)
    # ax[1].legend(loc="upper center",bbox_to_anchor=(0.5,1.15),fancybox=True,shadow=True)  


    # Make smooth graph at all col
    for col in pridictor_col:
        df_lowpass=LowPass.low_pass_filter(df_lowpass,col,fs,cutoff)
        df_lowpass[col]=df_lowpass[col+"_lowpass"]
        del df_lowpass[col+"_lowpass"]

    df_pca=df_lowpass.copy()
    PCA=PrincipalComponentAnalysis()
    pc_values=PCA.determine_pc_explained_variance(df_pca,pridictor_col)
    df_pca=PCA.apply_pca(df_pca,pridictor_col,3)

    df_squared=df_pca.copy()
    acc_r=df_squared["Accelerometer_x"]**2+df_squared["Accelerometer_y"]**2+df_squared["Accelerometer_z"]**2
    gyro_r=df_squared["Gyroscope_x"]**2+df_squared["Gyroscope_y"]**2+df_squared["Gyroscope_z"]**2
    df_squared["Accelerometer_r"]=np.sqrt(acc_r)
    df_squared["Gyroscope_r"]=np.sqrt(gyro_r)

    df_temporal=df_squared.copy()
    NumAbs=NumericalAbstraction()

    pridictor_col=pridictor_col+["Accelerometer_r","Gyroscope_r"]

    ws=int(1000/200)

    df_temporal_list=[]


    for s in df_temporal["Set"].unique():
        subset=df_temporal[df_temporal["Set"]==s].copy()
        for col in pridictor_col:
            subset=NumAbs.abstract_numerical(subset,[col],ws,"mean")
            subset=NumAbs.abstract_numerical(subset,[col],ws,"std")
        df_temporal_list.append(subset)

    df_temporal=pd.concat(df_temporal_list)

    df_frq=df_temporal.copy().reset_index()
    FrqAbs=FourierTransformation()

    fs= int(1000/200)
    ws=int(2800/200)


    df_frq_list=[]
    for s in df_frq["Set"].unique():
        subset=df_frq[df_frq["Set"]==s].reset_index(drop=True).copy()
        subset=FrqAbs.abstract_frequency(subset,pridictor_col,ws,fs)
        df_frq_list.append(subset)

    df_frq=pd.concat(df_frq_list).set_index("epoch (ms)",drop=True)
    df_frq=df_frq.dropna()
    df_frq=df_frq.iloc[::2]

    df_cluster=df_frq.copy()
    cluster_col=["Accelerometer_x","Accelerometer_y","Accelerometer_z"]
    kmeans=KMeans(n_clusters=5,n_init=20,random_state=0)
    subset=df_cluster[cluster_col]
    df_cluster["Cluster"]=kmeans.fit_predict(subset)

    df_cluster.to_pickle(output_data_path)
    print(f"Work Done ! \nCheck on {output_data_path}")


if __name__ == '__main__':
    input_data_path = "././data/interim/02_Outlier_Removed_Data.pkl"
    output_data_path="././data/interim/03_Featured_Data.pkl"
    main(input_data_path,output_data_path)

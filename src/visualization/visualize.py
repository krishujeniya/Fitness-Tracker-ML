import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"]=(20,5)
plt.rcParams["figure.dpi"]=100
plt.rcParams["lines.linewidth"]=2
def main(input_data_path):
    """
    Create reports on visualized data into png
    """
    df=pd.read_pickle(input_data_path)

    # df_label_unique=df["Label"].unique()
    # for label in df_label_unique:
    #     subset = df[df["Label"]==label]
    #     plt.plot(subset["Accelerometer_y"].reset_index(drop=True),label=label)
    #     plt.show()



    # catogory_df=df.query("Label=='squat'").query("Participants=='A'").reset_index()

    # fig,ax=plt.subplots()
    # catogory_df.groupby(["Category"])["Accelerometer_y"].plot()
    # ax.set_ylabel("Accelerometer")
    # ax.set_xlabel("samples")
    # plt.legend()



    # Participants_df=df.query("Label=='bench'").sort_values("Participants").reset_index()

    # fig,ax=plt.subplots()
    # Participants_df.groupby(["Participants"])["Accelerometer_y"].plot()
    # ax.set_ylabel("Accelerometer")
    # ax.set_xlabel("samples")
    # plt.legend()




    # label="squat"
    # participants="A"
    # all_axis_df=df.query(f"Label=='{label}'").query(f"Participants=='{participants}'").reset_index()

    # fig,ax=plt.subplots()
    # all_axis_df[["Accelerometer_x","Accelerometer_y","Accelerometer_z"]].plot(ax=ax)
    # ax.set_ylabel("Accelerometer")
    # ax.set_xlabel("samples")
    # plt.legend()




    # labels=df["Label"].unique()
    # participants=df["Participants"].unique()

    # for label in labels:
    #     for participant in participants:
    #         all_axis_df=df.query(f"Label=='{label}'").query(f"Participants=='{participant}'").reset_index()


    #         if len(all_axis_df)!=0:
    #             fig,ax=plt.subplots()
    #             all_axis_df[["Accelerometer_x","Accelerometer_y","Accelerometer_z"]].plot(ax=ax)
    #             ax.set_ylabel("Accelerometer")
    #             ax.set_xlabel("samples")
    #             plt.title(f"{label}({participant})".title())
    #             plt.legend()


    # labels=df["Label"].unique()
    # participants=df["Participants"].unique()

    # for label in labels:
    #     for participant in participants:
    #         all_axis_df=df.query(f"Label=='{label}'").query(f"Participants=='{participant}'").reset_index()


    #         if len(all_axis_df)!=0:
    #             fig,ax=plt.subplots()
    #             all_axis_df[["Gyroscope_x","Gyroscope_y","Gyroscope_z"]].plot(ax=ax)
    #             ax.set_ylabel("Gyroscope")
    #             ax.set_xlabel("samples")
    #             plt.title(f"{label}({participant})".title())
    #             plt.legend()







    labels=df["Label"].unique()
    participants=df["Participants"].unique()

    for label in labels:
        for participant in participants:
            com_plot_df=df.query(f"Label=='{label}'").query(f"Participants=='{participant}'").reset_index()


            if len(com_plot_df)!=0:

                fig,ax=plt.subplots(nrows=2,sharex=True)
                com_plot_df[["Accelerometer_x","Accelerometer_y","Accelerometer_z"]].plot(ax=ax[0])
                com_plot_df[["Gyroscope_x","Gyroscope_y","Gyroscope_z"]].plot(ax=ax[1])
                ax[1].set_xlabel("samples")
                ax[0].legend()
                ax[1].legend()    
                plt.savefig(f"./reports/figures/visualize_Figs/{label.title()}({participant}).png")
                # plt.show()
    print(f"Work Done ! \nCheck on ./reports/figures/visualize_Figs")


if __name__ == '__main__':
    input_data_path = "././data/interim/01_Data_Processed.pkl"
    main(input_data_path)

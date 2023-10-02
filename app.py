# Libraries
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


#-----------style----------

# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



st.markdown("""
<style>
.heading
         {
    font-size:80px !important;
    color: #f2180c;
}
</style>

""",  unsafe_allow_html=True)

# disabling warning
st.set_option('deprecation.showPyplotGlobalUse', False)


#------------------------------------------------
#heading
st.write(
"""<p class="heading"> Biogas Production.</p>

    """, unsafe_allow_html=True
    )
st.write("---")


st.header("Biogas Production")

df = pd.read_csv("biogasdf_ready.csv", parse_dates=['date'])
st.write(df.head())

col1, col2, col3 = st.columns(3)
with col1:
    st.write("Features in the dataset")
    st.write(list(df.columns))


bio1 = df[df['bio_id']==1]
bio2 = df[df['bio_id']==2]
phase_test0 = df[df['phase_test']==0]
phase_test1 = df[df['phase_test']==1]
phase_test2 = df[df['phase_test']==2]
phase_test3 = df[df['phase_test']==3]


with col2:
    st.write("There are two type of bio production")
    st.write(f"Bio 1 : {bio1.shape[0]}")
    st.write(f"Bio 2 : {bio2.shape[0]}")  

    st.write("There are 4 different types of Phases in the production") 
    st.write(f"PHASE TEST 0 : {phase_test0.shape[0]} Entries")
    st.write(f"PHASE TEST 1 : {phase_test1.shape[0]} Entries")
    st.write(f"PHASE TEST 2 : {phase_test2.shape[0]} Entries")
    st.write(f"PHASE TEST 3 : {phase_test3.shape[0]} Entries")

with col3:
    st.write("Gas production categories")
    st.write("## ")
    st.write(df['dm3_gas'].value_counts()) 
    st.write("##")
    st.write("0: There is no production")
    st.write("0.01: There was production")   
   

col1, col2, col3 = st.columns(3)
with col1:
    sns.countplot(x = df['dm3_gas'], hue=df['dm3_gas'])
    st.pyplot()
with col2:
    sns.countplot(x=df['phase_test'], hue=df['phase_test'])      
    st.pyplot()
with col3:
    sns.countplot(x=df['bio_id'], hue = df['bio_id'])    
    st.pyplot()
# Peak  points
prod0= df[df['dm3_gas']==0]
prod1 = df[df['dm3_gas']==0.01]

#max
peak_points0 = prod0.max()
peak_points1 = prod1.max()

#min
low_points0 = prod0.min()
low_points1 = prod1.min()


col1, col2 = st.columns(2)
with col1:
    st.write("Peak Points of Dependent Features in prod0:")
    st.write(peak_points0[['fluid_temp', 'ground_temp', 'air_umidity', 'air_temp', 'gas_umidity', 'gas_temp', 'dm3_gas']])
    st.write("##")
    st.write("Peak Points of Dependent Features in prod1:")
    st.write(peak_points1[['fluid_temp', 'ground_temp', 'air_umidity', 'air_temp', 'gas_umidity', 'gas_temp', 'dm3_gas']])

with col2:
    #prod0
    st.write("low Points of Dependent Features in prod0:")
    st.write(low_points0[['fluid_temp', 'ground_temp', 'air_umidity', 'air_temp', 'gas_umidity', 'gas_temp', 'dm3_gas']])
    st.write("##")
    st.write("low Points of Dependent Features in prod1:")
    st.write(low_points1[['fluid_temp', 'ground_temp', 'air_umidity', 'air_temp', 'gas_umidity', 'gas_temp', 'dm3_gas']])



#_________________VISUALS_______________________
df_a_temp = pd.read_csv("df_a_temp.csv")
df_gr_temp = pd.read_csv("df_gr_temp.csv")
df_f_temp = pd.read_csv("df_f_temp.csv")
df_g_temp = pd.read_csv("df_g_temp.csv")








col1, col2 = st.columns(2)
with col1:
    # Air Temperature
    

    # Plot settings
    plt.figure(figsize=(12, 5))
    plt.xticks(np.arange(len(df_a_temp)), [f"{start:.2f}-{end:.2f}" for start, 
                                        end in zip(df_a_temp["air_temp_start"], df_a_temp["air_temp_end"])], rotation=90)
    plt.title("Change in gas volume in relation to Air temperature change")
    plt.grid()

    # Plotting the data
    plt.plot(df_a_temp.index, df_a_temp["dm3_gas"], marker='o', linestyle='-', color='b')
    # Adding labels and legend
    plt.xlabel("Binned Air Temperature")
    plt.ylabel("Gas Volume (dm3_gas)")
    plt.legend(["Gas Volume"], loc="upper left")

    # Show the plot
    plt.tight_layout()
    st.pyplot()

#
# Plot settings
with col2:
    plt.figure(figsize=(12, 5))
    plt.xticks(np.arange(len(df_gr_temp)), [f"{start:.2f}-{end:.2f}" for start, 
                                        end in zip(df_gr_temp["ground_temp_start"], df_gr_temp["ground_temp_end"])], rotation=90)
    plt.title("Change in gas volume in relation to Ground temperature change")
    plt.grid()

    # Plotting the data
    plt.plot(df_gr_temp.index, df_gr_temp["dm3_gas"], marker='s', linestyle='-', color='r')

    # Adding labels and legend
    plt.xlabel("Binned Ground Temperature")
    plt.ylabel("Gas Volume (dm3_gas)")
    plt.legend(["Gas Volume"], loc="upper left")

    # Show the plot
    plt.tight_layout()
    st.pyplot()

col1, col2 = st.columns(2)
with col1:
    # Plot settings
    plt.figure(figsize=(12, 5))
    plt.xticks(np.arange(len(df_g_temp)), [f"{start:.2f}-{end:.2f}" for start, 
                                        end in zip(df_g_temp["gas_temp_start"], df_g_temp["gas_temp_end"])], rotation=90)
    plt.title("Change in gas volume per in relation to Gas temperature change")
    plt.grid()

    # Plotting the data
    plt.plot(df_g_temp.index, df_g_temp["dm3_gas"], marker='s', linestyle='-', color='b')

    # Adding labels and legend
    plt.xlabel("Binned Gas Temperature")
    plt.ylabel("Gas Volume (dm3_gas)")
    plt.legend(["Gas Volume"], loc="upper left")

    # Show the plot
    plt.tight_layout()
    st.pyplot()

with col2:
    plt.figure(figsize=(12, 5))
    plt.xticks(np.arange(len(df_f_temp)), [f"{start:.2f}-{end:.2f}" for start, 
                                        end in zip(df_f_temp["fluid_temp_start"], df_f_temp["fluid_temp_end"])], rotation=90)
    plt.title("Change in gas volume in relation to Fluid temperature change")
    plt.grid()

    # Plotting the data
    plt.plot(df_f_temp.index, df_f_temp["dm3_gas"], marker='o', linestyle='-', color='g')

    # Adding labels and legend
    plt.xlabel("Binned Fluid Temperature")
    plt.ylabel("Gas Volume (dm3_gas)")
    plt.legend(["Gas Volume"], loc="upper left")

    # Show the plot
    plt.tight_layout()    
    st.pyplot()






    # Featre importances
col1, col2 = st.columns(2)    
with col1:
    st.subheader("Feature importance")   
    correlation = df[['fluid_temp', 'ground_temp', 'air_umidity', 'air_temp', 'gas_umidity', 'gas_temp', 'dm3_gas']].corr()
    cor = (correlation['dm3_gas'].reset_index()).head(6)
    plt.figure(figsize=(7,4))
    sns.barplot(x=cor['index'], y = cor['dm3_gas'], hue=cor['dm3_gas'] )
    plt.xticks(rotation=90)
    st.pyplot() 
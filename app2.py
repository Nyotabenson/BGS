# Libraries
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import f_oneway




st.set_page_config(page_title="2.0 Bio-gas Production",page_icon=":biogas:", layout="wide")
#-----------style----------

# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



st.markdown("""
<style>
.heading
         {
    font-size:55px !important;
    color: #f2180c;
    font-weight:100;
}
</style>

""",  unsafe_allow_html=True)

# disabling warning
st.set_option('deprecation.showPyplotGlobalUse', False)


#------------------------------------------------
#heading
col1, col2, col3 = st.columns(3)
with col1:
    st.image("bio1.png")
with col2:
    st.write("##")
    st.write("##")
    st.write("##")
    st.write("##")
    st.write(
           """<p class="heading"> Biogas Production.</p>

                """, unsafe_allow_html=True
               )
with col3:
     st.image("bio2.png")

st.write("---")



st.subheader("Dataset")
df = pd.read_csv("biogasdf_ready.csv", parse_dates=['date'])
st.write(df.head())

col1, col2 = st.columns(2)
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

    st.write("Visualing different categories in the datset to understand more about the dataset")
col1, col2= st.columns(2)

with col1:
    sns.countplot(x=df['phase_test'], hue=df['phase_test'])
    plt.title("Phase categories")      
    st.pyplot()
with col2:
    sns.countplot(x=df['bio_id'], hue = df['bio_id'])
    plt.title("Bio production categories")    
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

st.write("##")
st.subheader("Peak and Low points of the features in Gas Production")
st.write("Peak and Low points when there is NO production")
#st.write("---")
col1, col2 = st.columns(2)
with col1:
    st.write("Peak Points of Dependent Features")
    st.write(peak_points0[['fluid_temp', 'ground_temp', 'air_umidity', 'air_temp', 'gas_umidity', 'gas_temp', 'dm3_gas']])
    st.write("##")
with col2:
    #prod0
    st.write("low Points of Dependent Features ")
    st.write(low_points0[['fluid_temp', 'ground_temp', 'air_umidity', 'air_temp', 'gas_umidity', 'gas_temp', 'dm3_gas']])
    st.write("##")
    
st.write("Peak and Low points when there is  production")
#st.write("---")
col1, col2 = st.columns(2)
with col1:
     st.write("Peak Points of Dependent Features")
     st.write(peak_points1[['fluid_temp', 'ground_temp', 'air_umidity', 'air_temp', 'gas_umidity', 'gas_temp', 'dm3_gas']])
with col2:
    st.write("low Points of Dependent Features")
    st.write(low_points1[['fluid_temp', 'ground_temp', 'air_umidity', 'air_temp', 'gas_umidity', 'gas_temp', 'dm3_gas']])
st.write("Its noticable that there are extreme values in no production cases and moderate value in production cases.")
st.write("Meaning: In very high temperatures and very low temperatures there is NO production at all")




#----------------------ANOVA-----------------

# Select the columns you want to analyze (e.g., fluid_temp, ground_temp, air_umidity)
selected_columns = ['fluid_temp', 'ground_temp', 'air_umidity', "air_temp", "gas_umidity", "gas_temp", "bio_id", "phase_test"]
st.subheader("ANalysis Of VAriation (ANOVA)")
st.write("##")
col1, col2 = st.columns(2)
with col1:

    # Perform ANOVA for each selected column
    for col in selected_columns:
        groups = []  # List to store the data for each group
        categories = df['dm3_gas'].unique()  # Assuming we want to analyze based on 'dm3_gas' for now

        for category in categories:
            group_data = df[df['dm3_gas'] == category][col]
            groups.append(group_data)

        # Perform ANOVA
        f_statistic, p_value = f_oneway(*groups)
        
        #Print the results for each column
        st.write(f"ANOVA results for '{col}':")
        st.write("F-statistic:", f_statistic)
        st.write("p-value:", p_value)
        st.write("\n")

with col2:
    st.write("""The F-statistic measures the variation between groups relative to the variation within groups. A larger F-statistic indicates a larger difference in means among groups.
            """)
    st.write("""The p-value is the probability of obtaining the observed F-statistic (or a more extreme value) if the null hypothesis is true. A small p-value 
             (typically < 0.05) indicates that there is a significant difference between the groups.""")
    
#----------------------------MOdels-------------------------


model2 =pickle.load(open('rfc_model2.pkl', 'rb'))
model3 =pickle.load(open('rfc_model3.pkl', 'rb'))

modelled_group = pd.read_csv("modelled_group.csv")
modelled2_group = pd.read_csv("modelled2_group.csv")


col1,col2,col3 = st.columns(3)
with col2:
    st.header("Modelling")
st.subheader("Training different algorithms to help in predicting bio gas production")    
st.write("---")
col1,col2 = st.columns(2)
with col1:
    st.subheader("Model 1")
    st.write("A model trained with all the features")
    st.write("##")
    #Visualization
    fig, ax = plt.subplots(figsize=(12,7))
    ax.plot(modelled_group.Hour, modelled_group.preds, label="preds")
    ax.plot(modelled_group.Hour, modelled_group.dm3, label = "actual")
    plt.title("Actual vs predicted gas production")
    plt.xlabel("Hours")
    plt.ylabel("Gas production (dm3)")
    ax.set_xticks(np.arange(0, 24, step=1))
    st.pyplot()
    st.text("Figure 3a.1")
    st.write("##")

    st.write("Plotting a line of best fit with the predicted values against the actual dm3 values")
    # fitting best fit line
    slope1, intercept1 = np.polyfit(modelled_group.dm3, modelled_group.preds, 1)
    regression_line1 = slope1 * modelled_group.dm3 + intercept1


    plt.scatter(x=modelled_group.dm3, y = modelled_group.preds)
    plt.plot(modelled_group.dm3, regression_line1, color='red', label="actual Linear Regression Line")
    plt.title("Predicted vs Actual values")
    plt.xlabel("Actual dm3 values")
    plt.ylabel("Predicted dm3 values")
    plt.grid()
    plt.legend()
    st.pyplot()
    st.text("Figure 3a.2")

    # Input features
    st.write("Kindly enter your parameters")
    # Key-in the inputs
    with st.form('entry form1', clear_on_submit=True):
        bio_id = st.number_input("bio_id: ")
        phase_test = st.number_input("Phase_test: ")
        Month = st.number_input("Month: ")
        Hour = st.number_input("Hour: ")
        air_temp = st.number_input("Air Temperatures: ")
        gas_temp = st.number_input("Gas Temperatures: ")
        gas_umidity = st.number_input("gas humidity")
        air_umidity = st.number_input("air_humidity")
        ground_temp = st.number_input("ground_temp")
        fluid_temp = st.number_input("fluid_temp")
        submitted1 = st.form_submit_button("Submit")
        if submitted1:
            st.success("Data Saved")
    inputs = pd.DataFrame({ "fluid_temp" : [fluid_temp], "ground_temp" : [ground_temp], "air_umidity":[air_umidity],  "air_temp"  : [air_temp],'gas_umidity' : [gas_umidity],  'gas_temp' : [gas_temp],    "bio_id": [bio_id], "phase_test": [phase_test] , "Month" : [Month], "Hour" : [Hour], })      
    def prediction(inputs):
            inputs_log = np.log(inputs)
            pred1 = model2.predict(inputs_log)
            pred1 =round(pred1[0],5)
            return pred1        
    if st.checkbox("View Prediction"):
        st.write(f"The production is: {prediction(inputs)}dm3")
        

with col2:
    st.subheader("Model 2")
    st.write("A more automated MODEL train to use two parameters as input")
    st.write("##")
    #Visualization
    fig, ax = plt.subplots(figsize=(12,7))

    ax.plot(modelled2_group.Hour, modelled2_group.pred2, label="preds")
    ax.plot(modelled2_group.Hour, modelled2_group.dm3, label = "actual")
    ax.set_xticks(np.arange(0, 24, step=1))
    plt.title("Actual vs predicted gas production")
    plt.xlabel("Hours")
    plt.ylabel("Gas production (dm3)")
    st.pyplot()
    st.text("Figure 3b.1")
    st.write("##")

    st.write("Plotting a line of best fit with the predicted values against the actual dm3 values")
    # fitting best fit line
    slope1, intercept1 = np.polyfit(modelled2_group.dm3, modelled2_group.pred2, 1)
    regression_line1 = slope1 * modelled2_group.dm3 + intercept1


    plt.scatter(x=modelled2_group.dm3, y = modelled2_group.pred2)
    plt.plot(modelled2_group.dm3, regression_line1, color='red', label="actual Linear Regression Line")
    plt.title("Predicted vs Actual values")
    plt.xlabel("Actual dm3 values")
    plt.ylabel("Predicted dm3 values")
    plt.grid()
    plt.legend()
    st.pyplot()
    st.text("Figure 3b.2")
    # Input features
    st.write("Kindly enter your parameters")
    # Key-in the inputs
    with st.form('entry form3', clear_on_submit=True):
       
       fluid_temp = st.number_input("Fluid_temperature: ")
       air_umidity = st.number_input("Air_umidity: ")
       submitted1 = st.form_submit_button("Submit")
       if submitted1:
            st.success("Data Saved")
    input2 = pd.DataFrame({"fluid_temp" : [fluid_temp], "air_umidity" : [air_umidity]})       
    def prediction(input2):
            input2_log = np.log(input2)
            preds2 = model3.predict(input2_log)
            preds2 =round(preds2[0],5)
            return preds2        
    if st.checkbox("View Predictions"):
        st.write(f"The production is: {prediction(input2)}dm3")
        

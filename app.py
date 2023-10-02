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
    font-weight:100;
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
    #st.write("##")
    st.write("Key")
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

st.write("##")
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
#_________________VISUALS_______________________
df_a_temp = pd.read_csv("df_a_temp.csv")
df_gr_temp = pd.read_csv("df_gr_temp.csv")
df_f_temp = pd.read_csv("df_f_temp.csv")
df_g_temp = pd.read_csv("df_g_temp.csv")







st.subheader("Visuals of Peak and Low points of the different features")
col1, col2 = st.columns(2)
with col1:
    # Air Temperature
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
st.write("In reference from the visual: except for Air Temperature, the rest temperatures shows that there is no gas production in temperatutes under 7.77")





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

#_____________Can biogas 1 produce as much as biogas 2 in any month or phase?_____________
bio1_month = pd.read_csv("bio1_month.csv")
bio2_month = pd.read_csv("bio2_month.csv")
st.subheader("Can biogas 1 produce as much as biogas 2 in any month or phase?")    

with col2:
    plt.figure(figsize=(7,7))
    sns.heatmap(df.drop(columns=['date','hour'], axis=1).corr(), cmap="YlGnBu", annot=True)
    st.pyplot()

col1, col2 = st.columns(2)
with col1:
    # Create a Figure and Axes
    fig, ax = plt.subplots(figsize=(12,5))


    ax.plot(bio1_month['Month'], bio1_month['dm3_gas'], label='bio1')
    ax.plot(bio2_month['Month'], bio2_month['dm3_gas'], label='bio2')

    # Add title and labels
    ax.set_title('Bio1 Bio2 Production across the Months')
    ax.set_xlabel('Months')
    ax.set_ylabel('Volume')
    ax.set_xticks(np.arange(1, 13, step=1))
    plt.grid()
    ax.legend()
    st.pyplot()

with col2:
    st.write(" ")
    st.write("Bio 1 canot produce as much as Bio2 across any month.")    
    st.write("Noticably in the second month Bio1 production almost equals that of Bio2.")


#________________Is it possible to know the months with high and low production?___________
st.subheader("Is it possible to know the months with high and low production?")
col1, col2 = st.columns(2)
with col1:
    # Create a Figure and Axes
    fig, ax = plt.subplots(figsize=(12,5))


    ax.plot(bio1_month['Month'], bio1_month['dm3_gas'], label='bio1')
    ax.plot(bio2_month['Month'], bio2_month['dm3_gas'], label='bio2')

    # Add title and labels
    ax.set_title('Bio1 Bio2 Production across the Months')
    ax.set_xlabel('Months')
    ax.set_ylabel('Volume')
    ax.set_xticks(np.arange(1, 13, step=1))
    plt.grid()
    ax.legend()
    st.pyplot()
with col2:
    df_month = pd.read_csv("df_month.csv") 
    plt.figure(figsize=(12,5))
    plt.plot(df_month.Month, df_month.dm3_gas)
    plt.xticks(np.arange(0, 13, step=1))
    plt.title("Change in gas volume per hour")
    plt.grid()
    st.pyplot()  
st.write("Bio 1; Months with highest production are 4th month(April), 3rd Month(March) followed by 7th month (July). On the other hand the last two months of the year has the lowest production.")    
st.write("Bio 2; Its noted that only one month had high poduction which is the 3rd month(March) and several months with real low production which are 6th,7th,8th,9th and 10th months.")

#__________________
df = pd.read_csv("biogasdf_ready.csv", parse_dates=['date'])
X = df[["fluid_temp","ground_temp","air_umidity","gas_umidity","gas_temp"]]
y = df['dm3_gas']
X = X.fillna(0)
df['dm3_gas'].replace({0.00: 0, 0.01: 1}, inplace=True)
X_train, X_test,y_train,  y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test))

#__________________________

st.subheader("Models")
st.write("Using Bio1 Phase2 because of the target value distribution")
df12 = pd.read_csv("df12.csv")
df12['dm3_gas'].replace({0.00: 0, 0.01: 1}, inplace=True)
col1,col2 = st.columns(2)
with col1:
    plt.figure(figsize=(8,5))
    sns.scatterplot(x='fluid_temp', y='ground_temp', hue='dm3_gas', data=df12)
    plt.grid()
    plt.title("ground_temp vs fluid temp in relation to gas production")
    st.pyplot()

with col2:
    
    slope1, intercept1 = np.polyfit(df12.fluid_temp, df12.ground_temp, 1)

    #1.2 for the predicted values
    #slope2, intercept2 = np.polyfit(X_train, y_train, 1)
    # Create the regression line equations for both actual and predicted values
    plt.figure(figsize=(8,5))
    regression_line1 = slope1 * df12.fluid_temp + intercept1
    #regression_line2 = slope2 * x + intercept2

    # Plot the data points
    plt.scatter(df12.fluid_temp, df12.ground_temp, c = df12.dm3_gas, label="Actual gas Production")
    #plt.scatter(x, y_pred, label="Predicted Values")

    # Plot the regression line
    plt.plot(df12.fluid_temp, regression_line1, color='red', label="actual Linear Regression Line")
    #plt.plot(x, regression_line2, color='green', label="predicted Linear Regression Line")


    # Title, labels and a legend
    plt.title("ground_temp vs fluid temp in relation to gas production & line of best fit")
    plt.xlabel("ground_temp")
    plt.ylabel("fluid_temp")
    #plt.xticks(np.arange(15,44,3))
    plt.legend()
    plt.grid()

    # Show the plot
    st.pyplot()   





model =pickle.load(open('rfc_model1.pkl', 'rb'))
st.write("Kindly puts your parameters")
# Key-in the inputs
gas_temp = st.number_input("Gas Temperatures: ")
gas_umidity = st.number_input("gas humidity")
air_umidity = st.number_input("air_humidity")
ground_temp = st.number_input("ground_temp")
fluid_temp = st.number_input("fluid_temp")

inputs = pd.DataFrame({'gas_temp' : [gas_temp], 'gas_umidity' : [gas_umidity],  "air_umidity"  : [air_umidity], "ground_temp" : [ground_temp], "fluid_temp" : [fluid_temp] })
def prediction(inputs):
    scaler = StandardScaler()
    X_pred_scaled = pd.DataFrame(scaler.fit_transform(inputs))
    preds1 = model.predict(X_pred_scaled)
    pred1 =round(preds1[0])
    return pred1
st.write(prediction(inputs))
st.write(f"{(round(model.score(X_test_scaled, y_test),2))*100}%")

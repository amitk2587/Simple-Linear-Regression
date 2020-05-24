
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

'''
1) Calories_consumed-> predict weight gained using calories consumed
'''
cal_con=pd.read_csv("D:/Data_Science/Data_Sci_Assignment/Simple Linear Regression/calories_consumed.csv")
cal_con.columns
np.corrcoef(cal_con['Weight gained (grams)'],cal_con['Calories Consumed'])

plt.hist(cal_con['Calories Consumed'])

#boxplot to check for outliers
plt.boxplot(cal_con['Calories Consumed'])
plt.boxplot(cal_con['Weight gained (grams)'])

#scaterplot to check the linera relation
plt.scatter(cal_con['Weight gained (grams)'],cal_con['Calories Consumed']);plt.xlabel('Weight gained (grams)');plt.ylabel('Calories Consumed')

#removing the outliers
cal_con1 = cal_con.drop(cal_con.index[cal_con['Weight gained (grams)'] == 62],axis=0)
cal_con2 = cal_con1.drop(cal_con.index[cal_con['Calories Consumed'] == 1400],axis=0)
cal_con = cal_con2

# cal_con['Weight gained (grams)'] ~ cal_con['Calories Consumed'] model1
import statsmodels.formula.api as smf
model_1=smf.ols("cal_con['Weight gained (grams)'] ~ cal_con['Calories Consumed']",data=cal_con).fit()

# For getting coefficients of the varibles used in equation
model.params
model.summary()



pred = model.predict(cal_con['Calories Consumed'])

error = cal_con['Weight gained (grams)'] - pred
np.mean(error) # 0
np.sqrt(sum(error**2)/12) # 94.61

import matplotlib.pyplot as plt
plt.scatter(pred,cal_con['Weight gained (grams)'],c="r")
#the plot looks slightly curvi linera so trying curvilinear equation


model_2=smf.ols("cal_con['Weight gained (grams)'] ~ np.log(cal_con['Calories Consumed'] )",data=cal_con).fit()

model_2.summary()

model_2.resid
pred = model_2.predict(cal_con['Calories Consumed'])
np.mean(model_2.resid) #0
np.sqrt(sum(model_2.resid**2)/12) #RMSE 94.61

import matplotlib.pyplot as plt
plt.scatter(pred,cal_con['Weight gained (grams)'],c="r");

'''
2) Delivery_time -> Predict delivery time using sorting time 
'''
Delivery_time=pd.read_csv("D:/Data_Science/Data_Sci_Assignment/Simple Linear Regression/delivery_time.csv")
Delivery_time.columns

#checking the correlation
Delivery_time.corr()
plt.scatter(np.log(Delivery_time['Delivery Time']),Delivery_time['Sorting Time'])
Delivery_time.describe()
#boxplot to check the outliers
plt.boxplot(Delivery_time['Delivery Time'])
plt.boxplot(Delivery_time['Sorting Time'])

#Delivery_time.drop(index=[18,7],inplace=True)

import statsmodels.formula.api as smf
#model=smf.ols("np.log(Delivery_time['Delivery Time']) ~ Delivery_time['Sorting Time'] + Delivery_time['Sorting Time']*Delivery_time['Sorting Time']",data=Delivery_time).fit()
model = smf.ols("Delivery_time['Delivery Time'] ~ Delivery_time['Sorting Time']",data=Delivery_time).fit()
model.summary()



Delivery_time_predict = model.predict(Delivery_time['Sorting Time'])
error = Delivery_time_predict -Delivery_time['Delivery Time']

np.mean(model.resid) #0
np.mean(error) #0
np.sqrt(sum(error**2)/12) #3.69

import matplotlib.pyplot as plt
plt.scatter(Delivery_time_predict,Delivery_time['Sorting Time'],c="r")

'''
3) Emp_data -> Build a prediction model for Churn_out_rate 
'''
Emp_data = pd.read_csv("D:/Data_Science/Data_Sci_Assignment/Simple Linear Regression/emp_data.csv")
Emp_data.columns
Emp_data.corr()

plt.scatter(Emp_data['Salary_hike'],Emp_data['Churn_out_rate']) #plot loooks curvilinear

model_emp = smf.ols("Emp_data['Churn_out_rate'] ~ Emp_data['Salary_hike'] + I(Emp_data['Salary_hike']*Emp_data['Salary_hike'])",data=Emp_data).fit()
model_emp.summary()



pred = model_emp.predict(Emp_data['Salary_hike'])
error = pred - Emp_data['Churn_out_rate']

np.mean(model_emp.resid)#0
np.sqrt(sum(error**2)/12) #1.44

import matplotlib.pyplot as plt
plt.scatter(pred,Emp_data['Salary_hike'],c="r")

'''
4) Salary_hike -> Build a prediction model for Salary_hike
'''
Salary_hike = pd.read_csv("D:/Data_Science/Data_Sci_Assignment/Simple Linear Regression/Salary_Data.csv")
Salary_hike.columns
Salary_hike.corr()

plt.scatter(Salary_hike['YearsExperience'], Salary_hike['Salary'])

model = smf.ols("Salary_hike['Salary'] ~ Salary_hike['YearsExperience']",data=Salary_hike).fit()
model.summary()



pred = model.predict(Salary_hike['YearsExperience'])
error = pred - Salary_hike['Salary']
np.mean(model.resid) #0

np.sqrt(sum(error**2)/12) #8841.79

import matplotlib.pyplot as plt
plt.scatter(pred,Salary_hike['YearsExperience'],c="r")

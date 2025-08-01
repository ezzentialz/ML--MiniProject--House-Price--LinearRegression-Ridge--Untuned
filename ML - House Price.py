import pandas as pd
import numpy as np

###################################### ขั้นตอนที่1 Load และ เช็ค ##############################################################
df_ames = pd.read_csv("D:/S/Python/MyRoadmap/Practice Python/Mini Project Practice/ML- Linear Regression- HousePrice/train.csv")
test_df = pd.read_csv("D:/S/Python/MyRoadmap/Practice Python/Mini Project Practice/ML- Linear Regression- HousePrice/test.csv")

print(f"train set : \n {df_ames.head()}") #debug : ดูว่า read_csv ของ train set เป็นยังไงบ้าง
print(f" test set : {test_df.head()}") #debug :ดูว่า read_csv ของ test set เป็นยังไงบ้าง

print(f" ------------------------------- เช็คข้อมูล train set ----------------------------------------------------- ")
print(df_ames.info())# debug: มีdtype 3ประเภท  float64(3), int64(35), object(43)
print(df_ames.describe())#debug: จำนวนแถว 1460 คอลัม 81 (รวมIDด้วย) 
print(f"ค่าว่าง: \n {df_ames.isnull().sum().sort_values(ascending=False)}")   

'''print(f" \n ------------------------------- เช็คข้อมูล test set ----------------------------------------------------- ")
print(test_df.info())
print(test_df.describe())
print(test_df.isnull().sum())
print(f" \n ------------------------------------------------------------------------------------ ")'''
'''print(f" ------------------------------- เช็คข้อมูล ['MSZoning'] ----------------------------------------------------- ")
print(df_ames['MSZoning'].value_counts(dropna=False)) # 5 types Zoning : RL, RM, FV, RH, C (all)
print(f" ------------------------------- เช็คข้อมูล ['Utilities'] ----------------------------------------------------- ")
print(df_ames['Utilities'].value_counts(dropna=False)) # 2 types utilities : Allpub, NoSeWa(Electricity and Gas only)
print(f" ------------------------------- เช็คข้อมูล ['SaleCondition'] ----------------------------------------------------- ")
print(df_ames['SaleCondition'].value_counts(dropna=False))# 6 conditions
print(f" ------------------------------- เช็คข้อมูล ['PoolQC'] ----------------------------------------------------- ")
print(df_ames['PoolQC'].value_counts(dropna=False))# nan 1453
print(f" ------------------------------- เช็คข้อมูล ['MiscFeature'] ----------------------------------------------------- ")
print(df_ames['MiscFeature'].value_counts(dropna=False)) #nan 1406
print(f" ------------------------------- เช็คข้อมูล ['Alley'] ----------------------------------------------------- ")
print(df_ames['Alley'].value_counts(dropna=False)) #nan 1369
print(f" ------------------------------- เช็คข้อมูล ['Fence'] ----------------------------------------------------- ")
print(df_ames['Fence'].value_counts(dropna=False)) #nan 1179
print(f" ------------------------------- เช็คข้อมูล ['MasVnrType'] ----------------------------------------------------- ")
print(df_ames['MasVnrType'].value_counts(dropna=False)) # Nan 872'''

################################################ ขั้นตอนที่ 2 Data Cleaning - replace - fill  / part 1 ##############################################
categorical_features = ['Alley','PoolQC','Fence','MiscFeature','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
        'BsmtFinType2','GarageType','GarageFinish','GarageQual','GarageCond','FireplaceQu']

numerical_features = ['MasVnrArea','GarageYrBlt','GarageCars','GarageArea','BsmtFinSF1',
        'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','Fireplaces']

#### แทน NA เป็น none หรือ 0 ###
df_ames[categorical_features] = df_ames[categorical_features].replace('NA', np.nan)
test_df[categorical_features] = test_df[categorical_features].replace('NA', np.nan)

df_ames[numerical_features] = df_ames[numerical_features].replace('NA', np.nan)
test_df[numerical_features] = test_df[numerical_features].replace('NA', np.nan)

### เติมค่า ว่าง ###
df_ames[categorical_features] = df_ames[categorical_features].fillna('None')
test_df[categorical_features] = test_df[categorical_features].fillna('None')

df_ames[numerical_features] = df_ames[numerical_features].fillna(0)
test_df[numerical_features] = test_df[numerical_features].fillna(0)

print(f" ------------------------------- เช็คข้อมูล train set ----------------------------------------------------- ")
print(f"ค่าว่าง: \n {df_ames.isnull().sum().sort_values(ascending=False)}")
print(f" ------------------------------- เช็คข้อมูล test set ----------------------------------------------------- ")
print(f"ค่าว่าง: \n {test_df.isnull().sum().sort_values(ascending=False)}")

print(f"train set : \n {df_ames.head()}") #debug : ดูว่า missing value ตรงไหนบ้าง
print(f" test set : \n {test_df.head()}") #debug :ดูว่า missing value ตรงไหนบ้าง


########################################## Data Cleaning / part 2  ############################################################
LotFrontage_median = df_ames['LotFrontage'].median()
df_ames['LotFrontage'] = df_ames['LotFrontage'].fillna(LotFrontage_median)
Electricital_mode = df_ames['Electrical'].mode()[0]
df_ames['Electrical'] = df_ames['Electrical'].fillna(Electricital_mode)

categorical_features_test_set = ['MSZoning','Utilities','Functional','Exterior2nd','Exterior1st','SaleType','KitchenQual']
numerical_features_test_set = ['LotFrontage']
### ใช้ mean จาก df_ames ####### ห้ามใช้ กับ test set เด็ดขาด เพื่อป้องกัน LEAK DATA !!! #############
numerical_features_median = df_ames[numerical_features_test_set].median()
test_df[numerical_features_test_set] = test_df[numerical_features_test_set].fillna(numerical_features_median)
### ใช้ mode จาก df_ames ###
for col in categorical_features_test_set:
    if col in test_df: #debug: if แรก เช็คว่า มีคอลลัมนี้อยู่ใน test_df ไหม ถ้ามี = ใช้ mode กับ ames_df / ถ้าไม่มี = ใช้ mode ของ test_df
        if col in df_ames.columns:
            cat_features_test_set_mode = df_ames[col].mode()[0] #กำหนดให้ col (รัน features ในcategorical_features_test_set ทีละตัว )เป็นmode [0] ตัวแรก จาก ames_df
        else:
            cat_features_test_set_mode = test_df[col].mode()[0] #กำหนดให้ col (รัน features ในcategorical_features_test_set ทีละตัว )เป็นmode [0] ตัวแรก จาก test_df
    test_df[col].fillna(cat_features_test_set_mode, inplace=True) #เมื่อ เลือกได้แล้วว่าจะใช้ mode ตัวไหน ก็ให้ เก็บเป็น cat_features_test_set_mode แล้วมาเติมใน fillna 

print(f" ------------------------------- เช็คข้อมูล train set ----------------------------------------------------- ")
print(f"ค่าว่าง: \n {df_ames.isnull().sum().sort_values(ascending=False)}")
print(f" ------------------------------- เช็คข้อมูล test set ----------------------------------------------------- ")
print(f"ค่าว่าง: \n {test_df.isnull().sum().sort_values(ascending=False)}")


########################################## ขั้นตอนที่ 3 การแปลงข้อมูล (Feature Engineering & Encoding) ############################################################

# แปลง MSSubClass จาก ตัวเลขของคลาส(dtype:int) ให้เป็น(dtype:object)
df_ames['MSSubClass'] = df_ames['MSSubClass'].astype(str)
test_df['MSSubClass'] = test_df['MSSubClass'].astype(str)

#ลด skew ใน SalePrice เพราะตัวเลข ในคอลลัมนี้ เยอะไป จะทำให้โมเดลทำนายไม่ดี (ตรงนี้ใช้สูตร log1p)
df_ames['SalePrice_log'] = np.log1p(df_ames['SalePrice'])
print(f"Debug: เช็คว่า SalePrice เป็นยังไง {df_ames['SalePrice']}")
print(f"Debug: เช็คว่า SalePrice_log เป็นยังไง {df_ames['SalePrice_log']}")

#ลบ ID กับ SalePrice และ SalePrice_log ออกก่อน เพื่อ ทำ OneHotEncoding
ID_ames = df_ames['Id']
Sales_Price_log_ames = df_ames['SalePrice_log']
ID_test = test_df['Id']

preprocessed_ames = df_ames.drop(columns=['Id','SalePrice','SalePrice_log'])
preprocess_test= test_df.drop(columns=['Id'])

print(f" \n ------------------------------------------- ลบ ID กับ SalePrice ออก ก่อนทำ Onehot--------------------------------------------------")
print(preprocessed_ames.head()) #debug: เช็คคอลัมว่า ลบแล้วหรือยัง
print(preprocess_test.head())

############################## รวมdata เข้าด้วยกันเพื่อทำ OneHotEncoding เพื่อป้องกัน ไม่ให้เวลาทำ Onehot แล้ว data แต่ละอัน ไม่เท่ากัน #########################
full_data = pd.concat([preprocessed_ames, preprocess_test], ignore_index=True)

#เลือกคอลลัม obect ใน categorical_features ก่อน ทำ OnehotEncoding **ขั้ตอนนี้จะสามารถเลือก dtype ทั้งหมดที่เป็น object ได้เลย

'''ames_objects_for_onehot = preprocessed_ames.select_dtypes(include='object').columns
test_objects_for_onehot = preprocess_test.select_dtypes(include='object').columns'''

full_data_object_column = full_data.select_dtypes(include='object').columns

#Onehotencoding สำหรับ categorical_features เพื่อแยกประเภท ออกเป็นตัวเลข

'''ames_final_df = pd.get_dummies(preprocessed_ames,columns=ames_objects_for_onehot)
test_final_df = pd.get_dummies(preprocess_test,columns=test_objects_for_onehot)'''

full_data_preprocessed = pd.get_dummies(full_data, columns=full_data_object_column)

###############################  แยก data หลังจากที่ทำ Onehot เสร็จแล้ว ############################################

ames_final_df = full_data_preprocessed.iloc[:len(preprocessed_ames)].copy() #ใช้ ก็อปปี้เพื่อป้องกัน SettingWithCopyWarning
test_final_df = full_data_preprocessed.iloc[len(preprocessed_ames):]


print(f" \n ------------------------------------------- ทำ OnehotEncoding เส็จแล้ว--------------------------------------------------")
#ใส่ ID กับ SalePrice และ SalePrice_log กลับมา หลังจากทำ OneHotEncoding
ames_final_df['Id'] = ID_ames
ames_final_df['SalePrice_log'] = Sales_Price_log_ames

test_final_df['Id'] = ID_test
print(f" \n ------------------------------------- ใส่ ID กับ SalePrice กลับเข้ามา หลังทำ Onehot----------------------------------------------")
print(ames_final_df.shape) #debug: เช็คขนาด
print(test_final_df.shape) #debug: เช็คขนาด

print(ames_final_df.isnull().sum().sort_values(ascending=False).head()) # เช็คค่าว่าง ames 
print(test_final_df.isnull().sum())

########################################## ขั้นตอนที่ 4  การสร้างและฝึกโมเดล (Model Training) ############################################################
X = ames_final_df.drop(columns=['Id','SalePrice_log'])
y = ames_final_df['SalePrice_log']
test_val = test_final_df.drop(columns=['Id'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f" จำนวน X_train {X_train.shape}")
print(f" จำนวน X_test {X_test.shape}")
print(f"จำนวน  y_train {y_train.shape} ")
print(f"จำนวน  y_test {y_test.shape} ")

#### ลองใช้ model linear Regression 
from sklearn.linear_model import LinearRegression
lr_model =  LinearRegression()
print(f"\n -----------------เริ่มการ train Linear Regression Model (Train set)-------------------")
lr_model.fit(X_train,y_train)
print(f"\n -----------------train Linear Regression Model เรียบร้อย ( Train Set)------------------")
y_lr_model_predict = lr_model.predict(X_test)

#แปลงค่า np.log1p  กลับ เป็นตัวเลขปกติ คิดว่าจน่าจะใช้สูตร expอะไรสักอย่าง
y_lr_model_predict_normal_price = np.expm1(y_lr_model_predict)
print(f"ราคาปกติของค่าบ้านที่โมเดลทำนาย : {y_lr_model_predict_normal_price.round(2)}")
#### วัดผล Model ใช้ metrix score MAE MSE R2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print(f"\n ------------- วัดผลประเมินของ train set --------------------------- \n")
mae_lr_model = mean_absolute_error(y_test, y_lr_model_predict)
mse_lr_model = mean_squared_error(y_test, y_lr_model_predict)
rmse_lr_model = np.sqrt(mse_lr_model)
r2_lr_model = r2_score(y_test, y_lr_model_predict)

print(f"MAE (untuned )Linear Regression: {mae_lr_model:.2f}")
print(f"MSE (untuned )Linear Regression: {mse_lr_model:.2f}")
print(f"RMSE (untuned )Linear Regression: {rmse_lr_model:.2f}")
print(f"R2 (untuned )Linear Regression: {r2_lr_model:.2f}")

################# ใช้กับข้อมมูล test set บ้าง ###########
print(f"\n -----------------เริ่มการ train Linear Regression Model (Test set - submission(house price))-------------------")
lr_model = LinearRegression()
lr_model.fit(X_train,y_train) #ทำแบบเดิม
test_val_predict = lr_model.predict(test_val) #predictด้วย test set
print(f"\n -----------------train Linear Regression Model เรียบร้อย ( Test Set - submission(house price))------------------")
y_test_val_predict_normal_price = np.expm1(test_val_predict).round(2)
print(f"ราคาปกติ(Test Set) ของค่าบ้านที่โมเดลทำนาย : {y_test_val_predict_normal_price}")

submission_df = pd.DataFrame({'Id': ID_test ,'SalePrice': y_test_val_predict_normal_price})
print(submission_df.head())


###############  to CSV ##########
print(f"สร้างsubmission เป็น ไฟล์CSV")
submission_df.to_csv('submission.csv', index=False)
print(f"สร้าง ไฟล์ subbmission.csv เรียบร้อย")







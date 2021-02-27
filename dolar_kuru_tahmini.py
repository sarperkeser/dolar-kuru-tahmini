#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#2. Veri Onisleme

#2.1. Veri Yukleme
data_tum = pd.read_excel('tum_veriler.xlsx')
# verileri ayırma
bist = data_tum.iloc[:,1:2]
cari_acik = data_tum.iloc[:,2:3]
dolar = data_tum.iloc[:,3:4]
doviz_rezervi = data_tum.iloc[:,5:6]
para_arzi = data_tum.iloc[:,6:7]
faiz = data_tum.iloc[:,11:12]
enflasyon = data_tum.iloc[:,10:11]
ihracat = data_tum.iloc[:,7:8]
ithalat = data_tum.iloc[:,8:9]
dis_ticaret = data_tum.iloc[:,9:10]

# multilinear için verileri birleştirme
son=data_tum.iloc[:,[1,2,5,6,7,8,9,10,11]]
son_value = son.values



# polinom için value yapma
bistvalue=bist.values
carivalue=cari_acik.values
dolarvalue=dolar.values
dövizvalue=doviz_rezervi.values
paravalue=para_arzi.values
faizvalue=faiz.values
enflasyonvalue=enflasyon.values
ihracatvalue=ihracat.values
ithalatvalue=ithalat.values
disvalue=dis_ticaret.values


# test ve train için 
y_train,y_test =train_test_split(dolar,test_size=0.33, random_state=0)
bist_train, bist_test= train_test_split(bist,test_size=0.33, random_state=0)
cari_train, cari_test = train_test_split(cari_acik,test_size=0.33, random_state=0)
döviz_train, döviz_test = train_test_split(doviz_rezervi,test_size=0.33, random_state=0)
para_train , para_test =train_test_split(para_arzi,test_size=0.33, random_state=0)
faiz_train, faiz_test=train_test_split(faiz,test_size=0.33, random_state=0)
enflasyon_train, enflasyon_test =train_test_split(enflasyon,test_size=0.33, random_state=0)
ihracat_train , ihracat_test=train_test_split(ihracat,test_size=0.33, random_state=0)
ithalat_train , ithalat_test =train_test_split(ithalat,test_size=0.33, random_state=0)
dis_train , dis_test =train_test_split(dis_ticaret,test_size=0.33, random_state=0)
son_train , son_test =train_test_split(son,test_size=0.33, random_state=0)


# verilerin sıralanması için
y_train = y_train.sort_index()
y_test = y_test.sort_index()
bist_train = bist_train.sort_index()
bist_test = bist_test.sort_index()
cari_train = cari_train.sort_index()
cari_test = cari_test.sort_index()
döviz_test =döviz_test.sort_index()
döviz_train = döviz_train.sort_index()
para_train=para_train.sort_index()
para_test=para_test.sort_index()
faiz_train=faiz_train.sort_index()
faiz_test=faiz_test.sort_index()
enflasyon_train=enflasyon_train.sort_index()
enflasyon_test=enflasyon_test.sort_index()
ihracat_train=ihracat_train.sort_index()
ihracat_test=ihracat_test.sort_index()
ithalat_train=ithalat_train.sort_index()
ithalat_test=ithalat_test.sort_index()
dis_train=dis_train.sort_index()
dis_test=dis_test.sort_index()
son_train =son_train.sort_index()
son_test = son_test.sort_index()



#verilerin olceklenmesi

sc1=StandardScaler()
son_olcekli = sc1.fit_transform(son_value)
sc2=StandardScaler()
dolar_olcekli = np.ravel(sc2.fit_transform(dolarvalue.reshape(-1,1)))

# linear regression
# bist
lr1=LinearRegression()
lr1.fit(bist_train,y_train)
tahmin1=lr1.predict(bist_test)

print("********************************************************************************")
print("Lineer Bist:")
model_lr1=sm.OLS(lr1.predict(bist_train),bist_train)
print(model_lr1.fit().summary())

# cari
lr2=LinearRegression()
lr2.fit(cari_train,y_train)
tahmin2=lr2.predict(cari_test)

print("********************************************************************************")
print("Lineer Cari Açık:")
model_lr2=sm.OLS(lr2.predict(cari_train),cari_train)
print(model_lr2.fit().summary())

# döviz
lr3=LinearRegression()
lr3.fit(döviz_train,y_train)
tahmin3=lr3.predict(döviz_test)

print("********************************************************************************")
print("Lineer Döviz Rezervi:")
model_lr3=sm.OLS(lr3.predict(döviz_train),döviz_train)
print(model_lr3.fit().summary())

# PARA
lr4=LinearRegression()
lr4.fit(para_train,y_train)
tahmin4=lr4.predict(para_test)

print("********************************************************************************")
print("Lineer Para Arzı:")
model_lr4=sm.OLS(lr4.predict(para_train),para_train)
print(model_lr4.fit().summary())

# FAİZ
lr5=LinearRegression()
lr5.fit(faiz_train,y_train)
tahmin5=lr5.predict(faiz_test)

print("********************************************************************************")
print("Lineer Faiz:")
model_lr5=sm.OLS(lr5.predict(faiz_train),faiz_train)
print(model_lr5.fit().summary())

# ENFLASYON
lr6=LinearRegression()
lr6.fit(enflasyon_train,y_train)
tahmin6=lr6.predict(enflasyon_test)

print("********************************************************************************")
print("Lineer Enflasyon:")
model_lr6=sm.OLS(lr6.predict(enflasyon_train),enflasyon_train)
print(model_lr6.fit().summary())

# İHRACAT
lr7=LinearRegression()
lr7.fit(ihracat_train,y_train)
tahmin7=lr7.predict(ihracat_test)

print("********************************************************************************")
print("Lineer İhracat:")
model_lr7=sm.OLS(lr7.predict(ihracat_train),ihracat_train)
print(model_lr7.fit().summary())

# İTHALAT
lr8=LinearRegression()
lr8.fit(ithalat_train,y_train)
tahmin8=lr8.predict(ithalat_test)

print("********************************************************************************")
print("Lineer İthalat:")
model_lr8=sm.OLS(lr8.predict(ithalat_train),ithalat_train)
print(model_lr8.fit().summary())

# DİS TİCARET
lr9=LinearRegression()
lr9.fit(dis_train,y_train)
tahmin9=lr9.predict(dis_test)

print("********************************************************************************")
print("Lineer Dış Ticaret:")
model_lr9=sm.OLS(lr9.predict(dis_train),dis_train)
print(model_lr9.fit().summary())

#multilinear
mlr1=LinearRegression()
mlr1.fit(son_train,y_train)
multilineartahmin=mlr1.predict(son_test)

print("********************************************************************************")
print("Multilineer:")
model_mlr1=sm.OLS(mlr1.predict(son_train),son_train)
print(model_mlr1.fit().summary())



# polinomal
# BİST
poly_reg1= PolynomialFeatures(degree=3)
x_poly1=poly_reg1.fit_transform(bistvalue)
mlr2=LinearRegression()
mlr2.fit(x_poly1,dolarvalue)
polinomtahmin1=mlr2.predict(poly_reg1.fit_transform(bistvalue))


print("********************************************************************************")
print("Polinom Bist:")
model_mlr2=sm.OLS(mlr2.predict(poly_reg1.fit_transform(bistvalue)),bistvalue)
print(model_mlr2.fit().summary())

# CARİ
poly_reg2= PolynomialFeatures(degree=3)
x_poly2=poly_reg2.fit_transform(carivalue)
mlr3=LinearRegression()
mlr3.fit(x_poly2,dolarvalue)
polinomtahmin2=mlr3.predict(poly_reg2.fit_transform(carivalue))

print("********************************************************************************")
print("Polinom CARİ AÇIK:")
model_mlr3=sm.OLS(mlr3.predict(poly_reg2.fit_transform(carivalue)),carivalue)
print(model_mlr3.fit().summary())


# DÖVİZ
poly_reg3= PolynomialFeatures(degree=3)
x_poly3=poly_reg3.fit_transform(dövizvalue)
mlr4=LinearRegression()
mlr4.fit(x_poly3,dolarvalue)
polinomtahmin3=mlr4.predict(poly_reg3.fit_transform(dövizvalue))

print("********************************************************************************")
print("Polinom Döviz Rezervi:")
model_mlr4=sm.OLS(mlr4.predict(poly_reg3.fit_transform(dövizvalue)),dövizvalue)
print(model_mlr4.fit().summary())


# PARAARZI
poly_reg4= PolynomialFeatures(degree=3)
x_poly4=poly_reg4.fit_transform(paravalue)
mlr5=LinearRegression()
mlr5.fit(x_poly4,dolarvalue)
polinomtahmin4=mlr5.predict(poly_reg4.fit_transform(paravalue))

print("********************************************************************************")
print("Polinom Para Arzı:")
model_mlr5=sm.OLS(mlr5.predict(poly_reg4.fit_transform(paravalue)),paravalue)
print(model_mlr5.fit().summary())


# FAİZ
poly_reg5= PolynomialFeatures(degree=3)
x_poly5=poly_reg5.fit_transform(faizvalue)
mlr6=LinearRegression()
mlr6.fit(x_poly5,dolarvalue)
polinomtahmin5=mlr6.predict(poly_reg5.fit_transform(faizvalue))

print("********************************************************************************")
print("Polinom Faiz:")
model_mlr6=sm.OLS(mlr6.predict(poly_reg5.fit_transform(faizvalue)),faizvalue)
print(model_mlr6.fit().summary())


# ENFLASYON
poly_reg6= PolynomialFeatures(degree=3)
x_poly6=poly_reg6.fit_transform(enflasyonvalue)
mlr7=LinearRegression()
mlr7.fit(x_poly6,dolarvalue)
polinomtahmin6=mlr7.predict(poly_reg6.fit_transform(enflasyonvalue))

print("********************************************************************************")
print("Polinom Enflasyon:")
model_mlr7=sm.OLS(mlr7.predict(poly_reg6.fit_transform(enflasyonvalue)),enflasyonvalue)
print(model_mlr7.fit().summary())


# İHRACAT
poly_reg7= PolynomialFeatures(degree=3)
x_poly7=poly_reg7.fit_transform(ihracatvalue)
mlr8=LinearRegression()
mlr8.fit(x_poly7,dolarvalue)
polinomtahmin7=mlr8.predict(poly_reg7.fit_transform(ihracatvalue))

print("********************************************************************************")
print("Polinom İhracat:")
model_mlr8=sm.OLS(mlr8.predict(poly_reg7.fit_transform(ihracatvalue)),ihracatvalue)
print(model_mlr8.fit().summary())


# İTHALAT
poly_reg8= PolynomialFeatures(degree=3)
x_poly8=poly_reg8.fit_transform(ithalatvalue)
mlr9=LinearRegression()
mlr9.fit(x_poly8,dolarvalue)
polinomtahmin8=mlr9.predict(poly_reg8.fit_transform(ithalatvalue))

print("********************************************************************************")
print("Polinom İthalat:")
model_mlr9=sm.OLS(mlr9.predict(poly_reg8.fit_transform(ithalatvalue)),ithalatvalue)
print(model_mlr9.fit().summary())


# DİS TİCARET
poly_reg9= PolynomialFeatures(degree=3)
x_poly9=poly_reg9.fit_transform(disvalue)
mlr10=LinearRegression()
mlr10.fit(x_poly9,dolarvalue)
polinomtahmin9=mlr10.predict(poly_reg9.fit_transform(disvalue))

print("********************************************************************************")
print("Polinom Dış Ticaret:")
model_mlr10=sm.OLS(mlr10.predict(poly_reg9.fit_transform(disvalue)),disvalue)
print(model_mlr10.fit().summary())

# multipolnom
toplam=np.append(x_poly1,x_poly2,axis=1)
toplam1=np.append(toplam,x_poly3,axis=1)
toplam2=np.append(toplam1,x_poly4,axis=1)
toplam3=np.append(toplam2,x_poly5,axis=1)
toplam4=np.append(toplam3,x_poly6,axis=1)
toplam5=np.append(toplam4,x_poly7,axis=1)
toplam6=np.append(toplam5,x_poly8,axis=1)
toplam7=np.append(toplam6,x_poly9,axis=1)
toplamson=np.delete(toplam7,[4,8,12,16,20,24,28,32],1)
poly_regmulti=PolynomialFeatures(degree=3)
toplam_poly=poly_regmulti.fit_transform(toplamson)
pmlr=LinearRegression()
pmlr.fit(toplam_poly,dolarvalue)
tahminmultipoly=pmlr.predict(poly_regmulti.fit_transform(toplamson))

print("********************************************************************************")
print("multi polinom:")
model_mlr11=sm.OLS(pmlr.predict(poly_regmulti.fit_transform(toplamson)),toplamson)
print(model_mlr11.fit().summary())


#SVR
#rbf
svr_reg1 = SVR(kernel='rbf')
svr_reg1.fit(son_olcekli,dolar_olcekli)

print("********************************************************************************")
print("SVR rbf:")
model_svr1=sm.OLS(svr_reg1.predict(son_olcekli),son_olcekli)
print(model_svr1.fit().summary())

#linear
svr_reg2 = SVR(kernel='linear')
svr_reg2.fit(son_olcekli,dolar_olcekli)
# predick yapılmamış
print("********************************************************************************")
print("SVR linear:")
model_svr2=sm.OLS(svr_reg2.predict(son_olcekli),son_olcekli)
print(model_svr2.fit().summary())

#poly
svr_reg3 = SVR(kernel='poly')
svr_reg3.fit(son_olcekli,dolar_olcekli)

print("********************************************************************************")
print("SVR poly:")
model_svr3=sm.OLS(svr_reg3.predict(son_olcekli),son_olcekli)
print(model_svr3.fit().summary())

#sigmoid
svr_reg4 = SVR(kernel='sigmoid')
svr_reg4.fit(son_olcekli,dolar_olcekli)

print("********************************************************************************")
print("SVR sigmoid:")
model_svr4=sm.OLS(svr_reg4.predict(son_olcekli),son_olcekli)
print(model_svr4.fit().summary())

#Decision Tree
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(son_value,dolarvalue)

print("********************************************************************************")
print("Decision Tree:")
model_dt=sm.OLS(r_dt.predict(son_value),son_value)
print(model_dt.fit().summary())

#Random Forest
rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(son_value,dolarvalue.ravel())
tahminrforest=rf_reg.predict(son_value)

print("********************************************************************************")
print("Random Forest:")
model_rf=sm.OLS(rf_reg.predict(son_value),son_value)
print(model_rf.fit().summary())


# p valuelere bakmak için backward 
# import statsmodels.api as sm
# X=np.append(arr=np.ones((93,1)).astype(int),values=son,axis=1)
# X_l=son.iloc[:,[0,1,2,3,4,5,6,7,8]].values
# x_l=np.array(X_l,dtype=float)
# model=sm.OLS(dolar,X_l).fit()
# print(model.summary())

# X_l=son.iloc[:,[0,2,3,4,5,6,7,8]].values
# x_l=np.array(X_l,dtype=float)
# model=sm.OLS(dolar,X_l).fit()
# print(model.summary())

# X_l=son.iloc[:,[0,3,4,5,6,7,8]].values
# x_l=np.array(X_l,dtype=float)
# model=sm.OLS(dolar,X_l).fit()
# print(model.summary())

# X_l=son.iloc[:,[3,4,5,6,7,8]].values
# x_l=np.array(X_l,dtype=float)
# model=sm.OLS(dolar,X_l).fit()
# print(model.summary())

korelasyon=data_tum.corr()

plt.scatter(bist,dolar)
plt.plot(bistvalue,mlr2.predict(poly_reg1.fit_transform(bistvalue)),color="green")
plt.title("bist-dolar")
plt.xlabel("bist")
plt.ylabel("dolar")
plt.show()

plt.scatter(cari_acik,dolar)
plt.plot(cari_test,lr2.predict(cari_test),color="red")
plt.plot(carivalue,mlr3.predict(poly_reg2.fit_transform(carivalue)),color="green")
plt.title("cari-dolar")
plt.xlabel("cari")
plt.ylabel("dolar")
plt.show()

plt.scatter(doviz_rezervi,dolar)
plt.plot(döviz_test,lr3.predict(döviz_test),color="red")
plt.plot(dövizvalue,mlr4.predict(poly_reg3.fit_transform(dövizvalue)),color="green")
plt.title("doviz_rezervi-dolar")
plt.xlabel("doviz_rezervi")
plt.ylabel("dolar")
plt.show()

plt.scatter(para_arzi,dolar)
plt.plot(para_test,lr4.predict(para_test),color="red")
plt.plot(paravalue,mlr5.predict(poly_reg4.fit_transform(paravalue)),color="green")
plt.title("para_arzi-dolar")
plt.xlabel("para_arzi")
plt.ylabel("dolar")
plt.show()

plt.scatter(faiz,dolar)
plt.plot(faiz_test,lr5.predict(faiz_test),color="red")
plt.plot(faizvalue,mlr6.predict(poly_reg5.fit_transform(faizvalue)),color="green")
plt.title("faiz-dolar")
plt.xlabel("faiz")
plt.ylabel("dolar")
plt.show()

plt.scatter(enflasyon,dolar)
plt.plot(enflasyon_test,lr6.predict(enflasyon_test),color="red")
plt.plot(enflasyonvalue,mlr7.predict(poly_reg6.fit_transform(enflasyonvalue)),color="green")
plt.title("enflasyon-dolar")
plt.xlabel("enflasyon")
plt.ylabel("dolar")
plt.show()

plt.scatter(ihracat,dolar)
plt.plot(ihracat_test,lr7.predict(ihracat_test),color="red")
plt.plot(ihracatvalue,mlr8.predict(poly_reg7.fit_transform(ihracatvalue)),color="green")
plt.title("ihracat-dolar")
plt.xlabel("ihracat")
plt.ylabel("dolar")
plt.show()

plt.scatter(ithalat,dolar)
plt.plot(ithalat_test,lr8.predict(ithalat_test),color="red")
plt.plot(ithalat,mlr9.predict(poly_reg8.fit_transform(ithalatvalue)),color="green")
plt.title("ithalat-dolar")
plt.xlabel("ithalat")
plt.ylabel("dolar")
plt.show()

plt.scatter(dis_ticaret,dolar)
plt.plot(dis_test,lr9.predict(dis_test),color="red")
plt.plot(disvalue,mlr10.predict(poly_reg9.fit_transform(disvalue)),color="green")
plt.title("dis_ticaret-dolar")
plt.xlabel("dis_ticaret")
plt.ylabel("dolar")
plt.show()

plt.scatter(para_arzi,dolar,color="red")
plt.scatter(para_arzi,tahminrforest,color="green")
plt.scatter(para_arzi,tahminmultipoly,color="blue")
plt.title("göstermek için")
plt.show()
import numpy as np
import pandas as pd

def similarity(a,b): ##https://www.datacamp.com/community/tutorials/fuzzy-string-python
    a = str(a)
    b = str(b)
    m = len(a)
    n = len(b)
    
    dis = np.zeros((m,n))
    
    for i in range(m):
        for j in range(n):
            if a[i-1] == b[j-1]:
                cost = 0
            else:
                cost = 2
                
            dis[i,j] = min([dis[i-1,j] + 1,
                          dis[i,j-1] + 1,
                          dis[i-1,j-1] + cost])
            
    r = ((m+n - dis[i][j]) / (m+n))
    return r   

def confidenceScoreCalc(results,
                        pernor,
                        addr,
                        nameWeight = 0.25,
                        surnameWeight = 0.25,
                        numWeight = 0.05,
                        addrWeight = 0.15,
                        addrYrWeight = 0.4,
                        postWeight = 0):
    confidenceScores = []
    
    for idx,row in results.iterrows():
        #gather data
        if pd.isna(row['address_key']) == True:
            confidenceScores.append(0)
        else:
            p = row['perno']
    
            add1 = row['addr_single_line']
            addYr1 = row['years_text']
            name1 = row['forename']
            surname1 = row['surname']
            num1 = row['telephone_number']
            mob1 = row['mobile']
    
            if pd.isna(add1) == False:
                post1 = add1.split(', ')[-1]
            else:
                post1 = 'NaN'
    
            pernod = pernor[pernor['PERNO'] == p]
            addd = addr[addr['PERNO'] == p]
    
            add2 = list(addd['ADDRESS'])
            post2 = list(addd['POSTCODE'])
            datesa = list(addd['DATE_FROM'])
            datesb = list(addd['DATE_TO'])
            addYr2 = []
            for i in range(len(datesa)):
                d1 = str(datesa[i]).split('-')[-1]
                if int(d1) > 25:
                    d1 = '19' + d1
                else:
                    d1 = '20' + d1
    
                d2 = datesb[i]
    
                if pd.isna(d2) == True:
                    addYr2.append(d1 + '-')
                else:
                    addYr2.append(d1 + '-' + str(d2))
    
            name2 = list(pernod['FIRST_NAME'])[0]
            surname2 = list(pernod['SURNAME'])[0]
            num2 = list(pernod['HOME_PHONE_NUMBER'])[0]
            mob2 = list(pernod['MOBILE_PHONE_NUMBER'])[0]
            tele2 = list(pernod['WORK_TELEPHONE_NUMBER'])[0]
    
            #run similarity chacks
            nameScore = similarity(name1,name2) * nameWeight
            surnameScore = similarity(surname1,surname2) * surnameWeight
    
            numSims = []
            i=0
            if pd.isna(num1) == False:
                if pd.isna(num2) == False:
                    numSims.append(similarity(num1,num2))
                    i+=1
                if pd.isna(mob2) == False:
                    numSims.append(similarity(num1,mob2))
                    i+=1
                if pd.isna(mob2) == False:
                    numSims.append(similarity(num1,tele2))
                    i+=1
            if pd.isna(mob1) == False:
                if pd.isna(num2) == False:
                    numSims.append(similarity(mob1,num2))
                    i+=1
                if pd.isna(mob2) == False:
                    numSims.append(similarity(mob1,mob2))
                    i+=1
                if pd.isna(mob2) == False:
                    numSims.append(similarity(mob1,tele2))
                    i+=1
    
            if i > 0:
                numScore = np.max(numSims) * numWeight
            else:
                numScore = 0
    
            addsims = []
            for a in add2:
                addsims.append(similarity(add1,a))
            addScore = np.max(addsims) * addrWeight
    
            postsims = []
            for a in post2:
                postsims.append(similarity(post1,a))
            postScore = np.max(postsims) * postWeight
    
            yrsims = []
            for a in addYr2:
                yrsims.append(similarity(addYr1,a))
            yrScore = np.max(yrsims) * addrYrWeight
            
            numResults = len(list(results[results['perno'] == p]['perno']))**0.1
    
            sim = (nameScore + surnameScore + numScore + addScore + yrScore + postScore) / numResults
    
            confidenceScores.append(sim)
            
    return confidenceScores
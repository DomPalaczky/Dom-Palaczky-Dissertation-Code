import xmltodict
import urllib3
import urllib
import pandas as pd

def APITracing(data, api_key, sampleSize = None, under18 = False, useDOB = False, useSex=False):
    
    pernos = []
    
    if under18 == False:
        #reduce to over 18s
        for idx,row in data.iterrows():
            dob = row['DOB']
            dob = dob.split('-')
            year = str(dob[2])
            if int(year) > 20:
                year = '19' + str(year)
            else:
                year = '20' + str(year)
            if int(year)<2003:
                pernos.append(row['PERNO'])
                
        datar = data[data['PERNO'].isin(pernos)]
    
    if sampleSize is not None:
        datar = datar.sample(sampleSize)
    

    
    #format DOB 
    monthDict = {'JAN': '01',
                'FEB': '02',
                'MAR': '03',
                'APR': '04',
                'MAY': '05',
                'JUN': '06',
                'JUL': '07',
                'AUG': '08',
                'SEP': '09',
                'OCT': '10',
                'NOV': '11',
                'DEC': '12'}

    resultdict = {}
    
    service_url = 'https://api.t2a.io/rest/'
    
    for idx, row in datar.iterrows():
        forename = row['FIRST_NAME']
        lastname = row['SURNAME']
        postcode = row['POSTCODE']
        if useSex == True:
            sex = row['SEX']
        if useDOB == True:
            dob = row['DOB']
            dob = dob.split('-')
            year = str(dob[2])
            if int(year) > 20:
                year = '19' + str(year)
            else:
                year = '20' + str(year)
                
            month = monthDict[dob[1]]
            day = dob[0]
        perno = row['PERNO']
        params = {'method': 'person_search',
                  'api_key': api_key,
                  'forename': forename,
                  'lastname': lastname,
                  'postcode': postcode}
        if useSex == True:
            params['sex'] = sex
        if useDOB == True:
            params['dob_y'] = year
            params['dob_m'] = month
            params['dob_d'] = day
    
        url = service_url +'?' + urllib.parse.urlencode(params)
        http = urllib3.PoolManager()
        response = http.request('GET', url)
        try:
            error = xmltodict.parse(response.data)['person_search_res']['error_code']
            if error is None:
                details = xmltodict.parse(response.data)['person_search_res']['person_list']['person']
                resultdict[perno] = details
            else:
                resultdict[perno] = {'error': error}
        except KeyError:
            details = xmltodict.parse(response.data)['person_search_res']['person_list']['person']
            result = details
            resultdict[perno] = result
            
    return resultdict

def todf(resultdict):
    l = []
    
    for resultlist in resultdict.items():
        perno = resultlist[0]
        if type(resultlist[1]) is list:
            for result in resultlist[1]:
                l.append([perno,
                          result['addr_single_line'],
                          result['address_id'],
                          result['address_key'],
                          result['years_text'],
                          result['forename'],
                          result['surname'],
                          result['telephone_number'],
                          result['mobile'],
                          result['person_id']])
        elif type(resultlist[1]) is dict:
            l.append([perno,['error'],resultdict[perno]['error']])
        else:
            result = resultlist[1]
            l.append([perno,
                      result['addr_single_line'],
                      result['address_id'],
                      result['address_key'],
                      result['years_text'],
                      result['forename'],
                      result['surname'],
                      result['telephone_number'],
                      result['mobile'],
                      result['person_id']])
    
    df = pd.DataFrame(l)
    
    df.columns = ['perno','addr_single_line','address_id','address_key',
                  'years_text','forename','surname','telephone_number',
                  'mobile','person_id']
    
    return df
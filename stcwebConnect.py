import requests

# select planID and offset
def plan_selection(token, tcID, PLAN_ID, stcweb_url):
    url = stcweb_url + 'tc/' + str(tcID) + '/plan/' + str(PLAN_ID)
    headers = {'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + token}
    response = requests.put(url, headers = headers)

    return response
     
# select planID, cycle, offset and stage duration
def plan_params(token, tcID, PLAN_ID, CYCLE_TIME, OFFSET, SYNC, stages, stcweb_url):
    url = stcweb_url+'tc/'+str(tcID)+'/planparams/'+str(PLAN_ID)+'/'+str(CYCLE_TIME)+'/'+str(OFFSET)+'/'+str(SYNC)+'/?stages='+str(stages)
    headers = {'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + token}
    response = requests.put(url, headers = headers)

    return response

# login to STCWeb to recieve token
def stcweb_login(email, password, stcweb_url):
    url = stcweb_url + 'login'
    data = {'email': email, 'password': password}
    headers = None
    response = requests.post(url, data = data, headers = headers)
    stc_token = response.json()
        
    return stc_token
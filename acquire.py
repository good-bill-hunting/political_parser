import pandas as pd
import numpy as np
from env import api_key
import requests
import json

def get_links_to_bills():
    """
    This function gets the links to the full text of bills from the gov info api.
    """
    #Builds the url using your api key.
    url = f"https://api.govinfo.gov/collections/BILLS/2001-01-01T01%3A01%3A01Z/2023-01-01T01%3A01%3A01Z?pageSize=1000&offsetMark=%2A{api_key}"
    
    #Gets each page
    response = requests.get(url)
    
    #Converts from json
    data = response.json()
    
    #Makes list to collect links
    list_of_links = []
    
    #Loop to collect links from first page
    for item in data['packages']:
        list_of_links.append(item['packageLink'])
        
    #Sets the variable to break the while loop
    next_page = data['nextPage']
    
    #While loop to collect the remaining links
    while next_page != None: #first json coming in, next_page url coming in.

        nextPage_link = next_page + "&api_key=" + f'{api_key}' #Creates new url 
        response = requests.get(nextPage_link) #Gets new info
        data = response.json() #Makes new info readable
        next_page = data['nextPage'] #nextpage link assigned

        for item in data['packages']:
            list_of_links.append(item['packageLink'])
            
    return list_of_links
import requests
from bs4 import BeautifulSoup

def get_object_names():
    '''getting all the object names from the web page'''
    r = requests.get("http://rll.berkeley.edu/bigbird/aliases/863afb5e73/")
    data = r.text
    soup = BeautifulSoup(data)

    names = []
    for tdd in soup.find_all('td'):
        if 'class' in tdd.attrs and tdd.attrs['class'][0] == 'name_cell':
            names.append(tdd.contents[0].strip())
    return names

names =  get_object_names()

for name in names:
    print name.strip()
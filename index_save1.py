import requests, json, os
from elasticsearch import Elasticsearch
res = requests.get('http://localhost:9200')
print (res.content)
directory = '/home/administrator/PycharmProjects/Internship/Intern/YOLOv3/demog'
es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
i = 1
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        f = open(filename)
        docket_content = f.read()
        # Send the data into es
        es.index(index='detection1', ignore=400, doc_type='doc',
        id=i, body=json.loads(docket_content))
        i = i + 1
print('success')



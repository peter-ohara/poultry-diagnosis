import requests

IN_PRODUCTION = False

if IN_PRODUCTION:
    API_URL = 'https://chicken-tinder.herokuapp.com'
else:
    API_URL = 'http://127.0.0.1:8000'

API_KEY = 'i0cgsdYL3hpeOGkoGmA2TxzJ8LbbU1HpbkZo8B3kFG2bRKjx3V'

headers = {'UserAPI-Key': API_KEY}

# response = requests.get('{}/files'.format(API_URL), headers=headers)
# print(response.json())

with open('chicken.jpg', 'rb') as fp:
    content = fp.read()

response = requests.post(
    '{}/classify_image'.format(API_URL),
    headers=headers,
    json={'url': 'https://www.almanac.com/sites/default/files/styles/primary_image_in_article/public/image_nodes/daffodil.jpg?itok=KSBgH9y9'}
)

print(response.text)

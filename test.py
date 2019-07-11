import requests

IN_PRODUCTION = False

if IN_PRODUCTION:
    API_URL = 'https://chicken-tinder.herokuapp.com'
else:
    API_URL = 'http://127.0.0.1:8000'

API_KEY = 'i0cgsdYL3hpeOGkoGmA2TxzJ8LbbU1HpbkZo8B3kFG2bRKjx3V'

headers = {'UserAPI-Key': API_KEY}

response = requests.get('{}/files'.format(API_URL), headers=headers)

print(response.json())

with open('test_file.png', 'rb') as fp:
    content = fp.read()

response = requests.post(
    '{}/classify_image'.format(API_URL),
    headers=headers,
    json={'url': 'https://res.cloudinary.com/tenderswift/image/upload/landing-page/horizontal-logo-inverse.png'}
)

print(response.text)

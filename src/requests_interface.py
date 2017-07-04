import requests

'''
Helper methods for url requests
'''
USER_AGENT = (
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 '
    + '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
)

def detect_redirection(response):
    if response.history:
        print("Request was redirected")
        for resp in response.history:
            print(resp.status_code, resp.url)
        print("Final destination:")
        print(response.status_code, response.url)
    else:
        print("Request was not redirected")

def get(url, params=None, headers=None):
    s = requests.Session()
    s.headers['User-Agent'] = USER_AGENT
    response = s.get(url, params=params, headers=headers)

    if response.status_code == 400:
        return response

    num_retries = 10
    num = 1
    while num <= 10 and response.status_code != 200:
        print("WARNING", response.status_code, 'Retry #{}'.format(num))
        response = s.get(url, params=params, headers=headers)
        num = num + 1

    if response.status_code != 200:
        print("WARNING", response.status_code, response.text)
    else:
        return response

def post(url, params=None, headers=None):
    s = requests.Session()
    s.headers['User-Agent'] = USER_AGENT
    response = s.post(url, params=params, headers=headers)

    num_retries = 10
    num = 1
    while num <= 10 and response.status_code != 200:
        print("WARNING", response.status_code, 'Retry #{}'.format(num))
        response = s.post(url, params=params, headers=headers)
        num = num + 1

    if response.status_code != 200:
        print("WARNING", response.status_code, response.text)
    else:
        return response

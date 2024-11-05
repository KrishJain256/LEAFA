import requests

def get_IP():
    ip_address = requests.get('https://api64.ipify.org?format=json').json()
    return ip_address["ip"]

import requests
import keepa
import time
from datetime import date
import json

api_endpoint = 'https://api.keepa.com/query?domain=1&key=ca82bdg9afv8kf35b23t5813tgnpeb6jmck42pjq34j2pd2n39u7mg5estjcb4do'
api = keepa.Keepa("ca82bdg9afv8kf35b23t5813tgnpeb6jmck42pjq34j2pd2n39u7mg5estjcb4do")
base_id = "applWV4PtK1OiEbS4"
table_id_or_name = "tblHEM0cyX7qV3r0N"


parms = {
    "monthlySold_gte": 250,
    "current_BUY_BOX_SHIPPING_gte": 1500,
    "deltaPercent7_BUY_BOX_SHIPPING_gte": 35,
    "deltaPercent90_BUY_BOX_SHIPPING_gte": 35,
    "brand": [
        "✜amazon"
    ],
    "sort": [
        [
            "current_SALES",
            "asc"
        ]
    ],
    "productType": [
        0,
        1,
        2
    ],
    "perPage": 100,
    "page": 0

}



def query_api():
        response = requests.post(api_endpoint, json=parms)
        asin_list = response.json().get('asinList', [])
        for product in asin_list:
            with open('all_asins.txt', 'a') as file:
                file.write(product + '\n')

def read_file_to_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return [line.strip() for line in lines]

def get_unique_asins(prior_asins, current_asins):
    unique_asins = {}
    for asin in current_asins:
        if asin not in prior_asins:
            unique_asins[asin] = True
    return unique_asins

def compare_asins():
    prior_asins = read_file_to_list('prior_asins.txt')
    current_asins = read_file_to_list('all_asins.txt')
    
    unique_asins = get_unique_asins(prior_asins, current_asins)
    for asin in unique_asins:
        add_record(asin)

                    
def add_record(asin):
    api.wait_for_tokens()
    product_info = api.query(asin, history=False)[0]
    category_tree = product_info.get('categoryTree')
    product_name = product_info.get('title')
    product_category = category_tree[0]['name']
    brand = product_info.get('brand')

    time.sleep(2)
    api.wait_for_tokens()
    token = "pat7dKdVUXa2sTH35.492d2d4be222c0e9fc575e72be5720827e9726c958152dc7b68d9c01c7995f1b"

    # API endpoint
    url = f"https://api.airtable.com/v0/{base_id}/{table_id_or_name}"

    # Request headers
    headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
    }

    current_date = date.today().isoformat()
    
    payload = {
        "records": [
            {
                "fields": {
                    "ASIN": asin if isinstance(asin, str) else list(asin),
                    "Product Name": product_name if isinstance(product_name, str) else list(product_name),
                    "Brand": brand if isinstance(brand, str) else list(brand),
                    "Linked Brand": brand if isinstance(brand, str) else list(brand),
                    "Category": product_category if isinstance(product_category, str) else list(product_category),
                    "Date Added": current_date
                }
            }
        ],
        "typecast": True
    }
    
    time.sleep(2)
    # Make the POST request
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Check the response
    if response.status_code == 200:
        print("Request successful!")
        print(response.json())
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)

    
def main():
    with open('all_asins.txt', 'r') as all_asins_file:
        with open('prior_asins.txt', 'w') as prior_asins_file:
            prior_asins_file.write(all_asins_file.read())
    
    open('all_asins.txt', 'w').close()
    query_api()
    compare_asins()
    
main()
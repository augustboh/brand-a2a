import keepa
import requests
import time
import json
from datetime import date
from itertools import cycle
from requests.exceptions import RequestException


# List of API keys
api_keys = [
    "3aat0atkip5dp1gp7jr2js3ghpefkjvl68q7av1ncga0d6ipcd4cn2k7chqlf5b0",
    "ca82bdg9afv8kf35b23t5813tgnpeb6jmck42pjq34j2pd2n39u7mg5estjcb4do",
    "73hq89k6akkfd9ou2bs2snsiaip7kcm958bo79p3hmq0lf1a2m0l0ajiddp3vj8u",
]
api_key_cycle = cycle(api_keys)

file_path = "./sellers.txt"
base_id = "applWV4PtK1OiEbS4"
table_id_or_name = "tblHEM0cyX7qV3r0N"

api = None

def initialize_api():
    global api
    current_key = next(api_key_cycle)
    api = keepa.Keepa(current_key)

initialize_api()

def wait_for_tokens_with_timeout(timeout=5):
    start_time = time.time()
    while True:
        if api.tokens_left:
            return True
        if time.time() - start_time > timeout:
            return False
        time.sleep(0.1)

def get_seller_ids_from_file(filepath):
    with open(filepath, "r") as file:
        return [line.strip() for line in file]

def get_product_info(asin, max_retries=5, delay=15):
    for attempt in range(max_retries):
        try:
            api.wait_for_tokens()
            product_info = api.query(asin, history=False)[0]
            
            # Get product name (title)
            product_name = product_info.get('title')
            
            # Get brand
            brand = product_info.get('brand')
            
            # Get root category
            category_tree = product_info.get('categoryTree')
            root_category = None
            if category_tree and len(category_tree) > 0:
                root_category = category_tree[0].get('name')
            
            return [product_name, root_category, brand]
        
        except Exception as e:
            print(f"Error occurred for ASIN {asin}. Attempt {attempt + 1} of {max_retries}. Error: {str(e)}")
            if "token" in str(e).lower() or "limit" in str(e).lower():
                print("API error related to tokens. Switching to next API key.")
                initialize_api()
            elif attempt < max_retries - 1:
                print(f"Waiting for {delay} seconds before retrying...")
                time.sleep(delay)
            else:
                print(f"Max retries reached for ASIN {asin}. Returning None values.")
                return [None, None, None]

    return [None, None, None]  # This line should never be reached, but it's here for completeness

def query_API(seller_id):
    global api
    seller_asins = set()
    max_retries = len(api_keys)
    for _ in range(max_retries):
        if not wait_for_tokens_with_timeout():
            print("Token wait timeout. Switching to next API key.")
            initialize_api()
            continue
        
        try:
            seller_info = api.seller_query(seller_id, "US", storefront=True)

            for info in seller_info.values():
                asinlist_value = info.get('asinList')
                seller_name = info.get('sellerName')

                if asinlist_value is not None:
                    for asin in asinlist_value:
                        seller_asins.add(asin)
            
            with open(f"./seller_asins/{seller_id}.txt", "w") as file:
                for asin in seller_asins:
                    file.write(asin + "\n")
            return [seller_name, seller_asins]
        except keepa.ApiError as e:
            print(f"API Error encountered: {str(e)}")
            if "token" in str(e).lower() or "limit" in str(e).lower():
                print("API error related to tokens. Switching to next API key.")
                initialize_api()
            else:
                raise
    raise Exception("All API keys exhausted. Unable to complete the request.")

def load_seller_from_file(seller_id):
    seller_asins = set()
    try:
        with open(f"./seller_asins/{seller_id}.txt", "r") as file:
            for line in file:
                seller_asins.add(line.strip())
    except FileNotFoundError:
        pass  # File doesn't exist yet, return an empty set
    return seller_asins

def check_asin_changes(previous_asins, current_asins, seller_id, seller_name):
    new_asins = current_asins - previous_asins
    if new_asins:
        for asin in new_asins:
            product_name, product_category, product_brand = get_product_info(asin)
            
            # Use default values if information is not available
            product_name = product_name or "Error scraping product name"
            product_category = product_category or "Error scraping product category"
            product_brand = product_brand or "Error scraping product brand"
            
            add_record(asin, product_name, product_category, seller_name, product_brand)

def add_record(asin, product_name, product_category, seller_name, brand):
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
                    "Seller": seller_name if isinstance(seller_name, str) else list(seller_name),
                    "Linked Seller": seller_name if isinstance(seller_name, str) else list(seller_name),
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

def check_asin_changes_for_sellers():
    seller_ids = get_seller_ids_from_file(file_path)
    for seller_id in seller_ids:
        old_asins = load_seller_from_file(seller_id)
        seller_name, new_asins = query_API(seller_id)
        check_asin_changes(old_asins, new_asins, seller_id, seller_name)

check_asin_changes_for_sellers()

print("ASIN check completed.")
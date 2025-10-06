import requests
import keepa
import time
from datetime import date, datetime, timezone
import json
from requests.exceptions import ReadTimeout
from urllib3.exceptions import ReadTimeoutError
from itertools import cycle

api_keys = [
    "cuc7bq8dcfholhkeep8nnh1gf3khc9tv0f4pc3eetglsq3l1p5njcn186vje2jhd",
    "3aat0atkip5dp1gp7jr2js3ghpefkjvl68q7av1ncga0d6ipcd4cn2k7chqlf5b0",
    "ca82bdg9afv8kf35b23t5813tgnpeb6jmck42pjq34j2pd2n39u7mg5estjcb4do",
    "73hq89k6akkfd9ou2bs2snsiaip7kcm958bo79p3hmq0lf1a2m0l0ajiddp3vj8u",
    "cc843n075pek8009m6n4a8vhvtr44vnaekj4qh75238soj0eegh68h54ucaploj8",
    "2rse0tvnecmdsmm393nj13aqbk3ce2duoe77s084u375jvndm9r1t02j19raa871",
    "1aqrvfb2f6njbdfpish2ccn81dajcnvs8j1o8f90an635oudg2mpvmojgl17n0mm"
]
api_key_cycle = cycle(api_keys)
api = None
current_key = None

base_id = "applWV4PtK1OiEbS4"
table_id_or_name = "tblHEM0cyX7qV3r0N"

a2a_parms = {
    "monthlySold_gte": 400,
    "current_BUY_BOX_SHIPPING_gte": 1500,
    "deltaPercent7_BUY_BOX_SHIPPING_gte": 25,
    "deltaPercent90_BUY_BOX_SHIPPING_gte": 25,
    "current_AMAZON_gte": 1500,
    "brand": ["✜amazon"],
    "sort": [["current_SALES", "asc"]],
    "productType": [0, 1, 2],
    "perPage": 2000,
    "page": 0
}

sub_and_save_parms = {
    "monthlySold_gte": 300,
    "current_BUY_BOX_SHIPPING_gte": 1500,
    "deltaPercent7_BUY_BOX_SHIPPING_gte": 25,
    "deltaPercent30_BUY_BOX_SHIPPING_gte": 25,
    "deltaPercent90_BUY_BOX_SHIPPING_gte": 25,
    "current_AMAZON_gte": 1500,
    "couponSNSPercent_gte": 10,
    "sort": [["monthlySold", "desc"]],
    "lastOffersUpdate_gte": 7238128,
    "productType": [0, 1, 2],
    "page": 0,
    "perPage": 100
}

electronics_parms = {
    "monthlySold_gte": 200,
    "current_BUY_BOX_SHIPPING_gte": 1500,
    "deltaPercent7_BUY_BOX_SHIPPING_gte": 13,
    "deltaPercent90_BUY_BOX_SHIPPING_gte": 10,
    "current_AMAZON_gte": 1500,
    "rootCategory": [
        "172282"
    ],
    "brand": ["✜amazon"],
    "sort": [["current_SALES", "asc"]],
    "productType": [0, 1, 2],
    "perPage": 2000,
    "page": 0
}

param_set = [a2a_parms, sub_and_save_parms]

def initialize_api():
    global api, current_key
    while True:
        current_key = next(api_key_cycle)
        print(f"[API] Trying key: {current_key[:8]}...")
        try:
            api = keepa.Keepa(current_key)
            # Use tokens_left directly instead of ping
            if api.tokens_left > 20:
                print(f"[API] Success! Tokens available: {api.tokens_left}")
                return True
            print(f"[API] No tokens available for key {current_key[:8]}")
        except Exception as e:
            print(f"[API] Error with key {current_key[:8]}: {str(e)}")
        time.sleep(1)

def check_and_rotate_api_key():
    if not api or api.tokens_left < 2:
        return initialize_api()
    return True

def wait_for_tokens_with_timeout(timeout=30):
    start_time = time.time()
    keys_tried = set()
    
    while len(keys_tried) < len(api_keys):
        if time.time() - start_time > timeout:
            return False
        
        if check_and_rotate_api_key():
            return True
            
        if current_key:
            keys_tried.add(current_key)
        time.sleep(1)
    
    return False

def query_api(params):
    if not check_and_rotate_api_key():
        return set()
        
    api_endpoint = f'https://api.keepa.com/query?domain=1&key={current_key}'
    try:
        response = requests.post(api_endpoint, json=params)
        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.text}")
            return set()
        return set(response.json().get('asinList', []))
    except Exception as e:
        print(f"Query error: {str(e)}")
        return set()

def get_product_info(asin, max_retries=3, retry_delay=5):
    for attempt in range(max_retries):
        try:
            check_and_rotate_api_key()
            api.wait_for_tokens()
            return api.query(asin, history=False)[0]
        except (ReadTimeout, ReadTimeoutError) as e:
            if attempt < max_retries - 1:
                print(f"Timeout error for ASIN {asin}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached for ASIN {asin}. Skipping.")
        except Exception as e:
            print(f"Error for ASIN {asin}: {str(e)}")
            if "token" in str(e).lower():
                initialize_api()
                continue
        return None

def add_record_to_airtable(asin, product_info):
    print(f"Adding record for ASIN: {asin}")
    category_tree = product_info.get('categoryTree')
    product_name = product_info.get('title')
    product_category = category_tree[0]['name'] if category_tree else 'Unknown'
    brand = product_info.get('brand')
    bilm = product_info.get('monthlySold')    
    who_is_seller = "Other" if product_info.get('availabilityAmazon') == -1 else "Amazon"

    check_and_rotate_api_key()
    api.wait_for_tokens()
    
    token = "pat7dKdVUXa2sTH35.492d2d4be222c0e9fc575e72be5720827e9726c958152dc7b68d9c01c7995f1b"
    url = f"https://api.airtable.com/v0/{base_id}/{table_id_or_name}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    current_date_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    
    payload = {
        "records": [{
            "fields": {
                "ASIN": asin,
                "Product Name": product_name,
                "Brand": brand,
                "Linked Brand": brand,
                "Category": product_category,
                "Date Added": current_date_time,
                "BILM": bilm,
                "Seller": who_is_seller
            }
        }],
        "typecast": True
    }
    
    time.sleep(2)
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        print(f"Record added successfully for ASIN: {asin}")
    else:
        print(f"Failed to add record for ASIN: {asin}. Status: {response.status_code}")
        print(response.text)

def read_file_to_set(filename):
    try:
        with open(filename, 'r') as file:
            return set(line.strip() for line in file)
    except FileNotFoundError:
        return set()

def write_set_to_file(filename, data_set):
    with open(filename, 'w') as file:
        for item in data_set:
            file.write(f"{item}\n")

def main():
    initialize_api()
    prior_asins = read_file_to_set('prior_asins.txt')
    
    all_current_asins = set()
    for params in param_set:
        if wait_for_tokens_with_timeout():
            current_asins = query_api(params)
            all_current_asins.update(current_asins)
        else:
            print("Failed to get tokens for query")
    
    new_asins = all_current_asins - prior_asins
    print(f"Total new ASINs: {len(new_asins)}")
    
    for asin in new_asins:
        product_info = get_product_info(asin)
        if product_info:
            print(f"Processing ASIN: {asin}")
            time.sleep(5)
            add_record_to_airtable(asin, product_info)
    
    write_set_to_file('prior_asins.txt', all_current_asins)
    print(f"Added {len(new_asins)} new ASINs to Airtable")
    print(f"Removed {len(prior_asins - all_current_asins)} ASINs from prior_asins")
    print("Process completed")

if __name__ == "__main__":
    main()

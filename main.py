import requests
import keepa
import time
from datetime import date
import json
from requests.exceptions import ReadTimeout
from urllib3.exceptions import ReadTimeoutError

# API configurations
api_endpoint = 'https://api.keepa.com/query?domain=1&key=ca82bdg9afv8kf35b23t5813tgnpeb6jmck42pjq34j2pd2n39u7mg5estjcb4do'
api = keepa.Keepa("ca82bdg9afv8kf35b23t5813tgnpeb6jmck42pjq34j2pd2n39u7mg5estjcb4do")
base_id = "applWV4PtK1OiEbS4"
table_id_or_name = "tblHEM0cyX7qV3r0N"

# API parameters
a2a_parms = {
    "monthlySold_gte": 100,
    "current_BUY_BOX_SHIPPING_gte": 1500,
    "deltaPercent7_BUY_BOX_SHIPPING_gte": 35,
    "deltaPercent90_BUY_BOX_SHIPPING_gte": 35,
    "brand": ["✜amazon"],
    "sort": [["current_SALES", "asc"]],
    "productType": [0, 1, 2],
    "perPage": 2000,
    "page": 0
}

lightning_deal_parms = {
    "monthlySold_gte": 100,
    "current_BUY_BOX_SHIPPING_gte": 1800,
    "current_LIGHTNING_DEAL_gte": 100,
    "sort": [["monthlySold", "desc"]],
    "productType": [0, 1, 2],
    "page": 0,
    "perPage": 2000
}

sub_and_save_parms = {
    "monthlySold_gte": 300,
    "current_BUY_BOX_SHIPPING_gte": 1500,
    "deltaPercent7_BUY_BOX_SHIPPING_gte": 25,
    "deltaPercent30_BUY_BOX_SHIPPING_gte": 25,
    "deltaPercent90_BUY_BOX_SHIPPING_gte": 25,
    "couponSNSPercent_gte": 10,
    "sort": [
        [
            "monthlySold",
            "desc"
        ]
    ],
    "lastOffersUpdate_gte": 7238128,
    "productType": [
        0,
        1,
        2
    ],
    "page": 0,
    "perPage": 100
}

param_set = [a2a_parms, sub_and_save_parms]

def calculate_price_drop(product_info):
    lightning_price = product_info.get('stats_parsed', {}).get('current', {}).get('LIGHTNING_DEAL')
    list_price = product_info.get('stats_parsed', {}).get('avg30', {}).get('LISTPRICE')
    
    if lightning_price is not None and list_price is not None and list_price != 0:
        return lightning_price / list_price
    else:
        return 0  # Return 1 if we can't calculate the price drop

def read_file_to_set(filename):
    with open(filename, 'r') as file:
        return set(line.strip() for line in file)

def write_set_to_file(filename, data_set):
    with open(filename, 'w') as file:
        for item in data_set:
            file.write(f"{item}\n")

def query_api(params):
    response = requests.post(api_endpoint, json=params)
    return set(response.json().get('asinList', []))

def get_product_info(asin, max_retries=3, retry_delay=5):
    for attempt in range(max_retries):
        try:
            return api.query(asin, history=False)[0]
        except (ReadTimeout, ReadTimeoutError) as e:
            if attempt < max_retries - 1:
                print(f"Timeout error occurred for ASIN {asin}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached for ASIN {asin}. Skipping this ASIN.")
                return None
        except Exception as e:
            print(f"An unexpected error occurred for ASIN {asin}: {str(e)}")
            return None


def add_record_to_airtable(asin, product_info):
    print(f"Adding record for ASIN: {asin}")
    category_tree = product_info.get('categoryTree')
    product_name = product_info.get('title')
    product_category = category_tree[0]['name'] if category_tree else 'Unknown'
    brand = product_info.get('brand')
    bilm = product_info.get('monthlySold')    
    who_is_seller = "Other" if product_info.get('availabilityAmazon') == -1 else "Amazon"

    time.sleep(2)
    api.wait_for_tokens()
    token = "pat7dKdVUXa2sTH35.492d2d4be222c0e9fc575e72be5720827e9726c958152dc7b68d9c01c7995f1b"

    url = f"https://api.airtable.com/v0/{base_id}/{table_id_or_name}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    current_date = date.today().isoformat()
    
    payload = {
        "records": [
            {
                "fields": {
                    "ASIN": asin,
                    "Product Name": product_name,
                    "Brand": brand,
                    "Linked Brand": brand,
                    "Category": product_category,
                    "Date Added": current_date,
                    "BILM": bilm,
                    "Seller": who_is_seller
                }
            }
        ],
        "typecast": True
    }
    
    time.sleep(2)
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        print(f"Record added successfully for ASIN: {asin}")
    else:
        print(f"Failed to add record for ASIN: {asin}. Status code: {response.status_code}")
        print(response.text)

def main():
    # Read prior ASINs
    prior_asins = read_file_to_set('prior_asins.txt')
    
    all_new_asins = set()
    for params in param_set:
        new_asins = query_api(params)
        all_new_asins.update(new_asins)
    
    all_new_asins -= prior_asins
    print(f"Total new ASINs: {len(all_new_asins)}")
    
    # Process new ASINs
    valid_asins = set()
    for asin in all_new_asins:
        product_info = get_product_info(asin)
        
        if product_info is None:
            print(f"Skipping ASIN {asin} due to error in retrieving product info")
            continue
        
        # if asin in lightning_asins:
        #     price_drop = calculate_price_drop(product_info)
        #     if price_drop < 0.7:
        #         valid_asins.add(asin)
        #         print(f"ASIN {asin} qualifies for lightning deal (price drop: {price_drop:.2f})")
        #     else:
        #         print(f"ASIN {asin} does not qualify for lightning deal (price drop: {price_drop:.2f})")
        # else:
        valid_asins.add(asin)
        print(f"ASIN {asin} qualifies for a2a parameters")
        
        # Add to Airtable
        add_record_to_airtable(asin, product_info)
    
    # Update prior_asins and all_asins files
    prior_asins |= valid_asins
    write_set_to_file('prior_asins.txt', prior_asins)
    write_set_to_file('all_asins.txt', valid_asins)
    
    print(f"Added {len(valid_asins)} new ASINs to Airtable")
    print("Process completed")

if __name__ == "__main__":
    main()

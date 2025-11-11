"""
Main application module for Brand A2A product tracking.

This module handles querying Keepa API for Amazon products and
adding new products to Airtable.
"""
import json
import logging
import time
from datetime import datetime, timezone
from itertools import cycle
from typing import Dict, List, Optional, Set, Any

import keepa
import requests
from requests.exceptions import ReadTimeout
from urllib3.exceptions import ReadTimeoutError

from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KeepaAPIManager:
    """Manages Keepa API connections and key rotation."""
    
    def __init__(self, api_keys: List[str]) -> None:
        """
        Initialize the Keepa API manager.
        
        Args:
            api_keys: List of Keepa API keys to rotate through
        """
        if not api_keys:
            raise ValueError("At least one API key is required")
        
        self.api_keys = api_keys
        self.api_key_cycle = cycle(api_keys)
        self.api: Optional[keepa.Keepa] = None
        self.current_key: Optional[str] = None
    
    def initialize_api(self) -> bool:
        """
        Initialize the Keepa API connection with available tokens.
        
        Returns:
            True if initialization successful, False otherwise
        """
        keys_tried = set()
        
        while len(keys_tried) < len(self.api_keys):
            self.current_key = next(self.api_key_cycle)
            
            if self.current_key in keys_tried:
                continue
                
            logger.info(f"Trying API key: {self._mask_key(self.current_key)}")
            
            try:
                self.api = keepa.Keepa(self.current_key)
                
                if self.api.tokens_left > Config.MIN_TOKENS_REQUIRED:
                    logger.info(
                        f"API initialized successfully. "
                        f"Tokens available: {self.api.tokens_left}"
                    )
                    return True
                
                logger.warning(
                    f"Insufficient tokens for key {self._mask_key(self.current_key)}. "
                    f"Available: {self.api.tokens_left}"
                )
            except Exception as e:
                logger.error(
                    f"Error initializing API with key {self._mask_key(self.current_key)}: {e}"
                )
            
            keys_tried.add(self.current_key)
            time.sleep(1)
        
        logger.error("Failed to initialize API with any available key")
        return False
    
    def check_and_rotate_api_key(self) -> bool:
        """
        Check if API has sufficient tokens and rotate if needed.
        
        Returns:
            True if API is ready, False otherwise
        """
        if not self.api or self.api.tokens_left < Config.MIN_TOKENS_FOR_QUERY:
            logger.info("Rotating API key due to insufficient tokens")
            return self.initialize_api()
        return True
    
    def wait_for_tokens_with_timeout(self, timeout: int = None) -> bool:
        """
        Wait for API tokens to become available with timeout.
        
        Args:
            timeout: Maximum time to wait in seconds (uses Config default if None)
        
        Returns:
            True if tokens available, False if timeout
        """
        if timeout is None:
            timeout = Config.TOKEN_WAIT_TIMEOUT
        
        start_time = time.time()
        keys_tried = set()
        
        while len(keys_tried) < len(self.api_keys):
            if time.time() - start_time > timeout:
                logger.error(f"Timeout waiting for tokens after {timeout} seconds")
                return False
            
            if self.check_and_rotate_api_key():
                return True
            
            if self.current_key:
                keys_tried.add(self.current_key)
            time.sleep(1)
        
        return False
    
    def query_api(self, params: Dict[str, Any]) -> Set[str]:
        """
        Query the Keepa API for product ASINs.
        
        Args:
            params: Query parameters for the Keepa API
        
        Returns:
            Set of ASIN strings returned by the API
        """
        if not self.check_and_rotate_api_key():
            logger.error("Cannot query API: no available keys")
            return set()
        
        api_endpoint = (
            f'https://api.keepa.com/query?domain={Config.KEEPA_DOMAIN}'
            f'&key={self.current_key}'
        )
        
        try:
            response = requests.post(api_endpoint, json=params, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            asin_list = result.get('asinList', [])
            logger.info(f"Query returned {len(asin_list)} ASINs")
            return set(asin_list)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API query error: {e}")
            return set()
        except Exception as e:
            logger.error(f"Unexpected error during API query: {e}")
            return set()
    
    def get_product_info(
        self, 
        asin: str, 
        max_retries: int = None, 
        retry_delay: int = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get product information for a given ASIN.
        
        Args:
            asin: Amazon ASIN to query
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        
        Returns:
            Product information dictionary or None if failed
        """
        if max_retries is None:
            max_retries = Config.MAX_RETRIES
        if retry_delay is None:
            retry_delay = Config.RETRY_DELAY
        
        for attempt in range(max_retries):
            try:
                if not self.check_and_rotate_api_key():
                    logger.error(f"Cannot get product info for {asin}: no available keys")
                    return None
                
                self.api.wait_for_tokens()
                product_data = self.api.query(asin, history=False)
                
                if product_data and len(product_data) > 0:
                    return product_data[0]
                
                logger.warning(f"No product data returned for ASIN: {asin}")
                return None
                
            except (ReadTimeout, ReadTimeoutError) as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Timeout error for ASIN {asin}. "
                        f"Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Max retries reached for ASIN {asin}. Skipping.")
            except Exception as e:
                error_msg = str(e).lower()
                if "token" in error_msg:
                    logger.warning(f"Token error for ASIN {asin}. Reinitializing API...")
                    self.initialize_api()
                    continue
                else:
                    logger.error(f"Error getting product info for ASIN {asin}: {e}")
        
        return None
    
    @staticmethod
    def _mask_key(key: str) -> str:
        """Mask API key for logging (show first 8 characters only)."""
        return f"{key[:8]}..." if key and len(key) > 8 else "***"


class AirtableManager:
    """Manages Airtable API interactions."""
    
    def __init__(
        self, 
        base_id: str, 
        table_id: str, 
        api_token: str
    ) -> None:
        """
        Initialize Airtable manager.
        
        Args:
            base_id: Airtable base ID
            table_id: Airtable table ID
            api_token: Airtable API token
        """
        self.base_id = base_id
        self.table_id = table_id
        self.api_token = api_token
        self.url = f"https://api.airtable.com/v0/{base_id}/{table_id}"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
    
    def add_record(self, asin: str, product_info: Dict[str, Any]) -> bool:
        """
        Add a product record to Airtable.
        
        Args:
            asin: Amazon ASIN
            product_info: Product information dictionary from Keepa
        
        Returns:
            True if record added successfully, False otherwise
        """
        logger.info(f"Adding record for ASIN: {asin}")
        
        try:
            category_tree = product_info.get('categoryTree', [])
            product_name = product_info.get('title', 'Unknown')
            product_category = (
                category_tree[0]['name'] 
                if category_tree and len(category_tree) > 0 
                else 'Unknown'
            )
            brand = product_info.get('brand', 'Unknown')
            bilm = product_info.get('monthlySold', 0)
            who_is_seller = (
                "Other" 
                if product_info.get('availabilityAmazon') == -1 
                else "Amazon"
            )
            
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
            
            time.sleep(Config.API_REQUEST_DELAY)
            
            response = requests.post(
                self.url, 
                headers=self.headers, 
                data=json.dumps(payload),
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"Record added successfully for ASIN: {asin}")
                return True
            else:
                logger.error(
                    f"Failed to add record for ASIN: {asin}. "
                    f"Status: {response.status_code}, Response: {response.text}"
                )
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error adding record for ASIN {asin}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error adding record for ASIN {asin}: {e}")
            return False


class FileManager:
    """Manages file operations for ASIN tracking."""
    
    @staticmethod
    def read_file_to_set(filename: str) -> Set[str]:
        """
        Read ASINs from a file into a set.
        
        Args:
            filename: Path to the file containing ASINs (one per line)
        
        Returns:
            Set of ASIN strings
        """
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                asins = {line.strip() for line in file if line.strip()}
                logger.info(f"Loaded {len(asins)} ASINs from {filename}")
                return asins
        except FileNotFoundError:
            logger.warning(f"File {filename} not found. Starting with empty set.")
            return set()
        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}")
            return set()
    
    @staticmethod
    def write_set_to_file(filename: str, data_set: Set[str]) -> None:
        """
        Write a set of ASINs to a file.
        
        Args:
            filename: Path to the output file
            data_set: Set of ASIN strings to write
        """
        try:
            with open(filename, 'w', encoding='utf-8') as file:
                for item in sorted(data_set):
                    file.write(f"{item}\n")
            logger.info(f"Wrote {len(data_set)} ASINs to {filename}")
        except Exception as e:
            logger.error(f"Error writing file {filename}: {e}")
            raise


def main() -> None:
    """Main application entry point."""
    try:
        # Validate configuration
        Config.validate()
        logger.info("Configuration validated successfully")
        
        # Initialize managers
        keepa_manager = KeepaAPIManager(Config.KEEPA_API_KEYS)
        airtable_manager = AirtableManager(
            Config.AIRTABLE_BASE_ID,
            Config.AIRTABLE_TABLE_ID,
            Config.AIRTABLE_API_TOKEN
        )
        file_manager = FileManager()
        
        # Initialize API
        if not keepa_manager.initialize_api():
            logger.error("Failed to initialize Keepa API. Exiting.")
            return
        
        # Load prior ASINs
        prior_asins = file_manager.read_file_to_set(Config.PRIOR_ASINS_FILE)
        
        # Query for current ASINs
        all_current_asins: Set[str] = set()
        query_params = Config.get_query_params()
        
        for params in query_params:
            if keepa_manager.wait_for_tokens_with_timeout():
                current_asins = keepa_manager.query_api(params)
                all_current_asins.update(current_asins)
            else:
                logger.error("Failed to get tokens for query. Skipping this parameter set.")
        
        # Find new ASINs
        new_asins = all_current_asins - prior_asins
        logger.info(f"Found {len(new_asins)} new ASINs out of {len(all_current_asins)} total")
        
        # Process new ASINs
        added_count = 0
        for asin in new_asins:
            product_info = keepa_manager.get_product_info(asin)
            if product_info:
                logger.info(f"Processing ASIN: {asin}")
                time.sleep(Config.PRODUCT_PROCESSING_DELAY)
                
                if airtable_manager.add_record(asin, product_info):
                    added_count += 1
            else:
                logger.warning(f"Could not retrieve product info for ASIN: {asin}")
        
        # Save updated ASIN list
        file_manager.write_set_to_file(Config.PRIOR_ASINS_FILE, all_current_asins)
        
        # Summary
        removed_count = len(prior_asins - all_current_asins)
        logger.info("=" * 60)
        logger.info(f"Process completed successfully")
        logger.info(f"  - New ASINs added to Airtable: {added_count}")
        logger.info(f"  - ASINs removed from tracking: {removed_count}")
        logger.info(f"  - Total ASINs tracked: {len(all_current_asins)}")
        logger.info("=" * 60)
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please check your environment variables and .env file")
    except Exception as e:
        logger.exception(f"Unexpected error in main: {e}")
        raise


if __name__ == "__main__":
    main()

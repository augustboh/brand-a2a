"""
Main application module for Brand A2A product tracking.

This module handles querying Keepa API for Amazon products and
adding new products to Airtable.
"""
import argparse
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
                
                # Must call update_status() to get actual token count
                # (tokens_left is 0 until first API call)
                self.api.update_status()
                
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
                # Request with offers=20 to get FBA fees data and history for price before drop
                product_data = self.api.query(asin, history=True, offers=20)
                
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
    
    def add_record(
        self, 
        asin: str, 
        product_info: Dict[str, Any],
        profitability: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a product record to Airtable.
        
        Args:
            asin: Amazon ASIN
            product_info: Product information dictionary from Keepa
            profitability: Optional profitability metrics dictionary
        
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
            
            # Build the fields dictionary
            fields = {
                "ASIN": asin,
                "Product Name": product_name,
                "Brand": brand,
                "Linked Brand": brand,
                "Category": product_category,
                "Date Added": current_date_time,
                "BILM": bilm,
                "Seller": who_is_seller
            }
            
            # Add profitability fields if available
            if profitability:
                if profitability.get('current_price') is not None:
                    fields["Current Price"] = profitability['current_price']
                if profitability.get('previous_price') is not None:
                    fields["Previous Price"] = profitability['previous_price']
                if profitability.get('fba_pick_and_pack_fee') is not None:
                    fields["FBA Pick Pack Fee"] = profitability['fba_pick_and_pack_fee']
                if profitability.get('referral_fee_percent') is not None:
                    fields["Referral Fee Percent"] = profitability['referral_fee_percent']
                if profitability.get('referral_fee_amount') is not None:
                    fields["Referral Fee Amount"] = profitability['referral_fee_amount']
                if profitability.get('net_revenue') is not None:
                    fields["Net Revenue"] = profitability['net_revenue']
                if profitability.get('estimated_profit') is not None:
                    fields["Estimated Profit"] = profitability['estimated_profit']
                if profitability.get('profit_margin_percent') is not None:
                    fields["Profit Margin Percent"] = profitability['profit_margin_percent']
            
            payload = {
                "records": [{
                    "fields": fields
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


class ProfitabilityCalculator:
    """Calculates profitability metrics from Keepa product data."""
    
    # Default referral fee percentage if not provided by Keepa (15% is common)
    DEFAULT_REFERRAL_FEE_PERCENT = 15.0
    
    # Categories with tiered referral fees (first $100 at higher rate, rest at lower rate)
    TIERED_FEE_CATEGORIES = {
        'Electronics': {'threshold': 100.0, 'below_rate': 15.0, 'above_rate': 8.0},
        'Computers': {'threshold': 100.0, 'below_rate': 15.0, 'above_rate': 8.0},
    }
    
    @staticmethod
    def get_product_category(product_info: Dict[str, Any]) -> Optional[str]:
        """Get the root category name for a product."""
        category_tree = product_info.get('categoryTree', [])
        if category_tree and len(category_tree) > 0:
            return category_tree[0].get('name', None)
        return None
    
    @staticmethod
    def calculate_referral_fee(sale_price: float, category: Optional[str], base_fee_percent: float) -> float:
        """
        Calculate referral fee with tiered pricing for certain categories.
        
        For Electronics: 15% on first $100, 8% on amount above $100
        
        Args:
            sale_price: The sale price in dollars
            category: The product category name
            base_fee_percent: The base referral fee percentage
        
        Returns:
            The referral fee amount in dollars
        """
        # Check if category has tiered fees
        if category and category in ProfitabilityCalculator.TIERED_FEE_CATEGORIES:
            tier = ProfitabilityCalculator.TIERED_FEE_CATEGORIES[category]
            threshold = tier['threshold']
            below_rate = tier['below_rate'] / 100
            above_rate = tier['above_rate'] / 100
            
            if sale_price <= threshold:
                return sale_price * below_rate
            else:
                # First $100 at 15%, rest at 8%
                fee_on_first = threshold * below_rate
                fee_on_rest = (sale_price - threshold) * above_rate
                return fee_on_first + fee_on_rest
        
        # Standard flat percentage
        return sale_price * (base_fee_percent / 100)
    
    @staticmethod
    def extract_fees(product_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract FBA fees and referral fee percentage from product data.
        
        Args:
            product_info: Product information dictionary from Keepa
        
        Returns:
            Dictionary containing:
                - fba_pick_and_pack_fee: FBA pick and pack fee in dollars
                - referral_fee_percent: Referral fee percentage (e.g., 15 for 15%)
        """
        fba_fees = product_info.get('fbaFees', {}) or {}
        
        # pickAndPackFee is in cents, convert to dollars
        pick_and_pack_fee_cents = fba_fees.get('pickAndPackFee', 0) or 0
        pick_and_pack_fee = pick_and_pack_fee_cents / 100
        
        # referralFeePercent is stored as percentage (e.g., 15 for 15%)
        # Default to 15% if not provided (common for most categories)
        referral_fee_percent = fba_fees.get('referralFeePercent')
        if referral_fee_percent is None:
            referral_fee_percent = ProfitabilityCalculator.DEFAULT_REFERRAL_FEE_PERCENT
        
        return {
            'fba_pick_and_pack_fee': pick_and_pack_fee,
            'referral_fee_percent': referral_fee_percent
        }
    
    @staticmethod
    def get_current_price(product_info: Dict[str, Any]) -> Optional[float]:
        """
        Get the current buy box price from product data.
        
        Args:
            product_info: Product information dictionary from Keepa
        
        Returns:
            Current price in dollars or None if not available
        """
        data = product_info.get('data', {})
        
        # Try BUY_BOX_SHIPPING first (includes shipping), then fall back to NEW
        price_history = data.get('BUY_BOX_SHIPPING')
        if price_history is None or (hasattr(price_history, '__len__') and len(price_history) == 0):
            price_history = data.get('NEW')
        
        if price_history is not None and hasattr(price_history, '__len__') and len(price_history) > 0:
            # Get the most recent price (last element)
            # Note: keepa library already converts prices to dollars
            current_price = price_history[-1]
            # Handle numpy types and check for valid price
            if current_price is not None:
                price_val = float(current_price)
                if price_val > 0:
                    return price_val
        
        return None
    
    @staticmethod
    def get_previous_price(product_info: Dict[str, Any], lookback_entries: int = 10) -> Optional[float]:
        """
        Get the price before the recent drop from price history.
        
        Looks back through the price history to find a higher price point
        that represents the price before the drop.
        
        Args:
            product_info: Product information dictionary from Keepa
            lookback_entries: Number of entries to look back in history
        
        Returns:
            Previous (higher) price in dollars or None if not available
        """
        data = product_info.get('data', {})
        
        # Try BUY_BOX_SHIPPING first, then fall back to NEW
        price_history = data.get('BUY_BOX_SHIPPING')
        if price_history is None or (hasattr(price_history, '__len__') and len(price_history) == 0):
            price_history = data.get('NEW')
        
        if price_history is None or not hasattr(price_history, '__len__') or len(price_history) < 2:
            return None
        
        # Note: keepa library already converts prices to dollars
        current_price = float(price_history[-1]) if price_history[-1] is not None else 0
        if current_price <= 0:
            return None
        
        # Look back through history to find a higher price (price before the drop)
        # Start from the second-to-last entry and go backwards
        for i in range(len(price_history) - 2, max(-1, len(price_history) - lookback_entries - 2), -1):
            price = price_history[i]
            if price is not None:
                price_val = float(price)
                if price_val > 0 and price_val > current_price:
                    return price_val
        
        return None
    
    @staticmethod
    def calculate_profitability(
        product_info: Dict[str, Any],
        purchase_price: Optional[float] = None,
        min_margin_percent: float = None
    ) -> Dict[str, Any]:
        """
        Calculate profitability metrics for a product.
        
        Formula: Profit = (sale_price * (1 - referral_fee%)) - FBA_pick_and_pack - purchase_price
        Margin = Profit / sale_price * 100
        
        Args:
            product_info: Product information dictionary from Keepa
            purchase_price: The price to purchase at (defaults to current price)
            min_margin_percent: Minimum margin % to consider profitable (default from Config)
        
        Returns:
            Dictionary containing:
                - current_price: Current price in dollars (purchase price)
                - previous_price: Price before the drop (simulated sell price) in dollars
                - fba_pick_and_pack_fee: FBA pick and pack fee in dollars
                - referral_fee_percent: Referral fee percentage
                - referral_fee_amount: Calculated referral fee in dollars
                - net_revenue: Revenue after Amazon fees (sale_price * (1-ref%) - FBA)
                - estimated_profit: Estimated profit in dollars
                - profit_margin_percent: Profit margin as percentage of sale price
                - meets_margin_threshold: Boolean indicating if margin >= min threshold
        """
        if min_margin_percent is None:
            min_margin_percent = Config.MIN_PROFIT_MARGIN_PERCENT
        
        fees = ProfitabilityCalculator.extract_fees(product_info)
        current_price = ProfitabilityCalculator.get_current_price(product_info)
        previous_price = ProfitabilityCalculator.get_previous_price(product_info)
        
        result = {
            'current_price': current_price,
            'previous_price': previous_price,
            'fba_pick_and_pack_fee': fees['fba_pick_and_pack_fee'],
            'referral_fee_percent': fees['referral_fee_percent'],
            'referral_fee_amount': None,
            'net_revenue': None,
            'estimated_profit': None,
            'profit_margin_percent': None,
            'meets_margin_threshold': False
        }
        
        # Use current price as purchase price if not specified
        if purchase_price is None:
            purchase_price = current_price
        
        # Can only calculate if we have both prices and fees
        if previous_price and purchase_price:
            sale_price = previous_price
            fba_fee = fees['fba_pick_and_pack_fee']
            
            # Get category for tiered fee calculation
            category = ProfitabilityCalculator.get_product_category(product_info)
            
            # Calculate referral fee amount (with tiered pricing for Electronics, etc.)
            referral_fee_amount = ProfitabilityCalculator.calculate_referral_fee(
                sale_price, category, fees['referral_fee_percent']
            )
            result['referral_fee_amount'] = round(referral_fee_amount, 2)
            
            # Net revenue = sale_price - referral_fee - FBA pick and pack
            net_revenue = sale_price - referral_fee_amount - fba_fee
            result['net_revenue'] = round(net_revenue, 2)
            
            # Profit = Net revenue - Purchase price
            estimated_profit = net_revenue - purchase_price
            result['estimated_profit'] = round(estimated_profit, 2)
            
            # Profit margin as percentage of sale price
            if sale_price > 0:
                profit_margin_percent = (estimated_profit / sale_price) * 100
                result['profit_margin_percent'] = round(profit_margin_percent, 2)
                
                # Check if meets minimum margin threshold (10% or better)
                result['meets_margin_threshold'] = profit_margin_percent >= min_margin_percent
            
            logger.info(
                f"Profitability: Buy=${purchase_price:.2f}, Sell=${sale_price:.2f}, "
                f"Net=${net_revenue:.2f}, Profit=${estimated_profit:.2f} "
                f"({result['profit_margin_percent']:.1f}%) - "
                f"{'✓ MEETS 10% THRESHOLD' if result['meets_margin_threshold'] else '✗ Below threshold'}"
            )
        else:
            logger.warning("Could not calculate profitability: missing price data")
        
        return result


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
        
        # Process new ASINs - only add those meeting 10% margin threshold
        added_count = 0
        profitable_count = 0
        skipped_count = 0
        for asin in new_asins:
            product_info = keepa_manager.get_product_info(asin)
            if product_info:
                logger.info(f"Processing ASIN: {asin}")
                
                # Calculate profitability metrics
                profitability = ProfitabilityCalculator.calculate_profitability(product_info)
                
                # Only add to Airtable if meets 10% margin threshold
                if profitability['meets_margin_threshold']:
                    profitable_count += 1
                    logger.info(
                        f"✓ ADDING TO AIRTABLE: {asin} - "
                        f"Profit: ${profitability['estimated_profit']:.2f} "
                        f"({profitability['profit_margin_percent']:.1f}% margin)"
                    )
                    
                    time.sleep(Config.PRODUCT_PROCESSING_DELAY)
                    
                    if airtable_manager.add_record(asin, product_info, profitability):
                        added_count += 1
                else:
                    skipped_count += 1
                    margin = profitability.get('profit_margin_percent')
                    if margin is not None:
                        logger.info(f"✗ Skipping {asin}: {margin:.1f}% margin (below 10% threshold)")
                    else:
                        logger.info(f"✗ Skipping {asin}: Could not calculate margin")
            else:
                logger.warning(f"Could not retrieve product info for ASIN: {asin}")
        
        # Save updated ASIN list
        file_manager.write_set_to_file(Config.PRIOR_ASINS_FILE, all_current_asins)
        
        # Summary
        removed_count = len(prior_asins - all_current_asins)
        logger.info("=" * 60)
        logger.info(f"Process completed successfully")
        logger.info(f"  - New ASINs found: {len(new_asins)}")
        logger.info(f"  - Profitable deals (≥10% margin): {profitable_count}")
        logger.info(f"  - Added to Airtable: {added_count}")
        logger.info(f"  - Skipped (below 10% margin): {skipped_count}")
        logger.info(f"  - ASINs removed from tracking: {removed_count}")
        logger.info(f"  - Total ASINs tracked: {len(all_current_asins)}")
        logger.info("=" * 60)
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please check your environment variables and .env file")
    except Exception as e:
        logger.exception(f"Unexpected error in main: {e}")
        raise


def test_asin(asin: str) -> None:
    """
    Test a specific ASIN to see profitability details and if it would be added.
    
    Args:
        asin: The Amazon ASIN to test
    """
    print("\n" + "=" * 70)
    print(f"  TESTING ASIN: {asin}")
    print("=" * 70 + "\n")
    
    try:
        # Validate configuration (only need Keepa keys for testing)
        if not Config.KEEPA_API_KEYS or not any(Config.KEEPA_API_KEYS):
            print("❌ ERROR: KEEPA_API_KEYS environment variable is required")
            return
        
        # Initialize Keepa API
        keepa_manager = KeepaAPIManager(Config.KEEPA_API_KEYS)
        if not keepa_manager.initialize_api():
            print("❌ ERROR: Failed to initialize Keepa API")
            return
        
        print(f"📡 Fetching product data from Keepa...")
        product_info = keepa_manager.get_product_info(asin)
        
        if not product_info:
            print(f"❌ ERROR: Could not retrieve product info for ASIN: {asin}")
            return
        
        # Display product info
        print("\n📦 PRODUCT INFORMATION:")
        print("-" * 50)
        print(f"  Title: {product_info.get('title', 'Unknown')}")
        print(f"  Brand: {product_info.get('brand', 'Unknown')}")
        category_tree = product_info.get('categoryTree', [])
        if category_tree:
            print(f"  Category: {category_tree[0].get('name', 'Unknown')}")
        print(f"  Monthly Sold: {product_info.get('monthlySold', 'N/A')}")
        seller = "Other" if product_info.get('availabilityAmazon') == -1 else "Amazon"
        print(f"  Seller: {seller}")
        
        # Calculate profitability
        print("\n💰 PROFITABILITY ANALYSIS:")
        print("-" * 50)
        profitability = ProfitabilityCalculator.calculate_profitability(product_info)
        
        # Display fee info
        print(f"  FBA Pick & Pack Fee: ${profitability['fba_pick_and_pack_fee']:.2f}")
        print(f"  Referral Fee %: {profitability['referral_fee_percent']}%")
        
        # Display prices
        current_price = profitability.get('current_price')
        previous_price = profitability.get('previous_price')
        
        if current_price is not None:
            print(f"\n  Current Price (Buy At): ${current_price:.2f}")
        else:
            print("\n  Current Price (Buy At): ❌ Not available")
        
        if previous_price is not None:
            print(f"  Previous Price (Sell At): ${previous_price:.2f}")
        else:
            print("  Previous Price (Sell At): ❌ Not available")
        
        # Display calculated values
        if profitability.get('referral_fee_amount') is not None:
            print(f"\n  Referral Fee Amount: ${profitability['referral_fee_amount']:.2f}")
        
        if profitability.get('net_revenue') is not None:
            print(f"  Net Revenue (after Amazon fees): ${profitability['net_revenue']:.2f}")
        
        if profitability.get('estimated_profit') is not None:
            print(f"  Estimated Profit: ${profitability['estimated_profit']:.2f}")
        
        if profitability.get('profit_margin_percent') is not None:
            print(f"  Profit Margin: {profitability['profit_margin_percent']:.2f}%")
        
        # Final verdict
        print("\n" + "=" * 50)
        min_margin = Config.MIN_PROFIT_MARGIN_PERCENT
        
        if profitability.get('meets_margin_threshold'):
            print(f"  ✅ RESULT: WOULD BE ADDED TO AIRTABLE")
            print(f"     Margin {profitability['profit_margin_percent']:.2f}% >= {min_margin}% threshold")
        else:
            margin = profitability.get('profit_margin_percent')
            if margin is not None:
                print(f"  ❌ RESULT: WOULD NOT BE ADDED")
                print(f"     Margin {margin:.2f}% < {min_margin}% threshold")
            else:
                print(f"  ❌ RESULT: WOULD NOT BE ADDED")
                print(f"     Could not calculate margin (missing price data)")
        
        print("=" * 50 + "\n")
        
        # Show the formula breakdown if we have the data
        if current_price and previous_price and profitability.get('net_revenue'):
            print("📊 CALCULATION BREAKDOWN:")
            print("-" * 50)
            ref_fee_amount = profitability['referral_fee_amount']
            fba_fee = profitability['fba_pick_and_pack_fee']
            
            # Check if tiered fees apply
            category_tree = product_info.get('categoryTree', [])
            category = category_tree[0].get('name', '') if category_tree else ''
            
            if category in ProfitabilityCalculator.TIERED_FEE_CATEGORIES and previous_price > 100:
                tier = ProfitabilityCalculator.TIERED_FEE_CATEGORIES[category]
                amount_above = previous_price - 100
                fee_on_first = 100 * (tier['below_rate'] / 100)
                fee_on_rest = amount_above * (tier['above_rate'] / 100)
                print(f"  Referral Fee (Tiered for {category}):")
                print(f"    = $100.00 × {tier['below_rate']}% + ${amount_above:.2f} × {tier['above_rate']}%")
                print(f"    = ${fee_on_first:.2f} + ${fee_on_rest:.2f}")
                print(f"    = ${ref_fee_amount:.2f}")
            else:
                ref_fee_pct = profitability['referral_fee_percent']
                print(f"  Referral Fee = ${previous_price:.2f} × {ref_fee_pct}%")
                print(f"              = ${ref_fee_amount:.2f}")
            
            print(f"\n  Net Revenue = Sale Price - Referral Fee - FBA Fee")
            print(f"             = ${previous_price:.2f} - ${ref_fee_amount:.2f} - ${fba_fee:.2f}")
            print(f"             = ${profitability['net_revenue']:.2f}")
            print(f"\n  Profit = Net Revenue - Purchase Price")
            print(f"         = ${profitability['net_revenue']:.2f} - ${current_price:.2f}")
            print(f"         = ${profitability['estimated_profit']:.2f}")
            print(f"\n  Margin = Profit / Sale Price × 100")
            print(f"         = ${profitability['estimated_profit']:.2f} / ${previous_price:.2f} × 100")
            print(f"         = {profitability['profit_margin_percent']:.2f}%")
            print()
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Brand A2A Product Tracking - Find profitable Amazon deals"
    )
    parser.add_argument(
        "--test",
        "-t",
        metavar="ASIN",
        help="Test a specific ASIN to see if it would be added to Airtable"
    )
    
    args = parser.parse_args()
    
    if args.test:
        test_asin(args.test)
    else:
        main()

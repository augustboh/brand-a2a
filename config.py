"""
Configuration management module for the Brand A2A application.

This module handles loading configuration from environment variables
and provides a centralized configuration object.
"""
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""
    
    # Keepa API Configuration
    KEEPA_API_KEYS: List[str] = [
        key.strip() 
        for key in os.getenv('KEEPA_API_KEYS', '').split(',') 
        if key.strip()
    ] if os.getenv('KEEPA_API_KEYS') else []
    
    # Airtable Configuration
    AIRTABLE_BASE_ID: str = os.getenv('AIRTABLE_BASE_ID', '')
    AIRTABLE_TABLE_ID: str = os.getenv('AIRTABLE_TABLE_ID', '')
    AIRTABLE_API_TOKEN: str = os.getenv('AIRTABLE_API_TOKEN', '')
    
    # File Configuration
    PRIOR_ASINS_FILE: str = os.getenv('PRIOR_ASINS_FILE', 'prior_asins.txt')
    
    # API Configuration
    KEEPA_DOMAIN: int = int(os.getenv('KEEPA_DOMAIN', '1'))
    MIN_TOKENS_REQUIRED: int = int(os.getenv('MIN_TOKENS_REQUIRED', '20'))
    MIN_TOKENS_FOR_QUERY: int = int(os.getenv('MIN_TOKENS_FOR_QUERY', '2'))
    TOKEN_WAIT_TIMEOUT: int = int(os.getenv('TOKEN_WAIT_TIMEOUT', '30'))
    
    # Retry Configuration
    MAX_RETRIES: int = int(os.getenv('MAX_RETRIES', '3'))
    RETRY_DELAY: int = int(os.getenv('RETRY_DELAY', '5'))
    
    # Rate Limiting
    API_REQUEST_DELAY: float = float(os.getenv('API_REQUEST_DELAY', '2.0'))
    PRODUCT_PROCESSING_DELAY: float = float(os.getenv('PRODUCT_PROCESSING_DELAY', '5.0'))
    
    # Profitability Thresholds
    MIN_PROFIT_MARGIN_PERCENT: float = float(os.getenv('MIN_PROFIT_MARGIN_PERCENT', '10.0'))
    
    @classmethod
    def validate(cls) -> None:
        """Validate that all required configuration is present."""
        errors = []
        
        if not cls.KEEPA_API_KEYS or not any(cls.KEEPA_API_KEYS):
            errors.append("KEEPA_API_KEYS environment variable is required")
        
        if not cls.AIRTABLE_BASE_ID:
            errors.append("AIRTABLE_BASE_ID environment variable is required")
        
        if not cls.AIRTABLE_TABLE_ID:
            errors.append("AIRTABLE_TABLE_ID environment variable is required")
        
        if not cls.AIRTABLE_API_TOKEN:
            errors.append("AIRTABLE_API_TOKEN environment variable is required")
        
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    @classmethod
    def get_query_params(cls) -> List[Dict[str, Any]]:
        """Get the query parameter sets for Keepa API."""
        return [
            {
                "monthlySold_gte": 50,
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
        ]


# A2A Product Tracking

Automated system for tracking Amazon products using Keepa API and managing them in Airtable.

## Features

- Queries Keepa API for Amazon products matching specific criteria
- Automatically adds new products to Airtable
- Manages multiple Keepa API keys with automatic rotation
- Tracks processed ASINs to avoid duplicates
- Comprehensive logging and error handling

## Prerequisites

- Python 3.9 or higher
- Keepa API keys (one or more)
- Airtable account with:
  - Base ID
  - Table ID
  - API token

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd brand-a2a-1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (see Configuration section below)

## Configuration

### Local Development

Create a `.env` file in the project root with the following variables:

```env
# Keepa API Configuration
# Provide comma-separated list of Keepa API keys
KEEPA_API_KEYS=key1,key2,key3,key4,key5,key6,key7

# Airtable Configuration
AIRTABLE_BASE_ID=your_base_id_here
AIRTABLE_TABLE_ID=your_table_id_here
AIRTABLE_API_TOKEN=your_airtable_token_here

# File Configuration (optional, defaults shown)
PRIOR_ASINS_FILE=prior_asins.txt

# API Configuration (optional, defaults shown)
KEEPA_DOMAIN=1
MIN_TOKENS_REQUIRED=20
MIN_TOKENS_FOR_QUERY=2
TOKEN_WAIT_TIMEOUT=30

# Retry Configuration (optional, defaults shown)
MAX_RETRIES=3
RETRY_DELAY=5

# Rate Limiting (optional, defaults shown)
API_REQUEST_DELAY=2.0
PRODUCT_PROCESSING_DELAY=5.0
```

**Important:** The `.env` file is ignored by git (see `.gitignore`). Never commit your API keys!

### GitHub Actions / CI/CD

For GitHub Actions workflows, you need to set up secrets in your GitHub repository:

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** and add the following secrets:

   - `KEEPA_API_KEYS`: Comma-separated list of your Keepa API keys
   - `AIRTABLE_BASE_ID`: Your Airtable base ID
   - `AIRTABLE_TABLE_ID`: Your Airtable table ID
   - `AIRTABLE_API_TOKEN`: Your Airtable API token

4. Update your GitHub Actions workflow file (`.github/workflows/actions.yml`) to use these secrets:

```yaml
- name: execute py script
  env:
    KEEPA_API_KEYS: ${{ secrets.KEEPA_API_KEYS }}
    AIRTABLE_BASE_ID: ${{ secrets.AIRTABLE_BASE_ID }}
    AIRTABLE_TABLE_ID: ${{ secrets.AIRTABLE_TABLE_ID }}
    AIRTABLE_API_TOKEN: ${{ secrets.AIRTABLE_API_TOKEN }}
  run: python main.py
```

## Usage

Run the script:

```bash
python main.py
```

The script will:
1. Initialize the Keepa API connection
2. Load previously processed ASINs from `prior_asins.txt`
3. Query Keepa API for products matching configured criteria
4. Identify new ASINs not in the prior list
5. Fetch product details for new ASINs
6. Add new products to Airtable
7. Update `prior_asins.txt` with all current ASINs

## Project Structure

```
brand-a2a-1/
├── main.py              # Main application entry point
├── config.py            # Configuration management
├── requirements.txt     # Python dependencies
├── prior_asins.txt      # Tracked ASINs (auto-generated)
├── .env                 # Environment variables (not in git)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Code Structure

The application is organized into several classes:

- **`Config`**: Manages configuration from environment variables
- **`KeepaAPIManager`**: Handles Keepa API connections and key rotation
- **`AirtableManager`**: Manages Airtable API interactions
- **`FileManager`**: Handles file I/O operations

## Logging

The application uses Python's `logging` module for comprehensive logging. Logs include:
- API initialization and key rotation
- Query results and product processing
- Success and error messages
- Summary statistics

## Error Handling

The application includes robust error handling:
- Automatic API key rotation when tokens are low
- Retry logic for failed API requests
- Timeout handling for long-running operations
- Graceful degradation when API keys are unavailable

## Best Practices Implemented

- ✅ Environment variables for sensitive data (API keys, tokens)
- ✅ Proper logging instead of print statements
- ✅ Type hints for better code clarity
- ✅ Docstrings for all functions and classes
- ✅ Class-based architecture instead of global variables
- ✅ Comprehensive error handling
- ✅ Configuration validation
- ✅ `.gitignore` to prevent committing sensitive files
- ✅ Modular code structure

## Troubleshooting

### "Configuration errors" on startup
- Ensure all required environment variables are set
- Check that your `.env` file exists and is properly formatted
- Verify API keys are comma-separated without spaces

### "Failed to initialize Keepa API"
- Check that your API keys are valid
- Ensure at least one API key has available tokens
- Verify your internet connection

### "Failed to add record to Airtable"
- Verify your Airtable credentials are correct
- Check that the table structure matches expected fields
- Ensure your API token has write permissions

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]


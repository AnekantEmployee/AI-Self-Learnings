from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
import requests
import os


@tool
def search_web(query: str) -> str:
    """Search the web for information using Google Serper API

    Args:
        query (str): The search query

    Returns:
        str: Search results
    """
    search = GoogleSerperAPIWrapper()
    return search.run(query)


@tool
def get_stock_price(symbol: str) -> str:
    """Fetch current stock price for a company symbol like TSLA, AAPL, etc.

    Args:
        symbol (str): Stock symbol (e.g., TSLA for Tesla, AAPL for Apple)

    Returns:
        str: Current stock price information
    """
    try:
        api_key = os.getenv("ALPHA_ADVANTAGE")
        if not api_key:
            return f"Error: Alpha Vantage API key not found. Please set ALPHA_ADVANTAGE environment variable."

        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Check for API errors
        if "Error Message" in data:
            return f"Error: {data['Error Message']}"

        if "Note" in data:
            return f"API Limit: {data['Note']}"

        # Extract quote data
        quote = data.get("Global Quote", {})
        if not quote:
            return f"No data found for symbol {symbol}"

        price = quote.get("05. price", "N/A")
        change = quote.get("09. change", "N/A")
        change_percent = quote.get("10. change percent", "N/A")

        return f"Stock: {symbol}\nPrice: ${price}\nChange: {change} ({change_percent})"

    except requests.exceptions.RequestException as e:
        return f"Network error fetching stock data: {str(e)}"
    except Exception as e:
        return f"Error fetching stock price: {str(e)}"
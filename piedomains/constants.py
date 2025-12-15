#!/usr/bin/env python3
"""
Constants and classification categories for piedomains package.

This module defines the core classification categories used by piedomains for
domain content classification, as well as filtering constants for text processing.

The categories are based on the Shallalist categorization system, a comprehensive
classification scheme originally developed for web filtering and content analysis.
These categories cover the major types of web content found across the internet.

Example:
    Accessing classification categories:
        >>> from piedomains.constants import classes, most_common_words
        >>> print(f"Available categories: {len(classes)}")
        >>> print(f"Example categories: {classes[:5]}")
        >>> print(f"Common words to filter: {most_common_words[:5]}")
"""

# Shallalist-based website categories for domain classification
classes: list[str] = [
    "adv",  # Advertising - commercial advertisements and banners
    "alcohol",  # Alcohol - sites promoting or selling alcoholic beverages
    "automobile",  # Automobile - car-related sites, dealers, automotive info
    "dating",  # Dating - dating services, personals, relationship sites
    "downloads",  # Downloads - software downloads, file sharing
    "drugs",  # Drugs - illegal drugs, drug paraphernalia
    "education",  # Education - schools, universities, educational content
    "finance",  # Finance - banks, financial services, investment
    "fortunetelling",  # Fortune telling - astrology, psychics, supernatural
    "forum",  # Forum - discussion boards, community forums
    "gamble",  # Gambling - casinos, betting, lottery sites
    "government",  # Government - official government sites and services
    "hobby",  # Hobby - recreational activities, hobbies, crafts
    "hospitals",  # Hospitals - medical facilities, health services
    "imagehosting",  # Image hosting - photo sharing, image storage services
    "isp",  # ISP - Internet service providers, telecom companies
    "jobsearch",  # Job search - employment, career sites, job boards
    "models",  # Models - fashion, modeling, photography
    "movies",  # Movies - film industry, movie reviews, entertainment
    "music",  # Music - music streaming, artists, music industry
    "news",  # News - news outlets, journalism, current events
    "politics",  # Politics - political parties, campaigns, political news
    "porn",  # Pornography - adult content (explicit material)
    "radiotv",  # Radio/TV - broadcasting, media companies
    "recreation",  # Recreation - sports, games, leisure activities
    "redirector",  # Redirector - URL redirects, link shorteners
    "religion",  # Religion - religious organizations, spiritual content
    "science",  # Science - scientific research, academic institutions
    "searchengines",  # Search engines - web search services, directories
    "sex",  # Sex - sexual content (non-pornographic), sexual health
    "shopping",  # Shopping - e-commerce, retail stores, online shopping
    "socialnet",  # Social networks - social media platforms, networking
    "spyware",  # Spyware - malicious software, security threats
    "tracker",  # Tracker - analytics, tracking services, monitoring
    "urlshortener",  # URL shortener - link shortening services
    "warez",  # Warez - pirated software, copyright violations
    "weapons",  # Weapons - firearms, military equipment, weapon sales
    "webmail",  # Webmail - email services, web-based email
    "webradio",  # Web radio - online radio stations, audio streaming
]
"""
List[str]: Complete list of website classification categories.

This list contains 41 categories used for domain content classification.
Categories are based on the Shallalist system and cover major website types
including commerce, media, government, adult content, and technology services.

The categories are used by both traditional ML models and LLM-based classification
to provide consistent categorization across different classification methods.
"""

# Common words to filter out during text processing
most_common_words: list[str] = [
    "home",  # Navigation element
    "contact",  # Contact information/page
    "us",  # Common in "contact us", "about us"
    "new",  # Generic content descriptor
    "news",  # Common content type
    "site",  # Website/site references
    "privacy",  # Privacy policy/settings
    "search",  # Search functionality
    "help",  # Help/support sections
    "copyright",  # Copyright notices
    "free",  # Marketing term
    "service",  # Service offerings
    "en",  # Language indicator (English)
    "get",  # Action verb (generic)
    "one",  # Generic article/number
    "find",  # Search/navigation verb
    "menu",  # Navigation element
    "account",  # User account references
    "next",  # Navigation control
]
"""
List[str]: Common words to filter out during text preprocessing.

These words are extremely common across all website types and provide little
discriminative value for classification. They are filtered out during text
processing to focus on more meaningful content words.

This list includes:
- Navigation elements (home, menu, next)
- Generic marketing terms (free, new, get)
- Common website sections (contact, help, privacy)
- Linguistic articles and connectors (us, one, en)

Used by text preprocessing functions to clean content before model input.
"""


# Category validation utilities
def get_valid_categories() -> list[str]:
    """
    Get a copy of all valid classification categories.

    Returns:
        List[str]: Complete list of valid category names for classification.

    Example:
        >>> categories = get_valid_categories()
        >>> if "news" in categories:
        ...     print("News category is available")
    """
    return classes.copy()


def is_valid_category(category: str) -> bool:
    """
    Check if a category name is valid for classification.

    Args:
        category (str): Category name to validate.

    Returns:
        bool: True if category is valid, False otherwise.

    Example:
        >>> is_valid_category("news")
        True
        >>> is_valid_category("invalid_category")
        False
    """
    return category in classes


def get_category_count() -> int:
    """
    Get the total number of available classification categories.

    Returns:
        int: Total number of classification categories.

    Example:
        >>> count = get_category_count()
        >>> print(f"Total categories available: {count}")
        Total categories available: 41
    """
    return len(classes)

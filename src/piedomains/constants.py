#!/usr/bin/env python3
"""Constants and classification categories for piedomains package.

This module defines the core classification categories used by piedomains for
domain content classification, as well as filtering constants for text processing.

The categories are based on the Shallalist categorization system, a comprehensive
classification scheme originally developed for web filtering and content analysis.
These categories cover the major types of web content found across the internet.

Example:
    Accessing classification categories:
        >>> from piedomains.constants import classes, most_common_words
        >>> print(f"Available categories: {len(classes)}")
        Available categories: 44
        >>> print(f"Example categories: {classes[:3]}")
        Example categories: ['adult', 'alcohol', 'automobile']
        >>> print(f"Common words to filter: {most_common_words[:3]}")
        Common words to filter: ['home', 'contact', 'us']
"""

# Website categories the text model emits.
#
# Derived from Shallalist, but not identical to it -- see training/taxonomy.py for the
# mapping and its reasoning. Three differences matter:
#
#   * `adv`, `tracker`, `spyware` and `redirector` are gone. They describe who runs a
#     site and how it is monetised, which a page does not state, and they accounted for
#     a third of evaluation errors (amazon -> adv, paypal -> spyware).
#   * `recreation` and `hobby` are split. `recreation` was 98% travel-or-sports and no
#     annotator could apply it consistently.
#   * `porn`, `sex` and `models` are merged into `adult`, which can be thresholded for
#     recall as a single safety signal.
#
# The loaded model's `labels.json` is authoritative at inference; this list documents
# the shipped set and is the last-resort fallback.
classes: list[str] = [
    "adult",  # Adult - pornography, sexual content and glamour. Merged from porn/sex/models
    "alcohol",  # Alcohol - breweries, wineries, distilleries
    "automobile",  # Automobile - cars, bikes, boats, planes
    "dating",  # Dating - dating and personals
    "downloads",  # Downloads - software and file downloads
    "drugs",  # Drugs - illegal drugs and paraphernalia
    "education",  # Education - schools, universities, educational content, sex education
    "finance",  # Finance - banking, insurance, lending, trading
    "fortunetelling",  # Fortune telling - astrology, psychics
    "forum",  # Forum - discussion boards
    "gamble",  # Gambling - casinos, betting, lottery
    "government",  # Government - official government sites
    "hobby/cooking",  # Hobby: cooking - split out from the 'hobby' grab-bag
    "hobby/games",  # Hobby: games. Merged: splitting by whether a game is played
    # online asks about delivery, not subject, and cost games-misc 34.4% of itself
    "hobby/gardening",  # Hobby: gardening - split out from the 'hobby' grab-bag
    "hobby/pets",  # Hobby: pets - split out from the 'hobby' grab-bag
    "homestyle",  # Homestyle - home and living
    "hospitals",  # Hospitals - medical facilities and health services
    "imagehosting",  # Image hosting - photo sharing and image storage
    "isp",  # ISP - internet service providers and telecoms
    "jobsearch",  # Job search - employment and career sites
    "library",  # Library - libraries and archives
    "military",  # Military - armed forces
    "movies",  # Movies - film and cinema
    "music",  # Music - artists, labels, music sites
    "news",  # News - newspapers, agencies, news portals
    "politics",  # Politics - parties, advocacy, political content
    "radiotv",  # Radio/TV - broadcast stations and streaming
    "realestate",  # Real estate - property listings and agents. Promoted out of finance
    "recreation/humor",  # Recreation: humor - split out; 'recreation' alone was 98% travel or sports
    "recreation/restaurants",  # Recreation: restaurants - split out; 'recreation' alone was 98% travel or sports
    "recreation/sports",  # Recreation: sports - split out; 'recreation' alone was 98% travel or sports
    "recreation/travel",  # Recreation: travel - split out; 'recreation' alone was 98% travel or sports
    "recreation/wellness",  # Recreation: wellness - split out; 'recreation' alone was 98% travel or sports
    "religion",  # Religion - religious groups and spirituality
    "science",  # Science - astronomy, chemistry, research
    "searchengines",  # Search engines - search and directories
    "shopping",  # Shopping - ecommerce, retail, marketplaces
    "socialnet",  # Social networks - social media platforms
    "urlshortener",  # URL shortener - link shortening services
    "weapons",  # Weapons - firearms and armaments
    "webmail",  # Webmail - browser-based email
    # A domain-parking placeholder rather than a site. Included because it is plainly
    # stated in the page text ("this domain is for sale"), unlike the hosting and
    # monetisation properties the taxonomy deliberately excludes -- and because leaving
    # it out was actively harmful: parked pages made up 42% of the `drugs` class, so the
    # model learned a for-sale template *meant* drugs and returned it for zappos.com.
    "parked",  # Domain parking placeholder, not a real site
    # The other way a domain serves bytes without being a site: a server autoindex, a
    # registrar's "coming soon", a suspended account, a bare 404. 321 of these sat in the
    # corpus wearing the label of whatever the domain used to be.
    "unavailable",  # Domain resolves, but there is no site behind it
]
"""
List[str]: Complete list of website classification categories.

This list contains 44 categories used for domain content classification.
Derived from Shallalist but not identical to it: classes describing how a site is
hosted or monetised are removed, so are those asking about delivery mechanism or
legality, `recreation` and `hobby` are split, and the adult categories are merged.
`parked` and `unavailable` are added, because a domain with no site behind it is a
fact the caller can act on and the page states it plainly. See training/taxonomy.py.

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
    """Get a copy of all valid classification categories.

    Returns:
        List[str]: Complete list of valid category names for classification.

    Example:
        >>> categories = get_valid_categories()
        >>> if "news" in categories:
        ...     print("News category is available")
        News category is available
    """
    return classes.copy()


def is_valid_category(category: str) -> bool:
    """Check if a category name is valid for classification.

    Args:
        category: Category name to validate.

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
    """Get the total number of available classification categories.

    Returns:
        int: Total number of classification categories.

    Example:
        >>> count = get_category_count()
        >>> print(f"Total categories available: {count}")
        Total categories available: 47
    """
    return len(classes)

#!/usr/bin/env python3
"""
twitter_content_generator.py - Generates social media content for WriteNow Agency

This script:
1. Selects a random image and uploads it to ImgBB
2. Scrapes trending topics from X (formerly Twitter)
3. Generates unique content using the Gemini API
4. Sends the content to a Google Spreadsheet via Apps Script
"""

import os
import random
import requests
import json
import time
import logging
import re
from datetime import datetime
from bs4 import BeautifulSoup
import google.generativeai as genai
from dotenv import load_dotenv

# Import the image processor
from image_processor import get_random_image, move_used_image

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="writenow_twitter.log",
)
logger = logging.getLogger(__name__)

# Content types with their weights for random selection
CONTENT_TYPES = {
    "educational": 0.3,
    "promotional": 0.3,
    "engaging": 0.3,
    "marketing": 0.1,
}

# Content generation modes with weights - determine if we use trends or not
CONTENT_MODES = {
    "with_trends": 0.5,  # 50% chance to use trends directly in content
    "append_hashtags": 0.5,  # 50% chance to just append trending hashtags at the end
}

# Tweet character limit
CHARACTER_LIMIT = 280

# Replacement strings
REPLACEMENTS = {
    "[link]": "https://writenowagency.com",
    "[email]": "info@writenowagency.com",
    "[phone]": "+27 123 456 789",
    "[consultation]": "https://writenowagency.com/free-consultation",
}

# Google Apps Script Web App URL
WEBAPP_URL = os.getenv("WEBAPP_URL")

# Secret API key - Must match the one in your Google Apps Script
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Company information
COMPANY_INFO = {
    "name": "WriteNow Agency PTY LTD",
    "industry": "Technology and Creative Solutions",
    "services": "Web Development, SaaS, AI, Software Solutions, Business Process Automation",
    "mission": "Transforming ideas into reality with innovative digital solutions that drive business growth and operational efficiency.",
    "locations": "Beverly Hills, California and Morningside, Sandton",
    "specialties": "Custom websites, SaaS, AI-driven solutions, custom software, business automation",
    "webDevelopment": "Corporate websites, e-commerce stores, landing pages & funnels, web applications, SEO-optimized CMS",
    "saasAi": "Custom SaaS development, AI-powered chatbots & virtual assistants, AI-driven content generation, machine learning & predictive analytics",
    "software": "Bespoke enterprise software, API development & integration, mobile & desktop applications, automated workflows & business tools",
    "automation": "Workflow automation, robotic process automation (RPA), AI-powered data processing, CRM & ERP integration, automated reporting & analytics",
    "pricing": {
        "webDevelopment": "ZAR 1,500 – ZAR 40,000+",
        "saasAi": "ZAR 50,000 – ZAR 150,000+",
        "software": "ZAR 15,000 – ZAR 40,000+",
        "automation": "Custom pricing based on business needs",
    },
    "bundles": {
        "webAiStarter": "45% OFF",
        "saasDevelopment": "40% OFF",
        "enterpriseAi": "50% OFF",
    },
    "contact": {"email": "info@writenowagency.com", "website": "writenowagency.com"},
}


def setup_gemini():
    """Setup Gemini API with safety settings and error handling"""
    try:
        print(
            f"Initializing Gemini with API key starting with: {GEMINI_API_KEY[:5]}..."
        )
        genai.configure(api_key=GEMINI_API_KEY)

        # Define safety settings that are more permissive for marketing content
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
        ]

        # Create the model with safety settings
        model = genai.GenerativeModel(
            "gemini-2.0-flash", safety_settings=safety_settings
        )

        print("Gemini model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Error setting up Gemini: {e}")
        print(f"Failed to initialize Gemini: {str(e)}")
        print("Check your API key and internet connection")
        return None


def scrape_trending_hashtags():
    """Scrape trending hashtags with improved logic similar to the original script"""
    try:
        # URLs to scrape with different probabilities (higher for South Africa)
        urls = [
            "https://xtrends.iamrohit.in/",
            "https://xtrends.iamrohit.in/south-africa/johannesburg",
            "https://xtrends.iamrohit.in/united-states",
            "https://xtrends.iamrohit.in/united-arab-emirates",
            "https://xtrends.iamrohit.in/united-kingdom",
            "https://xtrends.iamrohit.in/japan",
            "https://xtrends.iamrohit.in/indonesia",
            "https://xtrends.iamrohit.in/india",
            "https://xtrends.iamrohit.in/germany",
            "https://xtrends.iamrohit.in/turkey",
            "https://xtrends.iamrohit.in/brazil",
        ]

        # Higher weight for South Africa (0.9)
        probabilities = [0.3, 0.9, 0.4, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

        # Select URL based on probabilities
        selected_url = random.choices(urls, probabilities)[0]
        print(f"Selected trend source: {selected_url}")
        logger.info(f"Selected trend source: {selected_url}")

        # Scrape the selected URL
        response = requests.get(selected_url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract trends from the page
            start_keyword = "X Trends - "
            end_keyword = "Note: Trends refreshes every 30 minutes."

            main_content = soup.get_text()
            start_index = main_content.find(start_keyword)
            end_index = main_content.find(end_keyword, start_index + len(start_keyword))

            if start_index != -1 and end_index != -1:
                main_content = main_content[
                    start_index + len(start_keyword) : end_index
                ]
                topics = main_content.split("|")

                # Clean up the topics
                clean_topics = [topic.strip() for topic in topics if topic.strip()]

                # Use Gemini to select the top trending topics
                if clean_topics:
                    topics_string = " | ".join(clean_topics)

                    model = setup_gemini()
                    response = model.generate_content(
                        f"From the following list of trending topics: {topics_string}\n\n"
                        f"Give me only 3 of the most interesting topics as a comma-separated list. "
                        f"Return ONLY the list with no additional text."
                    )

                    top_trends = response.text.strip()
                    print(f"Selected top trends: {top_trends}")
                    logger.info(f"Selected top trends: {top_trends}")

                    # Convert to list of tuples with location
                    location = (
                        selected_url.split("/")[-1] if "/" in selected_url else "global"
                    )
                    if location == "johannesburg":
                        location = "south africa"

                    trend_tuples = [
                        (trend.strip(), location) for trend in top_trends.split(",")
                    ]
                    return trend_tuples

            # Fallback: extract trends directly from trend list
            trend_elements = soup.select(".trend-list li a")
            trends = [
                (trend.text.strip(), "global")
                for trend in trend_elements
                if trend.text.strip()
            ]

            if trends:
                # Take a random sample of 3 trends
                selected_trends = random.sample(trends, min(3, len(trends)))
                print(f"Fallback selected trends: {selected_trends}")
                logger.info(f"Fallback selected trends: {selected_trends}")
                return selected_trends

        # If all else fails, return empty list
        return []

    except Exception as e:
        logger.error(f"Error scraping trending hashtags: {e}")
        print(f"Error scraping trending hashtags: {e}")
        return []


def select_content_type():
    """Select a random content type based on weights"""
    types, weights = zip(*CONTENT_TYPES.items())
    content_type = random.choices(types, weights=weights, k=1)[0]
    logger.info(f"Selected content type: {content_type}")
    return content_type


def select_content_mode():
    """Select if we should use trends in content or just append hashtags"""
    modes, weights = zip(*CONTENT_MODES.items())
    mode = random.choices(modes, weights=weights, k=1)[0]
    logger.info(f"Selected content mode: {mode}")
    return mode


def generate_unique_tweet(
    model, trends, content_type, content_mode, previous_tweets=None
):
    """Generate a unique tweet based on the selected content type and content mode"""
    if previous_tweets is None:
        previous_tweets = []

    # Different prompt behavior based on content mode
    if content_mode == "with_trends" and trends:
        # Format trends for using within the content
        trend_text = ", ".join([f"{trend}" for trend, _ in trends])
        use_trends_prompt = f"""
        Try to relate to or incorporate these trending topics naturally in the content: {trend_text}
        """
    else:
        # Don't mention trends in the prompt - we'll append hashtags later
        use_trends_prompt = "Create an original, engaging post without referencing any specific trending topics."

    # Different prompts based on content type
    prompts = {
        "educational": f"""
            Create an educational social media post for WriteNow Agency that teaches followers something valuable about technology, digital solutions, or business efficiency. 
            The post should position WriteNow as an expert in their field.
            
            {use_trends_prompt}
            
            Requirements:
            - Be informative and provide genuinely useful knowledge
            - Include a brief tip or insight that demonstrates expertise
            - Position WriteNow Agency as a thought leader
            - Sound like it's written by a professional social media marketer
            - Include a link to the website using "https://writenowagency.com" not [link]
            - Do not include and hashtags in the post
            - Keep under {CHARACTER_LIMIT-30} characters (to leave room for hashtags)
            - End with a clear call to action
        """,
        "promotional": f"""
            Create a promotional social media post for WriteNow Agency that highlights one of their specific services or packages.
            
            {use_trends_prompt}
            
            Service details to highlight (choose one):
            - Web Development: {COMPANY_INFO['webDevelopment']} (Price: {COMPANY_INFO['pricing']['webDevelopment']})
            - SaaS & AI Solutions: {COMPANY_INFO['saasAi']} (Price: {COMPANY_INFO['pricing']['saasAi']})
            - Custom Software: {COMPANY_INFO['software']} (Price: {COMPANY_INFO['pricing']['software']})
            - Business Process Automation: {COMPANY_INFO['automation']} (Price: {COMPANY_INFO['pricing']['automation']})
            - Bundle Discounts: Web+AI Starter ({COMPANY_INFO['bundles']['webAiStarter']}), SaaS Development ({COMPANY_INFO['bundles']['saasDevelopment']}), or Enterprise AI ({COMPANY_INFO['bundles']['enterpriseAi']})
            
            Requirements:
            - Include a clear value proposition
            - Mention a specific benefit the service provides
            - Include a call to action with the actual URL (https://writenowagency.com), not [link]
            - Sound like it's written by a professional social media marketer
            - Do not include and hashtags in the post
            - Keep under {CHARACTER_LIMIT-30} characters (to leave room for hashtags)
        """,
        "engaging": f"""
            Create an engaging social media post for WriteNow Agency that starts a conversation with followers about technology, innovation, or business challenges.
            
            {use_trends_prompt}
            
            Requirements:
            - Ask a thought-provoking question that encourages replies
            - Touch on a pain point that WriteNow's services could solve
            - Keep the tone conversational and approachable
            - Sound like it's written by a professional social media marketer
            - Do not include and hashtags in the post
            - Include the actual website URL (https://writenowagency.com), not [link]
            - Keep under {CHARACTER_LIMIT-30} characters (to leave room for hashtags)
        """,
        "marketing": f"""
            Create a marketing-focused social media post for WriteNow Agency that showcases the company's brand values and unique selling points.
            
            {use_trends_prompt}
            
            Requirements:
            - Highlight what makes WriteNow unique in the tech industry
            - Emphasize their global reach (Beverly Hills and Sandton offices)
            - Mention one unique selling point like "12 Months of Free Hosting" or "End-to-End Digital Solutions"
            - Include their actual website URL: https://writenowagency.com (not [link])
            - Do not include and hashtags in the post
            - Sound like it's written by a professional social media marketer
            - Keep under {CHARACTER_LIMIT-30} characters (to leave room for hashtags)
        """,
    }

    prompt = prompts[content_type]

    # Add company information to the prompt
    company_info_text = f"""
    Company information:
    Name: {COMPANY_INFO['name']}
    Industry: {COMPANY_INFO['industry']}
    Services: {COMPANY_INFO['services']}
    Mission: {COMPANY_INFO['mission']}
    Locations: {COMPANY_INFO['locations']}
    """

    # Add previous tweets to ensure uniqueness
    previous_tweets_text = ""
    if previous_tweets:
        previous_tweets_text = f"""
        IMPORTANT: Your post must be completely different from these previous posts:
        {' '.join(previous_tweets[-10:])}
        
        Ensure original wording, structure, and approach to avoid repetition.
        """

    # Final prompt
    final_prompt = f"{prompt}\n\n{company_info_text}\n\n{previous_tweets_text}\n\nReturn ONLY the post text with no explanations, introductions or quotation marks."

    try:
        response = model.generate_content(final_prompt)
        tweet = response.text.strip()

        # Verify character limit (accounting for hashtags to be added later)
        if len(tweet) > (CHARACTER_LIMIT - 30):
            tweet = tweet[: CHARACTER_LIMIT - 30]

        # Check if it's too similar to previous tweets
        if previous_tweets and any(
            calculate_similarity(tweet, prev) > 0.7 for prev in previous_tweets
        ):
            logger.warning(
                "Generated tweet too similar to previous ones. Trying again."
            )
            return generate_unique_tweet(
                model, trends, content_type, content_mode, previous_tweets
            )

        logger.info(f"Generated tweet: {tweet}")
        return tweet
    except Exception as e:
        logger.error(f"Error generating tweet content: {e}")
        return f"WriteNow Agency offers innovative digital solutions - from web development to AI. Visit https://writenowagency.com to transform your business."


def calculate_similarity(text1, text2):
    """Simple similarity calculation between two texts"""
    # Convert to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))

    if union == 0:
        return 0
    return intersection / union


def create_hashtags_from_trends(trends, count=3):
    """Convert trends to hashtags"""
    if not trends:
        # Default hashtags if no trends available
        return ["#TechSolutions", "#Innovation", "#DigitalTransformation"]

    # Extract just the trend names (not locations)
    trend_names = [trend for trend, _ in trends]

    # If we don't have enough trends, add some default ones
    while len(trend_names) < count:
        default_tags = [
            "#TechSolutions",
            "#Innovation",
            "#DigitalTransformation",
            "#AI",
            "#WebDev",
            "#BusinessGrowth",
            "#SaaS",
        ]
        for tag in default_tags:
            if tag not in trend_names:
                trend_names.append(tag)
                if len(trend_names) >= count:
                    break

    # Convert to proper hashtags
    hashtags = []
    for trend in trend_names[:count]:
        # Remove any existing # symbol and spaces
        clean_trend = trend.replace("#", "").replace(" ", "")
        # Skip if empty after cleaning
        if not clean_trend:
            continue
        # Add # symbol
        hashtag = f"#{clean_trend}"
        hashtags.append(hashtag)

    return hashtags[:count]  # Ensure we only return the requested count


def append_hashtags_to_tweet(tweet, hashtags):
    """Append hashtags to the tweet, ensuring it doesn't exceed character limit"""
    if not hashtags:
        return tweet

    # Join hashtags with spaces
    hashtag_text = " " + " ".join(hashtags)

    # Check if adding hashtags would exceed character limit
    if len(tweet) + len(hashtag_text) > CHARACTER_LIMIT:
        # Try removing hashtags one by one until it fits
        while hashtags and (len(tweet) + len(hashtag_text) > CHARACTER_LIMIT):
            hashtags.pop()
            hashtag_text = " " + " ".join(hashtags)

        # If we still can't fit any hashtags, truncate the tweet
        if len(tweet) + len(hashtag_text) > CHARACTER_LIMIT:
            return tweet[: CHARACTER_LIMIT - len(hashtag_text)] + hashtag_text

    return tweet + hashtag_text


def replace_placeholders(text):
    """Replace placeholder strings like [link] with actual values"""
    result = text
    for placeholder, value in REPLACEMENTS.items():
        result = result.replace(placeholder, value)
    return result


def load_previous_tweets(filename="previous_tweets.txt"):
    """Load previous tweets from a file"""
    try:
        if os.path.exists(filename):
            with open(filename, "r") as file:
                tweets = [line.strip() for line in file.readlines()]
            return tweets
        return []
    except Exception as e:
        logger.error(f"Error loading previous tweets: {e}")
        return []


def save_tweet(tweet, filename="previous_tweets.txt"):
    """Save a tweet to the previous tweets file"""
    try:
        with open(filename, "a") as file:
            file.write(tweet + "\n")
    except Exception as e:
        logger.error(f"Error saving tweet: {e}")


def send_to_spreadsheet(
    text, image_path, image_url, image_caption, content_type, trends, content_mode
):
    """Send generated content to Google Sheets via Apps Script with enhanced debugging"""
    try:
        # Format the trends for the spreadsheet
        trend_text = (
            ", ".join([f"{trend} (trending in {country})" for trend, country in trends])
            if trends
            else "No trends available"
        )

        # Get relative image path for spreadsheet - this is just for reference
        image_filename = os.path.basename(image_path) if image_path else "No image"

        # Prepare payload for the Google Apps Script
        # The key change here is using 'imagePath' field for the full URL, not the filename
        payload = {
            "action": "addTwitterContent",
            "apiKey": API_SECRET_KEY,
            "content": {
                "text": text,
                "imagePath": image_url
                or "No image",  # Use the full URL here instead of filename
                "imageUrl": image_url or "",  # Keep this as a backup
                "imageCaption": image_caption or "",
                "contentType": content_type,
                "contentMode": content_mode,
                "trends": trend_text,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "Ready to Post",
            },
        }

        # Enhanced debugging
        print(f"Sending data to: {WEBAPP_URL}")
        print(f"API Key (first 5 chars): {API_SECRET_KEY[:5]}...")
        print(f"Image URL being sent as imagePath: {image_url}")
        print(f"Image Caption: {image_caption}")
        print(f"Payload sample: {json.dumps(payload)[:100]}...")

        headers = {"Content-Type": "application/json"}

        # Send the request to the Google Apps Script Web App
        response = requests.post(WEBAPP_URL, json=payload, headers=headers, timeout=30)

        # More debugging
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text[:200]}...")

        if response.status_code == 200:
            try:
                result = response.json()
                if result.get("status") == "success":
                    logger.info("Successfully sent Twitter content to spreadsheet")
                    return True, result.get("spreadsheetUrl", "")
                else:
                    logger.error(f"Error from Apps Script: {result.get('message')}")
                    print(f"Error message from Apps Script: {result.get('message')}")
                    return False, ""
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {response.text}")
                print(f"Failed to parse JSON response: {response.text}")
                return False, ""
        else:
            logger.error(f"HTTP Error: {response.status_code}, {response.text}")
            print(f"HTTP Error: {response.status_code}")
            return False, ""

    except requests.exceptions.Timeout:
        logger.error("Request to Google Apps Script timed out")
        print("Request to Google Apps Script timed out - check your network connection")
        return False, ""
    except requests.exceptions.ConnectionError:
        logger.error("Connection error when contacting Google Apps Script")
        print("Connection error - check your network and the Apps Script URL")
        return False, ""
    except Exception as e:
        logger.error(f"Error sending content to spreadsheet: {e}")
        print(f"Unexpected error: {str(e)}")
        return False, ""


def validate_config():
    """Validate the configuration values with enhanced feedback"""
    is_valid = True

    if not WEBAPP_URL:
        logger.error("WEBAPP_URL environment variable not set")
        print("ERROR: WEBAPP_URL environment variable not set in .env file")
        is_valid = False
    else:
        print(f"WEBAPP_URL: {WEBAPP_URL[:30]}...")

        # Check for common URL formatting issues
        if not WEBAPP_URL.startswith("https://script.google.com/macros/s/"):
            print(
                "WARNING: WEBAPP_URL doesn't match the expected format for Google Apps Script"
            )

        if "/exec" not in WEBAPP_URL:
            print("WARNING: WEBAPP_URL should end with '/exec'")

    if not API_SECRET_KEY:
        logger.error("API_SECRET_KEY environment variable not set")
        print("ERROR: API_SECRET_KEY environment variable not set in .env file")
        is_valid = False
    else:
        print(f"API_SECRET_KEY: {API_SECRET_KEY[:5]}...")

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY environment variable not set")
        print("ERROR: GEMINI_API_KEY environment variable not set in .env file")
        is_valid = False
    else:
        print(f"GEMINI_API_KEY: {GEMINI_API_KEY[:5]}...")

    return is_valid


def main():
    """Main function to generate content and upload to Google Sheets"""
    try:
        # Validate configuration
        if not validate_config():
            return

        print("=== Twitter Content Generation for WriteNow Agency ===")

        # Setup Gemini
        print("Setting up Gemini AI...")
        model = setup_gemini()
        if not model:
            print("Failed to initialize Gemini model. Exiting.")
            return

        # Load previous tweets to ensure uniqueness
        previous_tweets = load_previous_tweets()
        print(f"Loaded {len(previous_tweets)} previous tweets for comparison")

        # Get a random image, URL and caption from the image processor
        print("Selecting and uploading random image...")
        image_path, image_url, image_caption = get_random_image()

        if image_path:
            print(f"Selected image: {os.path.basename(image_path)}")
            print(f"Image URL: {image_url}")
            print(f"Image Caption: {image_caption}")
        else:
            print("No image selected. Continuing without an image.")

        # Scrape trending hashtags
        print("Scraping trending topics...")
        trends = scrape_trending_hashtags()

        if trends:
            print(f"Selected {len(trends)} trends for content generation:")
            for trend, location in trends:
                print(f"  - {trend} (trending in {location})")
        else:
            print("No trends found. Will use default hashtags.")

        # Select content type and mode
        content_type = select_content_type()
        content_mode = select_content_mode()
        print(f"Content type: {content_type}")
        print(f"Content mode: {content_mode}")

        # Generate base tweet
        print("Generating tweet content...")
        tweet_content = generate_unique_tweet(
            model, trends, content_type, content_mode, previous_tweets
        )
        print(f"\nBase tweet ({len(tweet_content)} chars):\n{tweet_content}")

        # Replace any placeholder strings
        tweet_content = replace_placeholders(tweet_content)

        # Create or append hashtags based on content mode
        if content_mode == "append_hashtags":
            # Create hashtags from trends
            hashtags = create_hashtags_from_trends(trends)
            print(f"Appending hashtags: {hashtags}")

            # Append hashtags to tweet
            final_tweet = append_hashtags_to_tweet(tweet_content, hashtags)
        else:
            # For "with_trends" mode, the trends should already be incorporated
            # But let's make sure we have at least one hashtag
            if not re.search(r"#\w+", tweet_content):
                hashtags = create_hashtags_from_trends(trends, count=1)
                final_tweet = append_hashtags_to_tweet(tweet_content, hashtags)
            else:
                final_tweet = tweet_content

        print(f"\nFinal tweet ({len(final_tweet)} chars):\n{final_tweet}")

        # Send to spreadsheet with image URL and caption
        print(f"Sending content to spreadsheet...")
        success, url = send_to_spreadsheet(
            final_tweet,
            image_path,
            image_url,
            image_caption,
            content_type,
            trends,
            content_mode,
        )

        if success:
            print("✓ Content added successfully to Google Sheets")
            if url:
                print(f"View your content at: {url}")

            # Save tweet to previous tweets
            save_tweet(final_tweet)
            print("✓ Saved tweet to previous tweets record")

            # Move used image to archive
            if image_path and move_used_image(image_path):
                print(f"✓ Moved used image to archive")
            elif image_path:
                print("✗ Failed to move image to archive")
        else:
            print("✗ Failed to add content to Google Sheets")

        print("\n=== Twitter Content Generation Complete ===")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()  # Print full traceback for better debugging


if __name__ == "__main__":
    main()

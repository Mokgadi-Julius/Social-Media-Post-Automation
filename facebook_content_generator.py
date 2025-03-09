#!/usr/bin/env python3
"""
facebook_content_generator.py - Generates engaging content for WriteNow Agency's Facebook page

This script:
1. Selects a random image and uploads it to ImgBB
2. Generates engaging, SEO-optimized Facebook content using the Gemini API
3. Sends the content to a Google Spreadsheet via Apps Script
"""

import os
import random
import requests
import json
import time
import logging
import re
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv

# Import the image processor
from facebook_image_processor import (
    get_random_image,
    move_used_image,
)  # Use a separate image processor

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="writenow_facebook.log",
)
logger = logging.getLogger(__name__)

# Content types with their weights for random selection
CONTENT_TYPES = {
    "engagement_post": 0.3,
    "educational": 0.25,
    "promotional": 0.25,
    "behind_the_scenes": 0.2,
}

# Facebook character limit (higher than Twitter, somewhat lower than LinkedIn)
CHARACTER_LIMIT = 2000

# Replacement strings
REPLACEMENTS = {
    "[link]": "https://writenowagency.com",
    "[email]": "info@writenowagency.com",
    "[phone]": "+27 11 083 9898",
    "[consultation]": "https://www.writenowagency.com/contact.php",
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
    "facebookPage": "https://facebook.com/WriteNowAgency",  # Fictional FB page
}

# Facebook SEO keywords - these will be incorporated subtly into the content
SEO_KEYWORDS = [
    "tech solutions",
    "website development",
    "business automation",
    "custom software",
    "AI tools for business",
    "digital marketing",
    "business growth strategies",
    "web apps",
    "mobile apps",
    "SaaS solutions",
    "tech innovation",
    "small business technology",
    "digital transformation",
    "business efficiency",
    "website design",
]

# Facebook engagement prompts - for increasing interaction
ENGAGEMENT_PROMPTS = [
    "What tech challenges is your business facing right now?",
    "Have you tried automating any of your business processes yet?",
    "What's one tech tool you can't live without?",
    "How has your business website performed this year?",
    "Tag a business owner who needs to see this!",
    "Share this with someone who's starting their digital journey!",
    "What feature would you add to your business website if money wasn't an issue?",
    "Drop an emoji that describes your feelings about technology! 👍 👎 🤔 😍 😱",
    "Tell us about a tech problem you solved recently!",
    "What's your biggest time-waster in your business operations?",
]


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


def select_content_type():
    """Select a random content type based on weights"""
    types, weights = zip(*CONTENT_TYPES.items())
    content_type = random.choices(types, weights=weights, k=1)[0]
    logger.info(f"Selected content type: {content_type}")
    return content_type


def select_random_keywords(count=3):
    """Select random SEO keywords to incorporate into the content"""
    return random.sample(SEO_KEYWORDS, min(count, len(SEO_KEYWORDS)))


def select_random_engagement_prompt():
    """Select a random engagement prompt to add to the post"""
    return random.choice(ENGAGEMENT_PROMPTS)


def generate_facebook_content(model, content_type, image_caption, previous_posts=None):
    """Generate Facebook content based on the selected content type"""
    if previous_posts is None:
        previous_posts = []

    # Select keywords to incorporate
    keywords = select_random_keywords()
    keywords_text = ", ".join(keywords)

    # Select an engagement prompt
    engagement_prompt = select_random_engagement_prompt()

    # Different prompts based on content type
    prompts = {
        "engagement_post": f"""
            Create an engaging Facebook post for WriteNow Agency that encourages followers to interact.
            
            Use the following image caption as inspiration if relevant: "{image_caption}"
            
            Requirements:
            - Start with an attention-grabbing first line
            - Focus on a relatable business or technology topic
            - Ask questions or create a scenario that encourages comments
            - Keep it conversational, approachable, and warm
            - Subtly incorporate these SEO keywords: {keywords_text}
            - Include this engagement prompt at the end: "{engagement_prompt}"
            - Include the actual website URL (https://writenowagency.com) not [link]
            - Use emojis naturally and strategically to increase engagement
            - Optimize for Facebook algorithm by encouraging meaningful interactions
            - Keep under {CHARACTER_LIMIT} characters
            - Format with appropriate line breaks for readability
            - DO NOT use any markdown formatting (no **, *, or other formatting) as this doesn't work on Facebook
        """,
        "educational": f"""
            Create an educational Facebook post for WriteNow Agency that teaches followers about technology, digital solutions, or business efficiency.
            
            Use the following image caption as inspiration if relevant: "{image_caption}"
            
            Requirements:
            - Start with a "Did you know" or interesting fact that hooks the reader
            - Share valuable knowledge, tips, or step-by-step instructions
            - Include specific examples that businesses can implement
            - Subtly incorporate these SEO keywords: {keywords_text}
            - Explain a complex concept in simple, easy-to-understand terms
            - End with a question to encourage comments
            - Include the actual website URL (https://writenowagency.com) not [link]
            - Add 2-3 relevant emojis to make the post more engaging
            - Optimize for Facebook algorithm by being educational and valuable
            - Keep under {CHARACTER_LIMIT} characters
            - Format with appropriate line breaks for readability
            - DO NOT use any markdown formatting (no **, *, or other formatting) as this doesn't work on Facebook
        """,
        "promotional": f"""
            Create a promotional Facebook post for WriteNow Agency that showcases one of their specific services or offers.
            
            Use the following image caption as inspiration if relevant: "{image_caption}"
            
            Choose one service to highlight from:
            - Web Development: {COMPANY_INFO['webDevelopment']} (Price: {COMPANY_INFO['pricing']['webDevelopment']})
            - SaaS & AI Solutions: {COMPANY_INFO['saasAi']} (Price: {COMPANY_INFO['pricing']['saasAi']})
            - Custom Software: {COMPANY_INFO['software']} (Price: {COMPANY_INFO['pricing']['software']})
            - Business Process Automation: {COMPANY_INFO['automation']} (Price: {COMPANY_INFO['pricing']['automation']})
            - Bundle Discounts: Web+AI Starter ({COMPANY_INFO['bundles']['webAiStarter']}), SaaS Development ({COMPANY_INFO['bundles']['saasDevelopment']}), or Enterprise AI ({COMPANY_INFO['bundles']['enterpriseAi']})
            
            Requirements:
            - Start with a pain point or challenge that resonates with business owners
            - Present the selected service as the solution to this challenge
            - Describe benefits rather than just features
            - Subtly incorporate these SEO keywords: {keywords_text}
            - Include specific details (pricing, features, or special offers)
            - Use social proof (mention satisfied clients or positive outcomes)
            - End with a clear call-to-action and the website link
            - Include the actual website URL (https://writenowagency.com) not [link]
            - Add a few emojis strategically to highlight key points
            - Optimize for Facebook algorithm by being authentic and solution-focused
            - Keep under {CHARACTER_LIMIT} characters
            - Format with appropriate line breaks for readability
            - DO NOT use any markdown formatting (no **, *, or other formatting) as this doesn't work on Facebook
        """,
        "behind_the_scenes": f"""
            Create a behind-the-scenes Facebook post for WriteNow Agency that shows the human side of the business.
            
            Use the following image caption as inspiration if relevant: "{image_caption}"
            
            Choose one angle from:
            - Day in the life of a team member
            - Project development process
            - Company culture or values in action
            - Office space or work environment
            - Recent team achievement or milestone
            
            Requirements:
            - Start with a personable greeting or statement that hooks readers
            - Share an authentic story or insight into the company's internal workings
            - Include specific details that make the story feel real and relatable
            - Subtly incorporate these SEO keywords: {keywords_text}
            - Show personality and company values
            - Use first-person language (we, our) to create connection
            - End with a question that invites followers to share their own experiences
            - Include the actual website URL (https://writenowagency.com) not [link]
            - Use emojis to convey emotion and personality
            - Optimize for Facebook algorithm by being authentic and relatable
            - Keep under {CHARACTER_LIMIT} characters
            - Format with appropriate line breaks for readability
            - DO NOT use any markdown formatting (no **, *, or other formatting) as this doesn't work on Facebook
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
    Facebook Page: {COMPANY_INFO['facebookPage']}
    """

    # Add previous posts to ensure uniqueness
    previous_posts_text = ""
    if previous_posts:
        previous_posts_text = f"""
        IMPORTANT: Your post must be completely different from these previous posts:
        {' '.join(previous_posts[-5:])}
        
        Ensure original wording, structure, and approach to avoid repetition.
        """

    # Final prompt
    final_prompt = f"{prompt}\n\n{company_info_text}\n\n{previous_posts_text}\n\nReturn ONLY the Facebook post text with appropriate line breaks and emojis. Do not include introductions or explanations."

    try:
        response = model.generate_content(final_prompt)
        content = response.text.strip()

        # Verify character limit
        if len(content) > CHARACTER_LIMIT:
            content = content[:CHARACTER_LIMIT]

        # Check if it's too similar to previous posts
        if previous_posts and any(
            calculate_similarity(content, prev) > 0.5 for prev in previous_posts
        ):
            logger.warning(
                "Generated content too similar to previous posts. Trying again."
            )
            return generate_facebook_content(
                model, content_type, image_caption, previous_posts
            )

        logger.info(f"Generated Facebook content: {content[:100]}...")
        return content
    except Exception as e:
        logger.error(f"Error generating Facebook content: {e}")
        return f"👋 Hey there! Looking to boost your business with custom tech solutions? WriteNow Agency offers cutting-edge web development, AI solutions, and business automation that can transform the way you work! Check out our services at https://writenowagency.com and let us know what challenges you're facing. We'd love to help you find the perfect solution! 💻✨ #TechSolutions #BusinessGrowth"


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


def replace_placeholders(text):
    """Replace placeholder strings like [link] with actual values"""
    result = text
    for placeholder, value in REPLACEMENTS.items():
        result = result.replace(placeholder, value)
    return result


def remove_markdown_formatting(text):
    """Remove Markdown formatting symbols from the text"""
    # Remove bold formatting (**text**)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

    # Remove italic formatting (*text*)
    text = re.sub(r"\*(.*?)\*", r"\1", text)

    # Remove other potential markdown elements if needed

    return text


def load_previous_posts(filename="previous_facebook_posts.txt"):
    """Load previous posts from a file"""
    try:
        if os.path.exists(filename):
            with open(filename, "r") as file:
                posts = [line.strip() for line in file.readlines()]
            return posts
        return []
    except Exception as e:
        logger.error(f"Error loading previous posts: {e}")
        return []


def save_post(post, filename="previous_facebook_posts.txt"):
    """Save a post to the previous posts file"""
    try:
        # Save just the first 300 characters to avoid huge files
        with open(filename, "a") as file:
            file.write(post[:300] + "\n")
    except Exception as e:
        logger.error(f"Error saving post: {e}")


def send_to_spreadsheet(text, image_path, image_url, image_caption, content_type):
    """Send generated content to Google Sheets via Apps Script"""
    try:
        # Get relative image path for spreadsheet
        image_filename = os.path.basename(image_path) if image_path else "No image"

        # Prepare payload for the Google Apps Script
        payload = {
            "action": "addFacebookContent",  # Changed action name for Facebook
            "apiKey": API_SECRET_KEY,
            "content": {
                "text": text,
                "imagePath": image_url or "No image",  # Use URL as path
                "imageUrl": image_url or "",
                "imageCaption": image_caption or "",
                "contentType": content_type,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "characterCount": len(text),
                "status": "Ready to Post",
                "platform": "Facebook",  # Specify platform
            },
        }

        # Enhanced debugging
        print(f"Sending data to: {WEBAPP_URL}")
        print(f"API Key (first 5 chars): {API_SECRET_KEY[:5]}...")
        print(f"Image URL being sent: {image_url}")
        print(f"Character count: {len(text)}")
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
                    logger.info("Successfully sent Facebook content to spreadsheet")
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
    """Main function to generate Facebook content and upload to Google Sheets"""
    try:
        # Validate configuration
        if not validate_config():
            return

        print("=== Facebook Content Generation for WriteNow Agency ===")

        # Setup Gemini
        print("Setting up Gemini AI...")
        model = setup_gemini()
        if not model:
            print("Failed to initialize Gemini model. Exiting.")
            return

        # Load previous posts to ensure uniqueness
        previous_posts = load_previous_posts()
        print(f"Loaded {len(previous_posts)} previous posts for comparison")

        # Get a random image, URL and caption from the image processor
        print("Selecting and uploading random image...")
        image_path, image_url, image_caption = get_random_image()

        if image_path:
            print(f"Selected image: {os.path.basename(image_path)}")
            print(f"Image URL: {image_url}")
            print(f"Image Caption: {image_caption}")
        else:
            print("No image selected. Continuing without an image.")
            image_caption = ""

        # Select content type
        content_type = select_content_type()
        print(f"Content type: {content_type}")

        # Generate Facebook content
        print("Generating Facebook content...")
        content = generate_facebook_content(
            model, content_type, image_caption, previous_posts
        )

        # Remove any Markdown formatting
        content = remove_markdown_formatting(content)

        print(f"\nGenerated Facebook content ({len(content)} chars):")
        print("---")
        print(content[:500] + ("..." if len(content) > 500 else ""))
        print("---")

        # Replace any placeholder strings
        content = replace_placeholders(content)

        # Send to spreadsheet with image URL and caption
        print(f"Sending content to spreadsheet...")
        success, url = send_to_spreadsheet(
            content, image_path, image_url, image_caption, content_type
        )

        if success:
            print("✓ Content added successfully to Google Sheets")
            if url:
                print(f"View your content at: {url}")

            # Save post to previous posts
            save_post(content)
            print("✓ Saved post to previous posts record")

            # Move used image to archive
            if image_path and move_used_image(image_path):
                print(f"✓ Moved used image to archive")
            elif image_path:
                print("✗ Failed to move image to archive")
        else:
            print("✗ Failed to add content to Google Sheets")

        print("\n=== Facebook Content Generation Complete ===")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()  # Print full traceback for better debugging


if __name__ == "__main__":
    main()

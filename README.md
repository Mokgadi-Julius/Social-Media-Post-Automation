# Social-Media-Post-AutomationSocial Media Post Automation

An AI-powered automation system for generating and scheduling social media content across Twitter, Facebook, and LinkedIn platforms.
📋 Overview

This application automates the creation and scheduling of social media content for multiple platforms. It uses Google's Gemini API to generate platform-specific content, uploads images to ImgBB, and sends the content to Google Sheets for review before posting.
Key features:

Generates platform-appropriate content for Twitter, Facebook, and LinkedIn
Integrates with trending topics on Twitter for relevance
Handles image selection, uploading, and management
Sends content to Google Sheets for review and scheduling
Built-in scheduler with health monitoring

🛠️ Technologies

Python 3.6+
Google Gemini API for AI-powered content generation
ImgBB API for image hosting
Google Apps Script for spreadsheet integration
APScheduler for automated scheduling
Beautiful Soup for web scraping trending topics

📁 Structure
Copy.
├── advanced_scheduler.py # Main scheduler for managing all content generation
├── image_processor.py # Core image processing for Twitter
├── twitter_content_generator.py # Twitter content generation
├── facebook_content_generator.py # Facebook content generation
├── facebook_image_processor.py # Image processor for Facebook
├── linkedin_content_generator.py # LinkedIn content generation
├── linkedin_image_processor.py # Image processor for LinkedIn
├── .env # Environment configuration (create this file)
├── images/ # Directory for Twitter images
├── facebook_images/ # Directory for Facebook images
├── linkedin_images/ # Directory for LinkedIn images
└── logs/ # Directory for log files
⚙️ Installation

Clone the repository:

bashCopygit clone https://github.com/Mokgadi-Julius/social-media-post-automation.git
cd social-media-post-automation

Create a virtual environment:

bashCopypython -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

Install dependencies:

bashCopypip install -r requirements.txt

Create required directories:

bashCopymkdir -p images/used
mkdir -p facebook_images/used
mkdir -p linkedin_images/used
mkdir -p logs

Create a .env file with your API keys:

CopyGEMINI_API_KEY=your_gemini_api_key
IMGBB_API_KEY=your_imgbb_api_key
WEBAPP_URL=your_google_apps_script_webapp_url
API_SECRET_KEY=your_secret_key_for_apps_script
EMAIL_NOTIFICATIONS=True
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USERNAME=your_email
EMAIL_PASSWORD=your_app_password
FROM_EMAIL=alerts@example.com
TO_EMAIL=admin@example.com
🖼️ Image Management

Place images in the appropriate platform directories:

Twitter: images/
Facebook: facebook_images/
LinkedIn: linkedin_images/

Images should be in JPG, PNG, or GIF format.
Used images are automatically moved to the used subdirectory to prevent repetition.

📊 Google Sheets Integration

Create a Google Apps Script that accepts JSON POST requests
The script should parse the content and add it to a Google Sheet
Configure your Web App URL and API key in the .env file

🚀 Usage
Run Individual Content Generators
bashCopy# Generate Twitter content
python twitter_content_generator.py

# Generate Facebook content

python facebook_content_generator.py

# Generate LinkedIn content

python linkedin_content_generator.py
Use the Advanced Scheduler
bashCopy# Start the scheduler (runs as a daemon)
python advanced_scheduler.py start

# Start with nohup (keeps running after terminal close)

python advanced_scheduler.py start --nohup

# Check status

python advanced_scheduler.py status

# Stop the scheduler

python advanced_scheduler.py stop

# Run a specific generator once

python advanced_scheduler.py run --script twitter
python advanced_scheduler.py run --script facebook
python advanced_scheduler.py run --script linkedin
⏱️ Scheduling
The advanced scheduler is configured to run:

Twitter content: 4 times per day
Facebook content: 3 times per day
LinkedIn content: 3 times per day

Times are optimized for each platform's peak engagement periods in the SAST (South African) timezone.
📝 Logging
Logs are stored in:

Main scheduler: logs/scheduler.log
Twitter: writenow_twitter.log
Facebook: writenow_facebook.log
LinkedIn: writenow_linkedin.log
Image processors: image_processor.log, facebook_image_processor.log, linkedin_image_processor.log

🔍 Troubleshooting
Common issues:

API Key errors: Ensure all API keys in .env are correct
Google Apps Script errors: Check the Web App URL and ensure the Apps Script is deployed correctly
Image upload failures: Verify ImgBB API key and check internet connection
Scheduler not running: Check logs for errors and ensure proper permissions

🛡️ Security

Keep your .env file secure and never commit it to version control
Use a .gitignore file to exclude sensitive information
Regularly rotate API keys and secrets

🧩 Extending
To add a new social media platform:

Create a new content generator script following the existing pattern
Create a corresponding image processor
Add configuration to the advanced scheduler
Update the Google Apps Script to handle the new content type

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
👥 Contributors

Your Julius Langa - Initial work

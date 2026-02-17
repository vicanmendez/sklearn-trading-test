import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_email():
    print("Testing Email Configuration...")
    
    sender = config.BREVO_SENDER_EMAIL
    recipient = config.EMAIL_RECIPIENT
    password = config.BREVO_API_KEY
    server_host = config.SMTP_SERVER
    port = config.SMTP_PORT
    
    print(f"Server: {server_host}:{port}")
    print(f"Sender: {sender}")
    print(f"Recipient: {recipient}")
    # Don't print password obviously
    
    if "YOUR_" in sender or "YOUR_" in password:
        logger.error("Please update config.py with your actual Brevo credentials first!")
        return

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = "Test Email from Trading Bot"

    body = "If you are reading this, your Brevo email configuration is working correctly!"
    msg.attach(MIMEText(body, 'plain'))

    try:
        print("Connecting to SMTP server...")
        with smtplib.SMTP(server_host, port) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
            
        print("SUCCESS: Test email sent successfully!")
    except Exception as e:
        print(f"FAILURE: Could not send email. Error: {e}")

if __name__ == "__main__":
    test_email()

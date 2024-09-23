import sys
import pysqlite3
# Force Python to use pysqlite3 instead of the older sqlite3
sys.modules['sqlite3'] = pysqlite3

import streamlit as st
from PIL import Image
from crewai import Agent, Task, Crew
from utils import get_openai_api_key
from crewai_tools import ScrapeWebsiteTool
import os

# Load the AI robot image
robot_image = Image.open('./DALL·E 2024-09-23 14.36.49 - A friendly and futuristic AI robot with a sleek, metallic design, standing upright and interacting with users through a glowing interface. The robot h.webp')

# Set OpenAI model and key
openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

# Title and AI robot introduction
st.title("🤖 Meet SensAI's Support Agent!")
st.write("This app simulates an AI support agent interacting with customer inquiries.")
st.image(robot_image, caption="Hello! I’m your AI assistant, ready to help!", use_column_width=True)

# Allow users to personalize their AI experience
st.subheader("Customize Your AI Experience")
greeting_style = st.radio(
    "Choose how you want the AI to greet you:",
    ("Friendly", "Professional", "Funny", "Helpful")
)

# Define personalized greetings
greetings = {
    "Friendly": "Hey there, friend! How can I make your day awesome?",
    "Professional": "Good day. How may I assist you with your query?",
    "Funny": "Why did the AI cross the road? To help you with your questions! 😄",
    "Helpful": "Here to help! Please tell me what you'd like to know."
}

st.markdown(f"### {greetings[greeting_style]}")

# User inputs for customer, person, and inquiry details
st.subheader("Submit a Customer Inquiry")
customer = st.text_input("Customer Name", "")
person = st.text_input("Contact Person", "")
inquiry = st.text_area("Customer Inquiry", "Please describe the issue or inquiry the customer has.")

# Dynamic message validation for input fields
if not customer or not person or not inquiry:
    st.warning("Please fill out all the fields before submitting.")
else:
    if st.button('Submit Inquiry'):
        # Define the support agent
        support_agent = Agent(
            role="Senior Support Representative",
            goal="Be the most friendly and helpful support representative in your team",
            backstory=(
                "You work at SensAI (https://sensai-consulting.com) and are now working on providing support to {customer}, a super important customer for your company."
                "You need to make sure that you provide the best support! Make sure to provide full complete answers, and make no assumptions."
                "You must have a call to action for SensAI who is your employer"
                "You are assiting customers on an app that means you must write in like you would when messaging not email"
                "Do not write a letter or email like, but more as a social media dm"
            ),
            allow_delegation=False,
            verbose=True
        )

        # Define quality assurance agent
        support_quality_assurance_agent = Agent(
            role="Support Quality Assurance Specialist",
            goal="Get recognition for providing the best support quality assurance in your team",
            backstory=(
                "You work at SensAI (https://sensai-consulting.com) and are now working with your team on a request from {customer} ensuring that the support representative "
                "is providing the best support possible.\n"
                "You need to make sure that the support representative is providing full, complete answers, and making no assumptions."
            ),
            verbose=True
        )

        # Tool for scraping website data
        docs_scrape_tool = ScrapeWebsiteTool(website_url="https://sensai-consulting.com")

        # Define task for inquiry resolution
        inquiry_resolution = Task(
            description=(
                "{customer} just reached out with a super important ask:\n"
                "{inquiry}\n\n"
                "{person} from {customer} is the one that reached out. "
                "Make sure to use everything you know to provide the best support possible."
                "You must strive to provide a complete and accurate response to the customer's inquiry."
            ),
            expected_output=(
                "A detailed, informative response to the customer's inquiry that addresses all aspects of their question.\n"
                "The response should include references to everything you used to find the answer, "
                "including external data or solutions. Ensure the answer is complete, "
                "leaving no questions unanswered, and maintain a helpful and friendly tone throughout."
            ),
            tools=[docs_scrape_tool],
            agent=support_agent,
        )

        # Define QA task (optional)
        quality_assurance_review = Task(
            description=(
                "Review the response drafted by the Senior Support Representative for {customer}'s inquiry. "
                "Ensure that the answer is comprehensive, accurate, and adheres to the high-quality standards expected for customer support.\n"
                "Verify that all parts of the customer's inquiry have been addressed thoroughly, with a helpful and friendly tone.\n"
                "Check for references and sources used to find the information, ensuring the response is well-supported and leaves no questions unanswered."
            ),
            expected_output=(
                "A final, detailed, and informative response ready to be sent to the customer. This response should fully address the customer's inquiry, "
                "incorporating all relevant feedback and improvements. Don't be too formal, we are a chill and cool company but maintain a professional and friendly tone throughout."
            ),
            agent=support_quality_assurance_agent,
        )

        # Define crew (team) to handle tasks
        crew = Crew(
            agents=[support_agent, support_quality_assurance_agent],  # Add/remove agents as needed
            tasks=[inquiry_resolution, quality_assurance_review],
            verbose=2,  # Higher verbosity for detailed logs
            memory=True
        )

        # Inputs for the crew
        inputs = {
            "customer": customer,
            "person": person,
            "inquiry": inquiry
        }

        # Display progress bar during AI processing
        st.markdown("🤖 **AI Support:** Processing your inquiry...")
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)

        # Run the task and get the result
        result = crew.kickoff(inputs=inputs)

        # Display result as Markdown
        st.markdown(result)

# Fun fact section for more user interaction
st.sidebar.header("🎮 Fun Fact Corner!")
st.sidebar.text("Did you know?\nAI can now create art, write code,\nand help you solve complex problems!")

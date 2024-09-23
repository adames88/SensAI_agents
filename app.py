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
st.write("SensAI Agent")
st.image(robot_image, caption="Hello! I’m your AI assistant, ready to help!", use_column_width=True)

# Allow users to personalize their AI experience
st.subheader("Customize Your AI Experience")
greeting_style = st.radio(
    "Choose how you want the AI to greet you:",
    ("Friendly", "Professional", "Funny", "Helpful")
)

# Define personalized greetings and agent tones
greetings = {
    "Friendly": "Hey there, friend! How can I make your day awesome?",
    "Professional": "Good day. How may I assist you with your query?",
    "Funny": "Why did the AI cross the road? To help you with your questions! 😄",
    "Helpful": "Here to help! Please tell me what you'd like to know."
}

# Define tones for agents based on greeting style
agent_tones = {
    "Friendly": (
        "You're the friendliest support agent around! Make sure to add warmth and kindness to all your responses, "
        "and make the customer feel valued."
    ),
    "Professional": (
        "You are a highly professional support agent. Your goal is to maintain a polite and formal tone while being thorough and efficient."
    ),
    "Funny": (
        "As a fun-loving agent, you're all about keeping the conversation lighthearted. Add some humor to your responses while still solving the problem."
    ),
    "Helpful": (
        "You're the helpful agent who is always ready to assist. Be extra detailed, making sure the customer feels fully supported every step of the way."
    )
}

# Display the selected greeting
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
        # Define the support agent with tone-based behavior
        support_agent = Agent(
            role="Senior Support Representative",
            goal=f"Be the most {greeting_style.lower()} and helpful support representative in your team.",
            backstory=(
                f"You work at SensAI (https://sensai-consulting.com) and are now working on providing support to {customer}, "
                "a super important customer for your company. You need to make sure that you provide the best support! "
                f"{agent_tones[greeting_style]}"
            ),
            allow_delegation=False,
            verbose=True
        )

        # Define quality assurance agent with tone-based behavior
        support_quality_assurance_agent = Agent(
            role="Support Quality Assurance Specialist",
            goal=f"Get recognition for providing the best support quality assurance in a {greeting_style.lower()} way.",
            backstory=(
                f"You work at SensAI (https://sensai-consulting.com) and are now working with your team on a request from {customer} "
                "ensuring that the support representative is providing the best support possible. "
                f"{agent_tones[greeting_style]}"
            ),
            verbose=True
        )

        # Tool for scraping website data
        docs_scrape_tool = ScrapeWebsiteTool(website_url="https://sensai-consulting.com")

        # Define task for inquiry resolution with agent tone
        inquiry_resolution = Task(
            description=(
                f"{customer} just reached out with a super important ask:\n{inquiry}\n\n"
                f"{person} from {customer} is the one that reached out. "
                "Make sure to use everything you know to provide the best support possible. "
                "You must strive to provide a complete and accurate response to the customer's inquiry."
            ),
            expected_output=(
                f"A detailed, informative response to the customer's inquiry that addresses all aspects of their question.\n"
                f"The response should be {greeting_style.lower()} and include references to everything you used to find the answer, "
                "including external data or solutions. Ensure the answer is complete, leaving no questions unanswered."
            ),
            tools=[docs_scrape_tool],
            agent=support_agent,
        )

        # Define QA task with agent tone (optional)
        quality_assurance_review = Task(
            description=(
                "Review the response drafted by the Senior Support Representative for {customer}'s inquiry. "
                f"Ensure that the answer is comprehensive, accurate, and maintains a {greeting_style.lower()} tone. "
                "Verify that all parts of the customer's inquiry have been addressed thoroughly, with a helpful and friendly tone."
            ),
            expected_output=(
                f"A final, detailed, and informative response ready to be sent to the customer. This response should fully address the customer's inquiry, "
                f"incorporating all relevant feedback and improvements, while maintaining a {greeting_style.lower()} tone throughout."
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
        st.markdown(f"🤖 **AI Support:** Processing your inquiry with a {greeting_style.lower()} tone...")
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)

        # Run the task and get the result
        result = crew.kickoff(inputs=inputs)

        # Display result with the corresponding tone
        st.markdown(f"### Here’s your {greeting_style.lower()} support response:")
        st.markdown(result)

# Fun fact section for more user interaction
st.sidebar.header("🎮 Fun Fact Corner!")
st.sidebar.text("Did you know?\nAI can now create art, write code,\nand help you solve complex problems!")

import streamlit as st
from crewai import Agent, Task, Crew
from utils import get_openai_api_key
from crewai_tools import ScrapeWebsiteTool

# Title and description
st.title("SensAI Support Agent")
st.write("This app simulates an AI support agent interacting with customer inquiries.")

# User inputs
customer = st.text_input("Customer Name", "")
person = st.text_input("Contact Person", "")
inquiry = st.text_area(
    "Customer Inquiry", 
    "Please describe the issue or inquiry the customer has."
)

# Dynamic message and validation
if not customer or not person or not inquiry:
    st.warning("Please fill out all the fields before submitting.")
else:
    if st.button('Submit Inquiry'):
        # Load the OpenAI API key
        openai_api_key = get_openai_api_key()
        
        # Define the support agent
        support_agent = Agent(
            role="Senior Support Representative",
            goal="Be the most friendly and helpful support representative in your team",
            backstory=(
                "You work at SensAI (https://sensai-consulting.com) and are now working on providing support to {customer}, a super important customer for your company."
                "You need to make sure that you provide the best support! Make sure to provide full complete answers, and make no assumptions."
            ),
            allow_delegation=False,
            verbose=True
        )
        
        # Define the quality assurance agent (optional if QA review is required)
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
        
        # Define tools for scraping website (you can modify or add more tools as needed)
        docs_scrape_tool = ScrapeWebsiteTool(website_url="https://sensai-consulting.com")
        
        # Define the task of resolving the inquiry
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

        # Define a task for QA review (optional)
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

        # Create a crew (team) to handle the tasks
        crew = Crew(
            agents=[support_agent, support_quality_assurance_agent],  # you can add/remove agents as needed
            tasks=[inquiry_resolution, quality_assurance_review],
            verbose=2,  # Higher verbosity for detailed logs
            memory=True
        )

        # Inputs to pass into the crew
        inputs = {
            "customer": customer,
            "person": person,
            "inquiry": inquiry
        }

        # Run the task and get the result
        result = crew.kickoff(inputs=inputs)

        # Display the result in Markdown format
        st.markdown(result)

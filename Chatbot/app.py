# This Streamlit application, "ShopEase: Your Personal Shopping Assistant," serves as an interactive
# interface between users and a chatbot powered by a FastAPI backend. The app includes a chat interface
# for user input and displays chatbot responses, as well as a sidebar with additional features:
# 1. Chat History Deletion: Allows users to clear the chat history and start a new conversation.
# 2. Feedback Section: Enables users to rate the chatbot's performance on a scale of 1-5, which is submitted to the backend.
# 3. Metrics Display: Retrieves and displays chatbot performance metrics, such as average response time and feedback score.
# 
# The main chat interface captures user input and displays responses from the backend. Communication with the
# backend occurs via POST and GET requests, handling user input and metrics/feedback respectively. Error handling
# is implemented to manage backend communication failures.

import streamlit as st
import requests

# Streamlit app configuration with a title
st.title("ShopEase: Your Personal Shopping Assistant")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Define the FastAPI backend's base URL
FASTAPI_BASE_URL = "http://127.0.0.1:8000"  

# Initialize session state variables for messages and chat history if not already set
if "messages" not in st.session_state:
    st.session_state.messages = []  # Stores chat history between user and assistant
    st.session_state.chat_history_ids = None  # Tracks conversation context

# Initialize session state for feedback visibility and submission status
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False  # Controls display of feedback section
if "rating_submitted" not in st.session_state:
    st.session_state.rating_submitted = False  # Tracks if feedback rating was submitted

# Sidebar options for chat controls and feedback
with st.sidebar:
    # Option to delete chat history and start fresh
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history_ids = None
        st.success("Chat history cleared. The chat will start fresh from the next message.")

    st.markdown("---")  # Divider for sidebar sections

    # Feedback section within an expandable container
    feedback_expander = st.expander("Rate this session")
    with feedback_expander:
        # Feedback form displayed if feedback hasn't been submitted yet
        if not st.session_state.rating_submitted:
            rating = st.slider("Please rate the chatbot (1-5):", 1, 5, key="feedback_rating")
            if st.button("Submit Rating", key="submit_rating"):
                feedback_url = f"{FASTAPI_BASE_URL}/feedback/"  # Feedback endpoint URL
                feedback_data = {"rating": rating}
                try:
                    # POST request to submit feedback rating to backend
                    feedback_response = requests.post(feedback_url, json=feedback_data)
                    feedback_response.raise_for_status()
                    st.success(f"Thank you for your feedback! You rated this session {rating}/5.")
                    st.session_state.rating_submitted = True  # Mark feedback as submitted
                except requests.exceptions.RequestException as e:
                    st.error("There was an error submitting your feedback.")  # Display error if request fails
        else:
            # Message shown if feedback was already submitted
            st.success("You have already submitted your feedback.")

    st.markdown("---")  # Divider for sidebar sections

    # Metrics section to display chatbot performance stats
    if st.button("Show Chatbot Metrics"):
        metrics_url = f"{FASTAPI_BASE_URL}/metrics/"  # Metrics endpoint URL
        try:
            # GET request to retrieve chatbot performance metrics from backend
            metrics_response = requests.get(metrics_url)
            metrics_response.raise_for_status()
            metrics = metrics_response.json()  # Parse JSON response
            # Display various chatbot metrics in sidebar
            st.write("**Chatbot Metrics:**")
            st.write(f"- **Average Response Time:** {metrics['average_response_time']:.2f} seconds")
            st.write(f"- **Average Feedback Score:** {metrics['average_feedback_score']:.2f}")
        except requests.exceptions.RequestException as e:
            st.error("There was an error retrieving the metrics.")  # Display error if request fails

# Display chat history in main chat interface
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Main chat input interface where users can enter their messages
if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})  # Add user input to chat history
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # Display assistant's response in chat interface
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()  # Placeholder for response
        full_response = ""

        try:
            # Make POST request to FastAPI backend to get assistant's response
            response = requests.post(
                f"{FASTAPI_BASE_URL}/chat/",
                json={
                    "user_input": prompt,
                    "chat_history_ids": st.session_state.chat_history_ids,
                },
            )
            response.raise_for_status()
            response_data = response.json()

            # Extract response content and update chat history context ID
            full_response = response_data.get("response", "Sorry, I didn't understand that.")
            st.session_state.chat_history_ids = response_data.get("chat_history_ids", None)

            # Display the assistant's response in the chat message
            message_placeholder.markdown(full_response)

            # Append assistant's response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except requests.exceptions.RequestException as e:
            # Display an error message in case of a backend communication issue
            error_message = "Error communicating with the backend."
            st.error(error_message)
            message_placeholder.markdown(error_message)

# FastAPI Chatbot with Customer, Order, and Product Management

# This code implements a FastAPI-based chatbot that handles user interactions related to customers, orders, 
# and products. It integrates a pre-trained DialoGPT model for conversational responses and includes several 
# endpoints for specific tasks:
# 1. Chatbot Endpoint (`/chat/`): Responds to user inputs by detecting intent (e.g., retrieving customer or 
#    order information, recommending products, or general conversation).
# 2. Feedback Endpoint (`/feedback/`): Collects user feedback ratings for each interaction session.
# 3. Metrics Endpoint (`/metrics/`): Provides statistics on average response time and average feedback score.

# Additionally, the code utilizes CSV datasets to manage and query customer, order, and product data.
# It tracks context to enhance response continuity, stores feedback and response times for analytics, 
# and supports interaction history using chat history IDs.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import datetime
import string
import time
import os
import base64
import io
from typing import Optional

# Step 1: Initialize the FastAPI application
app = FastAPI()

# Step 2: Load datasets from CSV files in "Cleaned_Datasets" directory
# Use the current file's directory as a base path for locating datasets
base_dir = os.path.dirname(os.path.abspath(__file__))
customers_df = pd.read_csv(os.path.join(base_dir, "../Cleaned_Datasets/customers_sg.csv"))
orders_df = pd.read_csv(os.path.join(base_dir, "../Cleaned_Datasets/orders.csv"))
products_df = pd.read_csv(os.path.join(base_dir, "../Cleaned_Datasets/products.csv"))

# Step 3: Ensure ID columns are in string format to avoid type mismatch issues
customers_df['customer_id'] = customers_df['customer_id'].astype(str)
orders_df['customer_id'] = orders_df['customer_id'].astype(str)
products_df['product_id'] = products_df['product_id'].astype(str)
orders_df['product_id'] = orders_df['product_id'].astype(str)

# Step 4: Load the pre-trained DialoGPT model and tokenizer for conversational responses
# "microsoft/DialoGPT-medium" model is used for generating chat responses
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 5: Set up context dictionary and tracking lists for user session and metrics
context = {
    "last_product": None,
    "last_intent": None,
    "last_customer": None,
    "last_order": None
}
feedback_scores = []  # Stores user feedback ratings
response_times = []    # Stores response times for each request

# Step 6: Define data models for the chat and feedback requests
class ChatRequest(BaseModel):
    user_input: str  # User message input
    chat_history_ids: Optional[str] = None  # Optional chat history for continuity

class FeedbackRequest(BaseModel):
    rating: int  # Rating given by user, expected to be between 1 and 5

# Step 7: Define a function to generate responses using the DialoGPT model
def chat_with_dialoGPT(user_input, chat_history_ids_base64=None):
    # Encode user input with end-of-sequence token
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    
    # Decode chat history if provided, maintaining conversation context
    if chat_history_ids_base64:
        try:
            chat_history_ids_bytes = base64.b64decode(chat_history_ids_base64)
            chat_history_ids_buffer = io.BytesIO(chat_history_ids_bytes)
            chat_history_ids_tensor = torch.load(chat_history_ids_buffer)
        except Exception as e:
            chat_history_ids_tensor = None
    else:
        chat_history_ids_tensor = None

    # Concatenate new input with existing chat history if available
    if chat_history_ids_tensor is not None:
        bot_input_ids = torch.cat([chat_history_ids_tensor, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    # Set attention mask for dynamic sequence length
    attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)

    # Generate response using DialoGPT, applying sampling strategies
    chat_history_ids_tensor = model.generate(
        bot_input_ids,
        max_length=1000,
        attention_mask=attention_mask,
        do_sample=True,
        top_p=0.92,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode generated response into text
    response = tokenizer.decode(chat_history_ids_tensor[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    # Serialize updated chat history for continuity
    buffer = io.BytesIO()
    torch.save(chat_history_ids_tensor, buffer)
    chat_history_ids_bytes = buffer.getvalue()
    chat_history_ids_base64 = base64.b64encode(chat_history_ids_bytes).decode('utf-8')

    return response, chat_history_ids_base64

# Step 8: Define functions for retrieving customer, order, and product information
def get_customer_info(customer_id):
    customer = customers_df[customers_df['customer_id'] == customer_id]
    
    if not customer.empty:
        last_login_days_ago = customer['last_login_day'].values[0]
        last_login_date = (datetime.datetime.now() - datetime.timedelta(days=int(last_login_days_ago))).strftime('%Y-%m-%d')
        
        checkout_message = "This customer has never checked out."
        
        last_checkout_days_ago = customer['last_checkout_day'].values[0]
        if str(last_checkout_days_ago).isdigit():
            last_checkout_date = (datetime.datetime.now() - datetime.timedelta(days=int(last_checkout_days_ago))).strftime('%Y-%m-%d')
            checkout_message = f"and their most recent checkout was {last_checkout_days_ago} days ago on {last_checkout_date}."
        
        context["last_customer"] = customer_id
        response = (f"Great news! We found the information for customer ID {customer_id}. Customer {customer_id} last logged in {last_login_days_ago} days ago on {last_login_date} {checkout_message}")
        return response
    
    return "Could not find the customer. Please check the ID!"

def recommend_products(customer_id):
    customer_orders = orders_df[orders_df['customer_id'] == customer_id]
    
    if customer_orders.empty:
        product_recommendations = products_df[['title', 'price_actual']].sample(3)
        response_intro = "No previous orders found. Here are some popular recommendations:\n\n"
    else:
        purchased_product_ids = customer_orders['product_id'].unique()
        available_products = products_df[~products_df['product_id'].isin(purchased_product_ids)]
        
        if available_products.empty:
            return "No new product recommendations at the moment."
        
        product_recommendations = available_products[['title', 'price_actual']].sample(min(3, len(available_products)))
        response_intro = "Based on your purchase history, I recommend these products:\n\n"
    
    product_recommendations['title'] = product_recommendations['title'].str.replace('|', '\\|', regex=False)
    recommendations_md = product_recommendations.to_markdown(index=False)
    
    response = response_intro + recommendations_md
    return response

def get_order_status(order_id):
    if order_id in orders_df['order_id'].astype(str).values:
        order = orders_df[orders_df['order_id'].astype(str) == str(order_id)].iloc[0]
        product_id = order['product_id']
        order_time = order['order_time']
        
        context["last_order"] = order_id  
        
        response = (
            f"Good news! Order {order_id} was placed on {order_time}. "
            f"The order includes product {product_id}. "
        )
        return response
    else:
        return (f"Oops, it seems there’s no order with ID {order_id}. "
                "Can you please check the ID and try again? I’m here to help!")

def get_product_info(product_name, info_type="price"):
    keywords = product_name.translate(str.maketrans('', '', string.punctuation)).split()
    
    matching_products = products_df[products_df['title'].apply(lambda x: all(kw.lower() in x.lower() for kw in keywords))]
    
    if not matching_products.empty:
        product = matching_products.iloc[0]
        price = product['price_actual']
        rating = product['item_rating']
        title = product['title']
        
        context["last_product"] = title
        context["last_intent"] = info_type
        
        if info_type == "price":
            response = f"I found the product you're looking for! The '{title}' is priced at {price}."
        else:
            response = f"The '{title}' has a rating of {rating} stars."
        return response
    
    return f"I couldn't find any product containing the keywords '{product_name}'."

# Step 12: Define a function to interpret user intent and extract relevant entities
def detect_intent_and_entities(user_input):
    if "recommend" in user_input.lower() and "customer" in user_input.lower():
        customer_id_match = re.search(r'\b\d+\b', user_input)
        customer_id = customer_id_match.group(0) if customer_id_match else context.get("last_customer")
        return "recommendation", {"customer_id": customer_id}

    if re.search(r'customer.*id.*\d+', user_input, re.IGNORECASE):
        customer_id = re.search(r'\d+', user_input).group(0)
        return "customer_query", {"customer_id": customer_id}
    
    if re.search(r'order.*id', user_input, re.IGNORECASE):
        order_id_match = re.search(r'\b\d+\b', user_input)
        order_id = order_id_match.group(0) if order_id_match else None
        return "order_query", {"order_id": order_id}
    
    if re.search(r'price|rating', user_input, re.IGNORECASE):
        product_name_match = re.search(r'(?:price|rating) of (.+)', user_input, re.IGNORECASE)
        product_name = product_name_match.group(1).strip() if product_name_match else None
        return ("product_price" if "price" in user_input.lower() else "product_rating", {"product_name": product_name})
    
    return "general", {}

# Step 13: Define the chat endpoint for handling chat requests and generating responses
@app.post("/chat/")
async def chat_with_bot(request: ChatRequest):
    start_time = time.time()
    
    user_input = request.user_input
    chat_history_ids_base64 = request.chat_history_ids or None

    intent, entities = detect_intent_and_entities(user_input)

    if intent == "customer_query":
        response = get_customer_info(entities["customer_id"])
        chat_history_ids_base64 = None
    elif intent == "order_query":
        response = get_order_status(entities["order_id"])
        chat_history_ids_base64 = None
    elif intent in ["product_price", "product_rating"]:
        product_name = entities.get("product_name") or context.get("last_product")
        if product_name:
            response = get_product_info(product_name, info_type="price" if intent == "product_price" else "rating")
        else:
            response = "Please specify a product name."
        chat_history_ids_base64 = None
    elif intent == "recommendation":
        response = recommend_products(entities["customer_id"])
        chat_history_ids_base64 = None
    else:
        response, chat_history_ids_base64 = chat_with_dialoGPT(user_input, chat_history_ids_base64)
    
    end_time = time.time()
    response_time = end_time - start_time
    response_times.append(response_time)

    return {
        "response": response,
        "chat_history_ids": chat_history_ids_base64
    }

# Step 14: Define the feedback endpoint for receiving and validating user ratings
@app.post("/feedback/")
async def collect_feedback(feedback: FeedbackRequest):
    if not (1 <= feedback.rating <= 5):
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5.")
    feedback_scores.append(feedback.rating)
    return {"message": f"Thank you for your feedback! You rated this session {feedback.rating}/5."}

# Step 15: Define metrics endpoint to retrieve average response time and feedback score
@app.get("/metrics/")
async def get_metrics():
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    avg_feedback_score = sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0
    
    return {
        "average_response_time": avg_response_time,
        "average_feedback_score": avg_feedback_score
    }

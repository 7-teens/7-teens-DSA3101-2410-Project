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

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import datetime
import string
import time
import os

# Step 1: Initialize FastAPI app instance
app = FastAPI()

# Step 2: Load datasets from CSV files
# Load customer, order, and product data from "Cleaned_Datasets" folder
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
customers_df = pd.read_csv(os.path.join(base_dir, "../Cleaned_Datasets/customer_SG_only.csv"), index_col=0)
orders_df = pd.read_csv(os.path.join(base_dir, "../Cleaned_Datasets/orders_generated.csv"), index_col=0)
products_df = pd.read_csv(os.path.join(base_dir, "../Cleaned_Datasets/products_cleaned.csv"), index_col=0)

# Step 3: Ensure customer_id columns are of string type to avoid mismatches
customers_df['customer_id'] = customers_df['customer_id'].astype(str)
orders_df['customer_id'] = orders_df['customer_id'].astype(str)

# Step 4: Load the pre-trained DialoGPT model and tokenizer for conversational responses
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Load tokenizer for processing text
model = AutoModelForCausalLM.from_pretrained(model_name)  # Load language model for generating responses

# Step 5: Define context dictionary and lists to track user session and metrics
context = {
    "last_product": None,
    "last_intent": None,
    "last_customer": None,
    "last_order": None
}
feedback_scores = []  # Stores feedback ratings
response_times = []    # Stores response times

# Step 6: Define data models for request bodies (chat and feedback)
class ChatRequest(BaseModel):
    user_input: str  # User message
    chat_history_ids: list = None  # Optional chat history for context

class FeedbackRequest(BaseModel):
    rating: int  # Feedback rating from 1 to 5

# Step 7: Define a function to generate responses using DialoGPT
def chat_with_dialoGPT(user_input, chat_history_ids=None):
    # Step 7a: Encode user input with an end-of-sentence token
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    
    # Step 7b: Concatenate new input with chat history if provided, to maintain context
    bot_input_ids = torch.cat([torch.tensor(chat_history_ids), new_input_ids], dim=-1) if chat_history_ids else new_input_ids
    
    # Step 7c: Set up an attention mask for variable sequence lengths
    attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)
    
    # Step 7d: Generate response from DialoGPT with sampling strategies for varied responses
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        attention_mask=attention_mask,
        do_sample=True,
        top_p=0.92,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )

    # Step 7e: Decode tokens into human-readable response and return with updated chat history
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids.tolist()

# Step 8: Define a function to get customer information based on customer ID
def get_customer_info(customer_id):
    # Step 8a: Search for the customer in the DataFrame
    customer = customers_df[customers_df['customer_id'] == customer_id]
    
    if not customer.empty:  # If customer is found
        # Step 8b: Calculate last login date based on days ago
        last_login_days_ago = customer['last_login_day'].values[0]
        last_login_date = (datetime.datetime.now() - datetime.timedelta(days=int(last_login_days_ago))).strftime('%Y-%m-%d')
        
        # Step 8c: Default checkout message if no checkout history
        checkout_message = "This customer has never checked out."
        
        # Step 8d: Format last checkout date if available
        last_checkout_days_ago = customer['last_checkout_day'].values[0]
        if str(last_checkout_days_ago).isdigit():
            last_checkout_date = (datetime.datetime.now() - datetime.timedelta(days=int(last_checkout_days_ago))).strftime('%Y-%m-%d')
            checkout_message = f"and their most recent checkout was {last_checkout_days_ago} days ago on {last_checkout_date}."
        
        # Step 8e: Update context and return formatted customer information
        context["last_customer"] = customer_id
        return f"Customer {customer_id} last logged in {last_login_days_ago} days ago on {last_login_date} {checkout_message}"
    
    # Step 8f: Return error message if customer not found
    return "Could not find the customer. Please check the ID!"

# Step 9: Define a function to recommend products based on customer purchase history
def recommend_products(customer_id):
    # Step 9a: Retrieve orders for the specified customer
    customer_orders = orders_df[orders_df['customer_id'] == customer_id]
    
    # Step 9b: If no previous orders, recommend popular products randomly
    if customer_orders.empty:
        product_recommendations = products_df[['title', 'price_actual']].sample(3)
        return "This customer has no past orders. Here are some popular recommendations:\n" + \
               product_recommendations.to_string(index=False)
    
    # Step 9c: Get IDs of products already purchased by customer
    purchased_product_ids = customer_orders['product_id'].unique()
    
    # Step 9d: Filter out already purchased products from recommendations
    available_products = products_df[~products_df['product_id'].isin(purchased_product_ids)]
    
    # Step 9e: If no new products are available, return message
    if available_products.empty:
        return "No new product recommendations at the moment."
    
    # Step 9f: Sample and return up to 3 new product recommendations
    recommended_products = available_products.sample(min(3, len(available_products)))
    return f"Based on your purchase history, I recommend these products:\n{recommended_products[['title', 'price_actual']].to_string(index=False)}"

# Step 10: Define a function to get order status based on order ID
def get_order_status(order_id):
    # Step 10a: Search for the order in orders DataFrame
    order = orders_df[orders_df['order_id'] == order_id]
    
    if not order.empty:  # If order is found
        # Step 10b: Extract product ID and order time
        product_id = order['product_id'].values[0]
        order_time = order['order_time'].values[0]
        
        # Step 10c: Update context and return formatted order status
        context["last_order"] = order_id
        return f"Order {order_id} was placed on {order_time} and includes product {product_id}."
    
    # Step 10d: Return error message if order not found
    return "Could not find the order. Please check the ID!"

# Step 11: Define a function to get product information (price or rating) based on product name
def get_product_info(product_name, info_type="price"):
    # Step 11a: Remove punctuation from product name and split into keywords
    keywords = product_name.translate(str.maketrans('', '', string.punctuation)).split()
    
    # Step 11b: Filter products where all keywords are in the title
    matching_products = products_df[products_df['title'].apply(lambda x: all(kw.lower() in x.lower() for kw in keywords))]
    
    if not matching_products.empty:  # If matching products found
        # Step 11c: Get price or rating and update context
        product = matching_products.iloc[0]
        price = product['price_actual']
        rating = product['item_rating']
        title = product['title']
        
        # Update context for last product and intent
        context["last_product"] = title
        context["last_intent"] = info_type
        
        # Return price or rating based on info_type
        return f"I found the product you're looking for! The '{title}' is priced at {price}." if info_type == "price" else f"The '{title}' has a rating of {rating} stars."
    
    # Step 11d: Return error if product not found
    return f"I couldn't find any product containing the keywords '{product_name}'."

# Step 12: Define a function to detect intent and extract entities from user input
def detect_intent_and_entities(user_input):
    # Step 12a: Check if user is requesting recommendations for a customer
    if "recommend" in user_input.lower() and "customer" in user_input.lower():
        customer_id = re.search(r'\d+', user_input).group(0)
        return "recommendation", {"customer_id": customer_id}

    # Step 12b: Check if user is querying customer information
    if re.search(r'customer.*id.*\d+', user_input, re.IGNORECASE):
        customer_id = re.search(r'\d+', user_input).group(0)
        return "customer_query", {"customer_id": customer_id}
    
    # Step 12c: Check if user is querying order status
    if re.search(r'order.*id', user_input, re.IGNORECASE):
        order_id = re.search(r'order_[0-9]+', user_input, re.IGNORECASE)
        return "order_query", {"order_id": order_id.group(0)} if order_id else {"order_id": None}
    
    # Step 12d: Check if user is asking for product price or rating
    if re.search(r'price|rating', user_input, re.IGNORECASE):
        product_name_match = re.search(r'(?:price|rating) of (.+)', user_input, re.IGNORECASE)
        product_name = product_name_match.group(1).strip() if product_name_match else None
        return ("product_price" if "price" in user_input.lower() else "product_rating", {"product_name": product_name})
    
    # Step 12e: Return general intent if no specific pattern is found
    return "general", {}

# Step 13: Define the chat endpoint to handle requests and generate responses
@app.post("/chat/")
async def chat_with_bot(request: ChatRequest):
    start_time = time.time()  # Step 13a: Start response time measurement
    
    user_input = request.user_input  # Step 13b: Extract user input
    chat_history_ids = request.chat_history_ids or None  # Get chat history if available

    # Step 13c: Detect intent and entities
    intent, entities = detect_intent_and_entities(user_input)

    # Step 13d: Handle specific queries based on detected intent
    if intent == "customer_query":
        response = get_customer_info(entities["customer_id"])
    elif intent == "order_query":
        response = get_order_status(entities["order_id"])
    elif intent in ["product_price", "product_rating"]:
        product_name = entities["product_name"] or context.get("last_product")
        if product_name:
            response = get_product_info(product_name, info_type="price" if intent == "product_price" else "rating")
        else:
            response = "Please specify a product name."
    elif intent == "recommendation":
        response = recommend_products(entities["customer_id"])
    else:
        # Step 13e: Use DialoGPT for general responses
        response, chat_history_ids = chat_with_dialoGPT(user_input, chat_history_ids)

    # Step 13f: Calculate and store response time
    end_time = time.time()
    response_time = end_time - start_time
    response_times.append(response_time)

    # Step 13g: Return response and chat history (if applicable)
    return {
        "response": response,
        "chat_history_ids": chat_history_ids if intent == "general" else None
    }

# Step 14: Define endpoint to collect feedback ratings
@app.post("/feedback/")
async def collect_feedback(feedback: FeedbackRequest):
    # Step 14a: Validate rating range (1 to 5)
    if not (1 <= feedback.rating <= 5):
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5.")
    feedback_scores.append(feedback.rating)  # Store the rating
    return {"message": f"Thank you for your feedback! You rated this session {feedback.rating}/5."}

# Step 15: Define endpoint to retrieve average response time, feedback score, and total feedback count
@app.get("/metrics/")
async def get_metrics():
    # Step 15a: Calculate average response time and feedback score
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    avg_feedback_score = sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0
    
    # Step 15b: Return calculated metrics
    return {
        "average_response_time": avg_response_time,
        "average_feedback_score": avg_feedback_score,
        "total_feedback_count": len(feedback_scores)
    }
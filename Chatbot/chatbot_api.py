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

# Initialize FastAPI app
app = FastAPI()

# Load datasets
base_dir = os.path.dirname(os.path.abspath(__file__))
customers_df = pd.read_csv(os.path.join(base_dir, "../Cleaned_Datasets/customer_SG_only.csv"), index_col=0)
orders_df = pd.read_csv(os.path.join(base_dir, "../Cleaned_Datasets/orders_generated.csv"), index_col=0)
products_df = pd.read_csv(os.path.join(base_dir, "../Cleaned_Datasets/products_cleaned.csv"), index_col=0)

# Convert customer_id columns to strings
customers_df['customer_id'] = customers_df['customer_id'].astype(str)
orders_df['customer_id'] = orders_df['customer_id'].astype(str)

# Load pre-trained DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Context management
context = {
    "last_product": None,
    "last_intent": None,
    "last_customer": None,
    "last_order": None
}

feedback_scores = []  # List to store feedback ratings
response_times = []   # List to store response times

class ChatRequest(BaseModel):
    user_input: str
    chat_history_ids: list = None

class FeedbackRequest(BaseModel):
    rating: int

# Function to use DialoGPT for general responses
def chat_with_dialoGPT(user_input, chat_history_ids=None):
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([torch.tensor(chat_history_ids), new_input_ids], dim=-1) if chat_history_ids else new_input_ids
    attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)
    
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        attention_mask=attention_mask,
        do_sample=True,
        top_p=0.92,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids.tolist()

# Function to get customer information
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
        return f"Customer {customer_id} last logged in {last_login_days_ago} days ago on {last_login_date} {checkout_message}"
    
    return "Could not find the customer. Please check the ID!"

# Function to recommend products based on customer history
def recommend_products(customer_id):
    customer_orders = orders_df[orders_df['customer_id'] == customer_id]
    if customer_orders.empty:
        product_recommendations = products_df[['title', 'price_actual']].sample(3)
        return "This customer has no past orders. Here are some popular recommendations:\n" + \
               product_recommendations.to_string(index=False)
    
    purchased_product_ids = customer_orders['product_id'].unique()
    available_products = products_df[~products_df['product_id'].isin(purchased_product_ids)]
    
    if available_products.empty:
        return "No new product recommendations at the moment."
    
    recommended_products = available_products.sample(min(3, len(available_products)))
    return f"Based on your purchase history, I recommend these products:\n{recommended_products[['title', 'price_actual']].to_string(index=False)}"

# Function to get order status
def get_order_status(order_id):
    order = orders_df[orders_df['order_id'] == order_id]
    if not order.empty:
        product_id = order['product_id'].values[0]
        order_time = order['order_time'].values[0]
        context["last_order"] = order_id
        return f"Order {order_id} was placed on {order_time} and includes product {product_id}."
    return "Could not find the order. Please check the ID!"

# Function to get product info based on last intent (price or rating)
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
        return f"I found the product you're looking for! The '{title}' is priced at {price}." if info_type == "price" else f"The '{title}' has a rating of {rating} stars."
    return f"I couldn't find any product containing the keywords '{product_name}'."

# Detect intent and extract entities
def detect_intent_and_entities(user_input):
    if "recommend" in user_input.lower() and "customer" in user_input.lower():
        customer_id = re.search(r'\d+', user_input).group(0)
        return "recommendation", {"customer_id": customer_id}

    if re.search(r'customer.*id.*\d+', user_input, re.IGNORECASE):
        customer_id = re.search(r'\d+', user_input).group(0)
        return "customer_query", {"customer_id": customer_id}
    
    if re.search(r'order.*id', user_input, re.IGNORECASE):
        order_id = re.search(r'order_[0-9]+', user_input, re.IGNORECASE)
        return "order_query", {"order_id": order_id.group(0)} if order_id else {"order_id": None}
    
    if re.search(r'price|rating', user_input, re.IGNORECASE):
        product_name_match = re.search(r'(?:price|rating) of (.+)', user_input, re.IGNORECASE)
        product_name = product_name_match.group(1).strip() if product_name_match else None
        return ("product_price" if "price" in user_input.lower() else "product_rating", {"product_name": product_name})
    
    return "general", {}

# Chat endpoint
@app.post("/chat/")
async def chat_with_bot(request: ChatRequest):
    start_time = time.time()  # Start timing the response
    
    user_input = request.user_input
    chat_history_ids = request.chat_history_ids or None

    # Detect intent
    intent, entities = detect_intent_and_entities(user_input)

    # Handle specific queries based on intent
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
        # Fall back to DialoGPT for general queries
        response, chat_history_ids = chat_with_dialoGPT(user_input, chat_history_ids)

    # End timing and calculate the response time
    end_time = time.time()
    response_time = end_time - start_time
    response_times.append(response_time)  # Store response time

    return {
        "response": response,
        "chat_history_ids": chat_history_ids if intent == "general" else None  # Only return chat history if using DialoGPT
    }

# Endpoint to collect feedback from the user
@app.post("/feedback/")
async def collect_feedback(feedback: FeedbackRequest):
    if not (1 <= feedback.rating <= 5):
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5.")
    feedback_scores.append(feedback.rating)
    return {"message": f"Thank you for your feedback! You rated this session {feedback.rating}/5."}

# Endpoint to view metrics like average response time and feedback score
@app.get("/metrics/")
async def get_metrics():
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    avg_feedback_score = sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0
    return {
        "average_response_time": avg_response_time,
        "average_feedback_score": avg_feedback_score,
        "total_feedback_count": len(feedback_scores)
    }


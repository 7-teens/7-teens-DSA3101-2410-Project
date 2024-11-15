# 7-teens-DSA3101-2410-Project
# Repository for 7-teens DSA3101 2410 Project

# Project Overview
E-commerce platforms like Shopee Singapore face numerous challenges in optimizing operations, understanding customer behavior, and maximizing profitability. This project aims to harness advanced data analysis techniques to gain deep insights into various facets of the e-commerce business. By focusing on sales trends, customer segmentation, inventory management, and pricing strategies, we will develop a robust analytical framework. This framework will empower e-commerce businesses to make informed, data-driven decisions that enhance performance, boost customer satisfaction, and drive revenue growth.

# Project Objective
The primary objective of this project is to create a comprehensive data analysis framework that enables Shopee Singapore to optimize its operations and strategic decision-making processes. Our team of eight data scientists seek to address key business questions, including:

Customer Behavior and Sales Analysis: To identify key factors influencing customer purchasing behavior, improve customer retention and lifetime value, and determine the most effective marketing channels and campaigns. This involves analyzing historical sales data, developing customer segmentation models, calculating churn rates, and evaluating the ROI of different marketing strategies.

Inventory Management and Pricing Optimization: To optimize inventory levels to minimize costs while ensuring product availability, implement effective pricing strategies to maximize revenue, and enhance supply chain efficiency. This includes developing demand forecasting models, creating inventory optimization algorithms, analyzing price elasticity, and optimizing order fulfillment processes.

By addressing these business questions, we aim to provide actionable insights that will drive Shopee Singapore’s growth and operational excellence. Our comprehensive analysis framework will enable the company to make data-driven decisions that enhance customer satisfaction, streamline operations, and increase profitability.


# Instructions for running the Docker container
Follow these steps to set up and run the project using Docker.

## Prerequisites
- Ensure you have [Docker](https://docs.docker.com/get-docker/) installed.

## Step-by-Step Guide

### Step 1: Clone the Repository
Clone the repository to your local machine and navigate to the project directory:
```
git clone https://github.com/7-teens/7-teens-DSA3101-2410-Project/
cd 7-teens-DSA3101-2410-Project
```
### Step 2: Build and Start the Docker Container
Run the following command to build and start the container:
```
docker-compose up --build
```
This command builds the Docker image using the `Dockerfile` and starts the container as defined in `docker-compose.yml`.

### Step 3: Access the Application
Once the container is running, open a web browser and go to:
```
http://127.0.0.1:1111/tree
```
### Step 4: Stop the Container
To stop the container and clean up, use:
```
docker-compose down
```


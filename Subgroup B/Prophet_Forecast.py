from prophet import Prophet
import plotly.graph_objects as go
import numpy as np
from datetime import timedelta

def forecast_next_n_months(train_df, regressors, n, current_stock, current_produce_cost, current_holding_cost, lead_time, product_price):
    df = train_df.groupby('order_time')['daily_sales'].sum().reset_index()
    df.columns = ['ds', 'y']
    for event, dates in regressors.items():
        df[f'{event}'] = df['ds'].isin(dates).astype(int)

    model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
    for event in regressors.keys():
        model.add_regressor(event)

    model.fit(df)

    future_dates = model.make_future_dataframe(periods=n*31, freq='D')
    start_date = df['ds'].max() + timedelta(days=1)
    end_date = start_date + timedelta(days=n*31)
    future_dates = future_dates[(future_dates['ds'] >= start_date) & (future_dates['ds'] <= end_date)]
    for event, dates in regressors.items():
            future_dates[f'{event}'] = future_dates['ds'].isin(dates).astype(int)

    forecast = model.predict(future_dates)
    
    # Calculate safety stock based on Prophet's upper bound forecast
    forecast['safety_stock'] = (forecast['yhat_upper'] - forecast['yhat']) * lead_time

    # Calculate the theoretical optimal inventory level (Prophet safety stock) for each day in August
    forecast['prophet_safety_stock'] = forecast['yhat'] + forecast['safety_stock']

    # Ensure non-negative values for demand predictions and safety stock
    forecast['yhat'] = np.maximum(0, forecast['yhat'])
    forecast['yhat_lower'] = np.maximum(0, forecast['yhat_lower'])
    forecast['prophet_safety_stock'] = np.maximum(0, forecast['prophet_safety_stock'])

    def plot_forecast(forecast):
        # Plot the forecast with actual August data for comparison
        fig = go.Figure()

        # Prophet Forecast
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'], mode='lines', name="Prophet Forecast", line=dict(color='red', dash='dash'), legendgroup="forecast"
        ))

        # Forecast Interval (shaded area), grouped with Prophet Forecast
        fig.add_trace(go.Scatter(
            x=forecast['ds'].tolist() + forecast['ds'][::-1].tolist(),
            y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'][::-1].tolist(),
            fill='toself', fillcolor='rgba(255, 0, 0, 0.2)', line=dict(color='rgba(255, 0, 0, 0)'),
            hoverinfo="skip", showlegend=False, legendgroup="forecast"
        ))

        # Prophet Safety Stock
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['prophet_safety_stock'],
            mode='lines', name="Prophet Safety Stock",
            line=dict(color='blue', dash='dash')
        ))


        # Layout adjustments
        fig.update_layout(
            title=f"Prophet August Forecast",
            xaxis_title="Date", yaxis_title="Daily Sales",
            template="plotly_white",
        )

        fig.show()
    
    def plot_inventory_strategy(forecast, current_stock, current_produce_cost, current_holding_cost, lead_time):
        forecast['inventory_level'] = current_stock
        optimal_inventory_levels = [current_stock]
        optimal_reorder_points = []
        optimal_reorder_labels = []
        optimal_order_costs = []
        optimal_delayed_orders = {}
        optimal_reorder_lock = False


        for i, row in forecast.iterrows():
            current_optimal_inventory = optimal_inventory_levels[-1]
            reorder_cost_optimal = 0

            if not optimal_reorder_lock:
                projected_inventory = current_optimal_inventory
                future_demand = forecast['prophet_safety_stock'][i:i + lead_time].sum()
                if (projected_inventory - future_demand) <= row['prophet_safety_stock']:
                    reorder_amount_optimal = int(future_demand - projected_inventory)
                    reorder_cost_optimal = reorder_amount_optimal * current_produce_cost
                    optimal_reorder_points.append((row['ds'], current_optimal_inventory))
                    optimal_reorder_labels.append(f"+{reorder_amount_optimal}")
                    arrival_day = i + lead_time
                    optimal_delayed_orders[arrival_day] = optimal_delayed_orders.get(arrival_day, 0) + reorder_amount_optimal
                    optimal_reorder_lock = True

            # Add holding cost and update inventory after demand and delayed stock arrival
            optimal_order_costs.append(reorder_cost_optimal + current_optimal_inventory * current_holding_cost)
            current_optimal_inventory = max(current_optimal_inventory - int(row['yhat']), 0) + optimal_delayed_orders.get(i, 0)

            # Release the reorder lock if stock has arrived
            if i in optimal_delayed_orders:
                optimal_reorder_lock = False
            optimal_inventory_levels.append(current_optimal_inventory)

    

        # Update forecast with calculated inventory levels and costs for each strategy
        forecast['optimal_inventory_level'] = optimal_inventory_levels[:-1]
        forecast['optimal_order_cost'] = np.cumsum(optimal_order_costs)

        # Plot inventory levels, reorder points, and safety stock levels
        fig = go.Figure()

        # Optimal Dynamic Inventory Level
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['optimal_inventory_level'],
            mode='lines', name="Optimal Dynamic Inventory Level", line=dict(color='orange', dash='dash')
        ))

        # Optimal Reorder Points with labels
        fig.add_trace(go.Scatter(
            x=[point[0] for point in optimal_reorder_points],
            y=[point[1] for point in optimal_reorder_points],
            mode='markers+text', marker_symbol='star', marker_size=10, marker_color='orange',
            text=optimal_reorder_labels, textposition="top center",
            name="Optimal Reorder Points"
        ))


        # Safety Stock Level
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['prophet_safety_stock'],
            mode='lines', name="Safety Stock Level", line=dict(color='red', dash='dot')
        ))


        # Update layout
        fig.update_layout(
            title="Inventory Levels, Reorder Points, and Safety Stock with Lead Time and Future Demand Consideration",
            xaxis_title="Date",
            yaxis_title="Inventory / Demand",
            legend=dict(x=1.02, y=1, bordercolor="Black", borderwidth=1),
            width=1000,
            height=900,
            template="plotly_white",
            showlegend=True
        )

        fig.show()

                
        # Cost comparison plot
        fig_cost = go.Figure()

        # Optimal Order Cumulative Cost
        fig_cost.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['optimal_order_cost'],
            mode='lines', name="Optimal Order Cumulative Cost", line=dict(color='orange', dash='dash')
        ))


        # Actual Cumulative Revenue
        fig_cost.add_trace(go.Scatter(
            x=forecast['ds'], y= np.cumsum(forecast['yhat'] * product_price),
            mode='lines', name="Forecasted Cumulative Revenue", line=dict(color='green')
        ))

        # Update layout for cost comparison
        fig_cost.update_layout(
            title="Cumulative Cost and Revenue Comparison for Inventory Management Strategies",
            xaxis_title="Date",
            yaxis_title="Cumulative Cost and Revenue",
            legend=dict(x=1.02, y=1, bordercolor="Black", borderwidth=1),
            template="plotly_white",
            showlegend=True,
            width=1000,
            height=900
        )

        fig_cost.show()

    plot_forecast(forecast)
    plot_inventory_strategy(forecast, current_stock, current_produce_cost, current_holding_cost, lead_time)

    return forecast


    



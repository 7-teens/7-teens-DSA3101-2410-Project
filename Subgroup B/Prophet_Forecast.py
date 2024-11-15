from prophet import Prophet
import plotly.graph_objects as go
import numpy as np
from datetime import timedelta, datetime

def forecast_next_n_months(train_df, lead_time, regressors=None, n=1):
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

    start_date_title = start_date.strftime("%d %B %Y")
    end_date_title = end_date.strftime("%d %B %Y")

    forecast = model.predict(future_dates)
    forecast['safety_stock'] = (forecast['yhat_upper'] - forecast['yhat']) * lead_time
    forecast['prophet_safety_stock'] = forecast['yhat'] + forecast['safety_stock']

    forecast['yhat'] = np.maximum(0, forecast['yhat'])
    forecast['yhat_lower'] = np.maximum(0, forecast['yhat_lower'])
    forecast['prophet_safety_stock'] = np.maximum(0, forecast['prophet_safety_stock'])

    def plot_forecast(forecast):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'], mode='lines', name="Prophet Forecast", line=dict(color='red', dash='dash'), legendgroup="forecast"
        ))

        fig.add_trace(go.Scatter(
            x=forecast['ds'].tolist() + forecast['ds'][::-1].tolist(),
            y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'][::-1].tolist(),
            fill='toself', fillcolor='rgba(255, 0, 0, 0.2)', line=dict(color='rgba(255, 0, 0, 0)'),
            hoverinfo="skip", showlegend=False, legendgroup="forecast"
        ))

        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['prophet_safety_stock'],
            mode='lines', name="Prophet Forecast Safety Stock",
            line=dict(color='blue', dash='dash')
        ))

        fig.update_layout(
            title=f"Prophet Forecast for {start_date_title} to {end_date_title}",
            xaxis_title="Date", yaxis_title="Daily Sales",
            template="plotly_white",
        )

        fig.show()

    plot_forecast(forecast)
    return forecast

def plot_inventory_strategy(forecast, current_stock, current_produce_cost, current_holding_cost, lead_time, product_price):
    forecast['inventory_level'] = current_stock
    optimal_inventory_levels = [current_stock]
    optimal_reorder_points = []
    optimal_reorder_labels = []
    optimal_order_costs = []
    optimal_delayed_orders = {}
    optimal_reorder_lock = False

    start_date = forecast['ds'].min()
    end_date = forecast['ds'].max()
    start_date_title = start_date.strftime("%d %B %Y")
    end_date_title = end_date.strftime("%d %B %Y")

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

        optimal_order_costs.append(reorder_cost_optimal + current_optimal_inventory * current_holding_cost)
        current_optimal_inventory = max(current_optimal_inventory - int(row['yhat']), 0) + optimal_delayed_orders.get(i, 0)

        if i in optimal_delayed_orders:
            optimal_reorder_lock = False
        optimal_inventory_levels.append(current_optimal_inventory)

    forecast['optimal_inventory_level'] = optimal_inventory_levels[:-1]
    forecast['optimal_order_cost'] = np.cumsum(optimal_order_costs)

    # Calculate final cumulative profit for each strategy
    optimal_order_final_cost = forecast['optimal_order_cost'].iloc[-1]
    actual_final_revenue = (forecast['yhat'] * product_price).sum()

    # Calculate profits
    optimal_order_profit = actual_final_revenue - optimal_order_final_cost

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['optimal_inventory_level'],
        mode='lines', name="Optimal Inventory Level", line=dict(color='orange', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=[point[0] for point in optimal_reorder_points],
        y=[point[1] for point in optimal_reorder_points],
        mode='markers+text', marker_symbol='star', marker_size=10, marker_color='orange',
        text=optimal_reorder_labels, textposition="top center",
        name="Optimal Reorder Points"
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['prophet_safety_stock'],
        mode='lines', name="Safety Stock Level", line=dict(color='red', dash='dot')
    ))

    fig.update_layout(
        title=f"Optimal Inventory Levels, Reorder Points, and Safety Stock Levels for {start_date_title} to {end_date_title}",
        xaxis_title="Date",
        yaxis_title="Inventory / Demand",
        legend=dict(x=1.02, y=1, bordercolor="Black", borderwidth=1),
        template="plotly_white",
        showlegend=True
    )

    fig.show()

    fig_cost = go.Figure()

    fig_cost.add_annotation(
    x=forecast['ds'].iloc[-1], y=optimal_order_final_cost,
    text=f"Optimal Profit: ${optimal_order_profit:,.2f}", showarrow=True, arrowhead=1, font=dict(color="orange")
    )

    # Optimal Order Cumulative Cost
    fig_cost.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['optimal_order_cost'],
        mode='lines', name="Optimal Strategy Cumulative Cost", line=dict(color='orange', dash='dash')
    ))

    # Actual Cumulative Revenue
    fig_cost.add_trace(go.Scatter(
        x=forecast['ds'], y=np.cumsum(forecast['yhat'] * product_price),
        mode='lines', name="Forecasted Cumulative Revenue", line=dict(color='green')
    ))

    fig_cost.update_layout(
        title=f"Forecasted Cumulative Cost and Revenue for {start_date_title} to {end_date_title}",
        xaxis_title="Date",
        yaxis_title="Cumulative Cost and Revenue",
        legend=dict(x=1.02, y=1, bordercolor="Black", borderwidth=1),
        template="plotly_white",
        showlegend=True
    )

    fig_cost.show()

    



SELECT 
    COUNT(DISTINCT [Order ID]) as total_orders,
    COUNT(DISTINCT Customer) as total_customers,
    SUM(Sales) as total_revenue,
    SUM(Profit) as total_profit,
    ROUND(SUM(Profit) / SUM(Sales) * 100, 2) as profit_margin_pct
FROM sales;

SELECT 
    [Year-Month],
    SUM(Sales) as monthly_sales,
    SUM(Profit) as monthly_profit,
    COUNT(DISTINCT [Order ID]) as orders
FROM sales
GROUP BY [Year-Month]
ORDER BY [Year-Month];

SELECT 
    Year,
    SUM(Sales) as yearly_sales,
    SUM(Profit) as yearly_profit,
    COUNT(DISTINCT Customer) as customers
FROM sales
GROUP BY Year
ORDER BY Year;

SELECT 
    Category,
    SUM(Sales) as category_sales,
    SUM(Profit) as category_profit,
    ROUND(SUM(Profit) / SUM(Sales) * 100, 2) as margin_pct
FROM sales
GROUP BY Category
ORDER BY category_sales DESC;

SELECT 
    Category,
    [Sub-Category],
    SUM(Sales) as sales,
    COUNT(DISTINCT [Order ID]) as orders
FROM sales
GROUP BY Category, [Sub-Category]
ORDER BY sales DESC;

SELECT 
    Region,
    SUM(Sales) as regional_sales,
    SUM(Profit) as regional_profit,
    COUNT(DISTINCT Customer) as customers
FROM sales
GROUP BY Region
ORDER BY regional_sales DESC;

SELECT 
    State,
    Region,
    SUM(Sales) as state_sales,
    SUM(Profit) as state_profit
FROM sales
GROUP BY State, Region
ORDER BY state_sales DESC
LIMIT 15;

SELECT 
    Product,
    Category,
    SUM(Sales) as product_sales,
    SUM(Quantity) as units_sold
FROM sales
GROUP BY Product, Category
ORDER BY product_sales DESC
LIMIT 10;

SELECT 
    Product,
    Category,
    SUM(Profit) as product_profit,
    SUM(Sales) as product_sales
FROM sales
GROUP BY Product, Category
ORDER BY product_profit ASC
LIMIT 10;

SELECT 
    Customer,
    SUM(Sales) as customer_revenue,
    COUNT(DISTINCT [Order ID]) as order_count,
    ROUND(AVG(Sales), 2) as avg_order_value
FROM sales
GROUP BY Customer
ORDER BY customer_revenue DESC
LIMIT 20;

SELECT 
    [Payment Mode],
    COUNT(DISTINCT [Order ID]) as orders,
    SUM(Sales) as total_sales,
    ROUND(AVG(Sales), 2) as avg_order_value
FROM sales
GROUP BY [Payment Mode]
ORDER BY total_sales DESC;

SELECT 
    [Shipping Mode],
    COUNT(DISTINCT [Order ID]) as orders,
    SUM(Sales) as total_sales,
    ROUND(SUM(Sales) * 100.0 / (SELECT SUM(Sales) FROM sales), 2) as pct_of_total
FROM sales
GROUP BY [Shipping Mode]
ORDER BY orders DESC;

SELECT 
    CASE 
        WHEN Discount = 0 THEN 'No Discount'
        WHEN Discount <= 0.1 THEN '1-10%'
        WHEN Discount <= 0.2 THEN '11-20%'
        WHEN Discount <= 0.3 THEN '21-30%'
        ELSE '30%+'
    END as discount_band,
    COUNT(DISTINCT [Order ID]) as orders,
    SUM(Sales) as total_sales,
    SUM(Profit) as total_profit,
    ROUND(SUM(Profit) / SUM(Sales) * 100, 2) as margin_pct
FROM sales
GROUP BY discount_band
ORDER BY discount_band;

SELECT 
    order_frequency,
    COUNT(*) as customer_count
FROM (
    SELECT 
        Customer,
        COUNT(DISTINCT [Order ID]) as order_frequency
    FROM sales
    GROUP BY Customer
)
GROUP BY order_frequency
ORDER BY order_frequency;

SELECT 
    Region,
    Category,
    SUM(Sales) as sales
FROM sales
GROUP BY Region, Category
ORDER BY Region, sales DESC;

SELECT 
    Year,
    Quarter,
    SUM(Sales) as quarterly_sales,
    SUM(Profit) as quarterly_profit
FROM sales
GROUP BY Year, Quarter
ORDER BY Year, Quarter;

SELECT 
    [Year-Month],
    ROUND(SUM(Profit) / SUM(Sales) * 100, 2) as margin_pct,
    SUM(Sales) as sales
FROM sales
GROUP BY [Year-Month]
ORDER BY [Year-Month];

SELECT 
    City,
    State,
    Region,
    SUM(Profit) as city_profit,
    SUM(Sales) as city_sales
FROM sales
GROUP BY City, State, Region
HAVING SUM(Profit) < 0
ORDER BY city_profit;

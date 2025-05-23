use bpx_api_client::{BpxClient, BACKPACK_API_BASE_URL};
use bpx_api_types::order::{OrderType, Side, ExecuteOrderPayload, Order};
use std::env;
use rust_decimal::prelude::*;
use rust_decimal_macros::dec;
use std::thread;
use std::time::Duration;
use chrono::Local;

#[tokio::main]
async fn main() {
    let base_url = env::var("BASE_URL").unwrap_or_else(|_| BACKPACK_API_BASE_URL.to_string());
    let secret = env::var("SECRET").expect("Missing SECRET environment variable");

    println!("=== 网格交易策略启动 ===");
    println!("时间: {}", Local::now().format("%Y-%m-%d %H:%M:%S"));

    let client = BpxClient::init(base_url, &secret, None)
        .expect("Failed to initialize Backpack API client");

    // 交易参数设置
    let symbol = "FARTCOIN_USDC_PERP";  // 期货合约交易对
    let leverage = 5;  // 杠杆倍数
    let grid_size = 0.5;  // 网格间距（百分比）
    let grid_count = 5;   // 网格数量
    let trade_amount = dec!(100); // 每个网格的交易金额（USDC）
    let mut active_orders: Vec<String> = Vec::new();  // 存储活跃订单
    let mut last_price: Option<f64> = None;

    // 设置杠杆
    println!("设置杠杆倍数: {}x", leverage);
    // 注意：由于API限制，暂时注释掉杠杆设置
    // match client.set_leverage(symbol, leverage).await {
    //     Ok(_) => println!("杠杆设置成功"),
    //     Err(e) => {
    //         eprintln!("设置杠杆失败: {:?}", e);
    //         return;
    //     }
    // }

    loop {
        let current_time = Local::now();
        println!("\n=== 检查时间: {} ===", current_time.format("%Y-%m-%d %H:%M:%S"));

        // 获取当前价格
        match client.get_ticker(symbol).await {
            Ok(ticker) => {
                let current_price = ticker.last_price.to_f64().unwrap();
                
                // 打印价格变化
                if let Some(last) = last_price {
                    let price_change = ((current_price - last) / last) * 100.0;
                    println!("价格变化: {:.2}% (从 {:.2} 到 {:.2})", 
                        price_change, last, current_price);
                }
                last_price = Some(current_price);

                // 取消所有现有订单
                for order_id in &active_orders {
                    println!("取消订单: {}", order_id);
                    if let Err(e) = client.cancel_order(symbol, Some(order_id.as_str()), None).await {
                        eprintln!("取消订单失败: {:?}", e);
                    }
                }
                active_orders.clear();

                // 计算网格价格
                let grid_prices = calculate_grid_prices(current_price, grid_size, grid_count);
                
                // 在网格价格处下单
                for (i, &price) in grid_prices.iter().enumerate() {
                    let quantity = trade_amount / Decimal::from_f64(price).unwrap();
                    
                    // 买单
                    let buy_order = ExecuteOrderPayload {
                        symbol: symbol.to_string(),
                        side: Side::Bid,
                        order_type: OrderType::Limit,
                        quantity: Some(quantity),
                        price: Some(Decimal::from_f64(price).unwrap()),
                        ..Default::default()
                    };

                    // 卖单
                    let sell_order = ExecuteOrderPayload {
                        symbol: symbol.to_string(),
                        side: Side::Ask,
                        order_type: OrderType::Limit,
                        quantity: Some(quantity),
                        price: Some(Decimal::from_f64(price * (1.0 + grid_size/100.0)).unwrap()),
                        ..Default::default()
                    };

                    println!("网格 {}: 买单价格 {:.2}, 卖单价格 {:.2}", 
                        i + 1, price, price * (1.0 + grid_size/100.0));

                    // 执行买单
                    match client.execute_order(buy_order).await {
                        Ok(order) => {
                            if let Order::Limit(limit_order) = order {
                                println!("买单已提交: ID={}", limit_order.id);
                                active_orders.push(limit_order.id);
                            }
                        },
                        Err(e) => eprintln!("买单失败: {:?}", e),
                    }

                    // 执行卖单
                    match client.execute_order(sell_order).await {
                        Ok(order) => {
                            if let Order::Limit(limit_order) = order {
                                println!("卖单已提交: ID={}", limit_order.id);
                                active_orders.push(limit_order.id);
                            }
                        },
                        Err(e) => eprintln!("卖单失败: {:?}", e),
                    }
                }

                // 打印当前订单状态
                println!("\n=== 当前订单状态 ===");
                println!("活跃订单数量: {}", active_orders.len());
                for order_id in &active_orders {
                    println!("订单ID: {}", order_id);
                }
            },
            Err(e) => {
                eprintln!("获取价格失败: {:?}", e);
                if e.to_string().to_lowercase().contains("invalid symbol") {
                    eprintln!("错误: 交易对 {} 可能不存在或格式不正确", symbol);
                    break;
                }
            }
        }

        // 等待一段时间再进行下一次检查
        println!("\n等待10秒后进行下一次检查...");
        thread::sleep(Duration::from_secs(10));
    }
}

// 计算网格价格
fn calculate_grid_prices(current_price: f64, grid_size: f64, grid_count: usize) -> Vec<f64> {
    let mut prices = Vec::with_capacity(grid_count);
    let grid_step = grid_size / 100.0;
    let half_grids = (grid_count as f64) / 2.0;
    
    for i in 0..grid_count {
        let offset = (i as f64 - half_grids) * grid_step;
        let price = current_price * (1.0 + offset);
        prices.push(price);
    }
    
    prices
} 
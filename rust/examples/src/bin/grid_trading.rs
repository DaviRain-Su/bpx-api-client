use bpx_api_client::{BpxClient, BACKPACK_API_BASE_URL};
use bpx_api_types::markets::KlineInterval;
use bpx_api_types::order::{ExecuteOrderPayload, Order, OrderType, Side};
use chrono::Local;
use rust_decimal::prelude::*;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use std::env;
use std::thread;
use std::time::Duration;

// 格式化价格到指定精度
fn format_price(price: f64, precision: u32) -> Decimal {
    let multiplier = 10.0_f64.powi(precision as i32);
    let rounded = (price * multiplier).round() / multiplier;
    Decimal::from_f64(rounded).unwrap()
}

// 格式化数量到指定精度
fn format_quantity(quantity: Decimal, precision: u32) -> Decimal {
    let scale = 10u32.pow(precision);
    (quantity * Decimal::from(scale)).round() / Decimal::from(scale)
}

// 计算K线振幅
fn calculate_amplitude(klines: &[f64]) -> (f64, f64) {
    let mut positive_amplitudes = Vec::new();
    let mut negative_amplitudes = Vec::new();

    for i in 0..klines.len() - 1 {
        let change = (klines[i + 1] - klines[i]) / klines[i];
        if change > 0.0 {
            positive_amplitudes.push(change);
        } else {
            negative_amplitudes.push(change.abs());
        }
    }

    let avg_positive = if !positive_amplitudes.is_empty() {
        positive_amplitudes.iter().sum::<f64>() / positive_amplitudes.len() as f64
    } else {
        0.0
    };

    let avg_negative = if !negative_amplitudes.is_empty() {
        negative_amplitudes.iter().sum::<f64>() / negative_amplitudes.len() as f64
    } else {
        0.0
    };

    (avg_positive, avg_negative)
}

#[tokio::main]
async fn main() {
    let base_url = env::var("BASE_URL").unwrap_or_else(|_| BACKPACK_API_BASE_URL.to_string());
    let secret = env::var("SECRET").expect("Missing SECRET environment variable");

    println!("=== 网格交易策略启动 ===");
    println!("时间: {}", Local::now().format("%Y-%m-%d %H:%M:%S"));

    let client = BpxClient::init(base_url, &secret, None).expect("Failed to initialize Backpack API client");

    // 交易参数设置
    let symbol = "FARTCOIN_USDC_PERP"; // 期货合约交易对
    let _leverage = 5; // 杠杆倍数（暂时未使用）
    let grid_count = 3; // 网格数量
    let trade_amount = dec!(100); // 每个网格的交易金额（USDC）
    let max_position = dec!(300); // 最大持仓量
    let max_drawdown = dec!(0.02); // 最大回撤 2%
    let mut active_orders: Vec<String> = Vec::new(); // 存储活跃订单
    let mut last_price: Option<f64> = None;
    let price_precision = 2; // 价格精度（小数位数）
    let quantity_precision = 1; // 数量精度（小数位数）

    // 持仓管理
    let mut long_position = Decimal::ZERO;
    let mut short_position = Decimal::ZERO;
    let mut buy_entry_prices: HashMap<String, Decimal> = HashMap::new();
    let mut sell_entry_prices: HashMap<String, Decimal> = HashMap::new();
    let mut buy_tp_orders: HashMap<String, Decimal> = HashMap::new();
    let mut sell_tp_orders: HashMap<String, Decimal> = HashMap::new();
    let mut max_equity = Decimal::ZERO;
    let mut initial_equity = None;

    loop {
        let current_time = Local::now();
        println!("\n=== 检查时间: {} ===", current_time.format("%Y-%m-%d %H:%M:%S"));

        // 获取K线数据
        let start_time = chrono::Utc::now()
            .checked_sub_signed(chrono::Duration::minutes(60)) // 获取5分钟的数据
            .unwrap()
            .timestamp();

        match client
            .get_k_lines(symbol, KlineInterval::OneMinute, Some(start_time), None)
            .await
        {
            Ok(klines) => {
                println!("获取到 {} 条K线数据", klines.len());

                // 提取收盘价
                let closes: Vec<f64> = klines
                    .iter()
                    .filter_map(|k| {
                        if let Some(close) = k.close.as_ref() {
                            if let Ok(price) = close.to_string().parse::<f64>() {
                                if price > 0.0 {
                                    return Some(price);
                                }
                            }
                        }
                        None
                    })
                    .collect();

                println!("有效收盘价数量: {}", closes.len());
                if closes.len() < 30 {
                    println!("数据不足，等待更多数据, 当前数据: {}", closes.len());
                    thread::sleep(Duration::from_secs(5));
                    continue;
                }

                // 计算振幅
                let (avg_positive, avg_negative) = calculate_amplitude(&closes);
                let current_price = closes.last().unwrap();

                // 打印价格变化
                if let Some(last) = last_price {
                    let price_change = ((current_price - last) / last) * 100.0;
                    println!(
                        "价格变化: {:.2}% (从 {:.2} 到 {:.2})",
                        price_change, last, current_price
                    );
                }
                last_price = Some(*current_price);

                // 更新最大权益
                let current_equity = long_position - short_position;
                if current_equity > max_equity {
                    max_equity = current_equity;
                }

                // 检查最大回撤
                if let Some(init_equity) = initial_equity {
                    if current_equity < init_equity * (Decimal::ONE - max_drawdown) {
                        println!("触发最大回撤保护，执行清仓");
                        // 清仓逻辑
                        if long_position > Decimal::ZERO {
                            let order = ExecuteOrderPayload {
                                symbol: symbol.to_string(),
                                side: Side::Ask,
                                order_type: OrderType::Market,
                                quantity: Some(long_position),
                                ..Default::default()
                            };
                            if let Err(e) = client.execute_order(order).await {
                                eprintln!("清仓失败: {:?}", e);
                            }
                        }
                        if short_position > Decimal::ZERO {
                            let order = ExecuteOrderPayload {
                                symbol: symbol.to_string(),
                                side: Side::Bid,
                                order_type: OrderType::Market,
                                quantity: Some(short_position),
                                ..Default::default()
                            };
                            if let Err(e) = client.execute_order(order).await {
                                eprintln!("清仓失败: {:?}", e);
                            }
                        }
                        return;
                    }
                } else {
                    initial_equity = Some(current_equity);
                }

                // 检查并更新止盈订单
                for (order_id, entry_price) in buy_entry_prices.clone() {
                    if !active_orders.contains(&order_id) {
                        // 订单已成交，更新持仓
                        let quantity = trade_amount / entry_price;
                        long_position += quantity;

                        // 设置止盈
                        let tp_price = entry_price * (Decimal::ONE + dec!(0.01)); // 1% 止盈
                        let tp_order = ExecuteOrderPayload {
                            symbol: symbol.to_string(),
                            side: Side::Ask,
                            order_type: OrderType::Limit,
                            quantity: Some(quantity),
                            price: Some(tp_price),
                            ..Default::default()
                        };
                        if let Ok(order) = client.execute_order(tp_order).await {
                            if let Order::Limit(limit_order) = order {
                                let order_id = limit_order.id.clone();
                                buy_tp_orders.insert(order_id.clone(), tp_price);
                                active_orders.push(order_id);
                            }
                        }
                    }
                }

                for (order_id, entry_price) in sell_entry_prices.clone() {
                    if !active_orders.contains(&order_id) {
                        // 订单已成交，更新持仓
                        let quantity = trade_amount / entry_price;
                        short_position += quantity;

                        // 设置止盈
                        let tp_price = entry_price * (Decimal::ONE - dec!(0.01)); // 1% 止盈
                        let tp_order = ExecuteOrderPayload {
                            symbol: symbol.to_string(),
                            side: Side::Bid,
                            order_type: OrderType::Limit,
                            quantity: Some(quantity),
                            price: Some(tp_price),
                            ..Default::default()
                        };
                        if let Ok(order) = client.execute_order(tp_order).await {
                            if let Order::Limit(limit_order) = order {
                                let order_id = limit_order.id.clone();
                                sell_tp_orders.insert(order_id.clone(), tp_price);
                                active_orders.push(order_id);
                            }
                        }
                    }
                }

                // 检查止盈订单成交情况
                for (order_id, _) in buy_tp_orders.clone() {
                    if !active_orders.contains(&order_id) {
                        // 止盈订单已成交，减少持仓
                        if let Some(quantity) = buy_entry_prices.get(&order_id).map(|price| trade_amount / price) {
                            long_position -= quantity;
                        }
                    }
                }

                for (order_id, _) in sell_tp_orders.clone() {
                    if !active_orders.contains(&order_id) {
                        // 止盈订单已成交，减少持仓
                        if let Some(quantity) = sell_entry_prices.get(&order_id).map(|price| trade_amount / price) {
                            short_position -= quantity;
                        }
                    }
                }

                // 取消所有现有订单
                for order_id in &active_orders {
                    println!("取消订单: {}", order_id);
                    match client.cancel_order(symbol, Some(order_id.as_str()), None).await {
                        Ok(_) => println!("订单取消成功: {}", order_id),
                        Err(e) => {
                            if e.to_string().contains("Order not found") {
                                println!("订单已不存在: {}", order_id);
                            } else {
                                eprintln!("取消订单失败: {:?}", e);
                            }
                        }
                    }
                }
                active_orders.clear();

                // 计算网格价格
                let buy_threshold = avg_negative * 0.75;
                let sell_threshold = avg_positive * 0.75;

                // 买单网格
                if long_position < max_position {
                    for i in 0..grid_count {
                        let price = current_price * (1.0 - buy_threshold - i as f64 * 0.25 * avg_negative);
                        let formatted_price = format_price(price, price_precision);
                        let quantity = format_quantity(trade_amount / formatted_price, quantity_precision);

                        let order = ExecuteOrderPayload {
                            symbol: symbol.to_string(),
                            side: Side::Bid,
                            order_type: OrderType::Limit,
                            quantity: Some(quantity),
                            price: Some(formatted_price),
                            ..Default::default()
                        };

                        match client.execute_order(order).await {
                            Ok(order) => {
                                if let Order::Limit(limit_order) = order {
                                    println!(
                                        "买单已提交: ID={}, 价格={}, 数量={}",
                                        limit_order.id, formatted_price, quantity
                                    );
                                    active_orders.push(limit_order.id.clone());
                                    buy_entry_prices.insert(limit_order.id, formatted_price);
                                }
                            }
                            Err(e) => eprintln!("买单失败: {:?}", e),
                        }
                    }
                }

                // 卖单网格
                if short_position < max_position {
                    for i in 0..grid_count {
                        let price = current_price * (1.0 + sell_threshold + i as f64 * 0.25 * avg_positive);
                        let formatted_price = format_price(price, price_precision);
                        let quantity = format_quantity(trade_amount / formatted_price, quantity_precision);

                        let order = ExecuteOrderPayload {
                            symbol: symbol.to_string(),
                            side: Side::Ask,
                            order_type: OrderType::Limit,
                            quantity: Some(quantity),
                            price: Some(formatted_price),
                            ..Default::default()
                        };

                        match client.execute_order(order).await {
                            Ok(order) => {
                                if let Order::Limit(limit_order) = order {
                                    println!(
                                        "卖单已提交: ID={}, 价格={}, 数量={}",
                                        limit_order.id, formatted_price, quantity
                                    );
                                    active_orders.push(limit_order.id.clone());
                                    sell_entry_prices.insert(limit_order.id, formatted_price);
                                }
                            }
                            Err(e) => eprintln!("卖单失败: {:?}", e),
                        }
                    }
                }

                // 打印当前状态
                println!("\n=== 当前状态 ===");
                println!("多头持仓: {}", long_position);
                println!("空头持仓: {}", short_position);
                println!("最大权益: {}", max_equity);
                println!("当前权益: {}", current_equity);
                println!("活跃订单数量: {}", active_orders.len());
                println!("平均正向振幅: {:.4}%", avg_positive * 100.0);
                println!("平均负向振幅: {:.4}%", avg_negative * 100.0);
            }
            Err(e) => {
                eprintln!("获取K线数据失败: {:?}", e);
                if e.to_string().to_lowercase().contains("invalid symbol") {
                    eprintln!("错误: 交易对 {} 可能不存在或格式不正确", symbol);
                    break;
                }
            }
        }

        // 等待一段时间再进行下一次检查
        println!("\n等待5秒后进行下一次检查...");
        let _ = tokio::time::sleep(Duration::from_secs(5));
    }
}

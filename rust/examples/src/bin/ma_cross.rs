use bpx_api_client::{BpxClient, BACKPACK_API_BASE_URL};
use bpx_api_types::markets::{KlineInterval};
use bpx_api_types::order::{OrderType, Side, ExecuteOrderPayload, Order};
use std::env;
use rust_decimal::prelude::*;
use rust_decimal_macros::dec;
use std::thread;
use std::time::Duration;
use chrono::Local;

// 计算简单移动平均线
fn calculate_sma(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() < period {
        return Vec::new();
    }

    let mut sma = Vec::with_capacity(prices.len() - period + 1);
    let mut sum = prices[..period].iter().sum::<f64>();
    sma.push(sum / period as f64);

    for i in period..prices.len() {
        sum = sum - prices[i - period] + prices[i];
        sma.push(sum / period as f64);
    }

    sma
}

// 生成交易信号
fn generate_signals(fast_sma: &[f64], slow_sma: &[f64]) -> Vec<i8> {
    if fast_sma.len() != slow_sma.len() {
        return Vec::new();
    }

    let mut signals = Vec::with_capacity(fast_sma.len());
    let mut prev_diff = fast_sma[0] - slow_sma[0];

    for i in 1..fast_sma.len() {
        let curr_diff = fast_sma[i] - slow_sma[i];
        let signal = if curr_diff > 0.0 && prev_diff <= 0.0 {
            1  // 买入信号
        } else if curr_diff < 0.0 && prev_diff >= 0.0 {
            -1 // 卖出信号
        } else {
            0  // 无信号
        };
        signals.push(signal);
        prev_diff = curr_diff;
    }

    signals
}

#[tokio::main]
async fn main() {
    let base_url = env::var("BASE_URL").unwrap_or_else(|_| BACKPACK_API_BASE_URL.to_string());
    let secret = env::var("SECRET").expect("Missing SECRET environment variable");

    println!("=== 移动平均线交叉策略期货交易机器人启动 ===");
    println!("时间: {}", Local::now().format("%Y-%m-%d %H:%M:%S"));
    println!("交易对: FARTCOIN-PERP");
    println!("初始资金: 100 USDC");

    let client = BpxClient::init(base_url, &secret, None)
        .expect("Failed to initialize Backpack API client");

    // 交易参数设置
    let symbol = "FARTCOIN_USD";  // 期货合约交易对
    let interval = KlineInterval::OneHour;  // 1小时K线
    let trade_amount = dec!(100); // 每次交易金额（USDC）
    let mut current_order_id: Option<String> = None;
    let mut last_price: Option<f64> = None;
    let mut total_trades = 0;
    let mut successful_trades = 0;
    let mut position_size = Decimal::ZERO;  // 当前持仓大小
    let mut position_side = None;  // 当前持仓方向

    loop {
        let current_time = Local::now();
        println!("\n=== 检查时间: {} ===", current_time.format("%Y-%m-%d %H:%M:%S"));

        // 计算开始时间（24小时前）
        let start_time = chrono::Utc::now()
            .checked_sub_signed(chrono::Duration::hours(24))
            .unwrap()
            .timestamp();

        // 获取K线数据
        match client.get_k_lines(symbol, interval, Some(start_time), None).await {
            Ok(klines) => {
                // 提取收盘价
                let closes: Vec<f64> = klines.iter()
                    .filter_map(|k| k.close.as_ref()
                        .map(|c| c.to_string().parse::<f64>().unwrap_or(0.0)))
                    .collect();

                if closes.is_empty() {
                    eprintln!("错误: 未获取到有效的价格数据");
                    continue;
                }

                // 计算快速和慢速均线
                let fast_period = 10;
                let slow_period = 20;
                let fast_sma = calculate_sma(&closes, fast_period);
                let slow_sma = calculate_sma(&closes, slow_period);

                if fast_sma.is_empty() || slow_sma.is_empty() {
                    eprintln!("错误: 数据不足以计算移动平均线");
                    continue;
                }

                // 生成交易信号
                let signals = generate_signals(&fast_sma, &slow_sma);
                
                // 获取最新信号
                if let Some(&latest_signal) = signals.last() {
                    let current_price = closes.last().unwrap();
                    
                    // 打印价格变化
                    if let Some(last) = last_price {
                        let price_change = ((current_price - last) / last) * 100.0;
                        println!("价格变化: {:.2}% (从 {:.2} 到 {:.2})", 
                            price_change, last, current_price);
                    }
                    last_price = Some(*current_price);

                    // 打印均线数据
                    if let (Some(&fast), Some(&slow)) = (fast_sma.last(), slow_sma.last()) {
                        println!("快速均线: {:.2}, 慢速均线: {:.2}, 差值: {:.2}%", 
                            fast, slow, ((fast - slow) / slow) * 100.0);
                    }

                    // 打印当前持仓信息
                    if position_size != Decimal::ZERO {
                        println!("当前持仓: {} {} (方向: {})", 
                            position_size.to_string(),
                            symbol,
                            if position_side == Some(Side::Bid) { "做多" } else { "做空" }
                        );
                    }

                    println!("当前价格: {:.2}, 信号: {}", current_price, 
                        match latest_signal {
                            1 => "做多",
                            -1 => "做空",
                            _ => "持有"
                        });

                    // 如果有未完成的订单，先取消
                    if let Some(order_id) = &current_order_id {
                        println!("正在取消之前的订单: {}", order_id);
                        if let Err(e) = client.cancel_order(symbol, Some(order_id.as_str()), None).await {
                            eprintln!("取消订单失败: {:?}", e);
                        } else {
                            println!("订单取消成功");
                        }
                        current_order_id = None;
                    }

                    // 根据信号执行交易
                    match latest_signal {
                        1 => { // 做多信号
                            if position_side != Some(Side::Bid) {
                                total_trades += 1;
                                let quantity = trade_amount / Decimal::from_f64(*current_price).unwrap();
                                println!("准备做多: 数量={:.4} {}, 价格={:.2} USDC", 
                                    quantity.to_f64().unwrap(), symbol, current_price);
                                
                                let order_payload = ExecuteOrderPayload {
                                    symbol: symbol.to_string(),
                                    side: Side::Bid,
                                    order_type: OrderType::Market,
                                    quantity: Some(quantity),
                                    ..Default::default()
                                };
                                
                                match client.execute_order(order_payload).await {
                                    Ok(order) => {
                                        println!("做多订单已提交: {:?}", order);
                                        successful_trades += 1;
                                        match order {
                                            Order::Market(market_order) => {
                                                let order_id = market_order.id.clone();
                                                current_order_id = Some(order_id);
                                                position_size = quantity;
                                                position_side = Some(Side::Bid);
                                                println!("订单ID: {}", market_order.id);
                                            },
                                            Order::Limit(limit_order) => {
                                                let order_id = limit_order.id.clone();
                                                current_order_id = Some(order_id);
                                                position_size = quantity;
                                                position_side = Some(Side::Bid);
                                                println!("订单ID: {}", limit_order.id);
                                            }
                                        }
                                    },
                                    Err(e) => {
                                        eprintln!("做多订单失败: {:?}", e);
                                        if e.to_string().to_lowercase().contains("invalid symbol") {
                                            eprintln!("错误: 交易对 {} 可能不存在或格式不正确", symbol);
                                            break;
                                        }
                                    },
                                }
                            } else {
                                println!("已经持有多头仓位，无需操作");
                            }
                        },
                        -1 => { // 做空信号
                            if position_side != Some(Side::Ask) {
                                total_trades += 1;
                                let quantity = trade_amount / Decimal::from_f64(*current_price).unwrap();
                                println!("准备做空: 数量={:.4} {}, 价格={:.2} USDC", 
                                    quantity.to_f64().unwrap(), symbol, current_price);
                                
                                let order_payload = ExecuteOrderPayload {
                                    symbol: symbol.to_string(),
                                    side: Side::Ask,
                                    order_type: OrderType::Market,
                                    quantity: Some(quantity),
                                    ..Default::default()
                                };
                                
                                match client.execute_order(order_payload).await {
                                    Ok(order) => {
                                        println!("做空订单已提交: {:?}", order);
                                        successful_trades += 1;
                                        match order {
                                            Order::Market(market_order) => {
                                                let order_id = market_order.id.clone();
                                                current_order_id = Some(order_id);
                                                position_size = quantity;
                                                position_side = Some(Side::Ask);
                                                println!("订单ID: {}", market_order.id);
                                            },
                                            Order::Limit(limit_order) => {
                                                let order_id = limit_order.id.clone();
                                                current_order_id = Some(order_id);
                                                position_size = quantity;
                                                position_side = Some(Side::Ask);
                                                println!("订单ID: {}", limit_order.id);
                                            }
                                        }
                                    },
                                    Err(e) => {
                                        eprintln!("做空订单失败: {:?}", e);
                                        if e.to_string().to_lowercase().contains("invalid symbol") {
                                            eprintln!("错误: 交易对 {} 可能不存在或格式不正确", symbol);
                                            break;
                                        }
                                    },
                                }
                            } else {
                                println!("已经持有空头仓位，无需操作");
                            }
                        },
                        _ => println!("无交易信号"),
                    }

                    // 打印交易统计
                    println!("\n=== 交易统计 ===");
                    println!("总交易次数: {}", total_trades);
                    println!("成功交易次数: {}", successful_trades);
                    println!("成功率: {:.2}%", 
                        if total_trades > 0 { (successful_trades as f64 / total_trades as f64) * 100.0 } else { 0.0 });
                }
            },
            Err(err) => {
                eprintln!("获取K线数据失败: {:?}", err);
                if err.to_string().to_lowercase().contains("invalid symbol") {
                    eprintln!("错误: 交易对 {} 可能不存在或格式不正确", symbol);
                    break;
                }
            },
        }

        // 等待一段时间再进行下一次检查
        println!("\n等待60秒后进行下一次检查...");
        thread::sleep(Duration::from_secs(60));
    }
} 
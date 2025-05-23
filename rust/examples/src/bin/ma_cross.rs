use bpx_api_client::{BpxClient, BACKPACK_API_BASE_URL};
use bpx_api_types::markets::{Kline, KlineInterval};
use std::env;
use std::collections::VecDeque;
use chrono::Utc;
use rust_decimal::prelude::*;

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

    let client = BpxClient::init(base_url, &secret, None)
        .expect("Failed to initialize Backpack API client");

    // 获取K线数据
    let symbol = "SOL_USDC";
    let interval = KlineInterval::OneHour;  // 1小时K线
    let limit = 100;      // 获取100根K线

    // 计算开始时间（24小时前）
    let start_time = chrono::Utc::now()
        .checked_sub_signed(chrono::Duration::hours(24))
        .unwrap()
        .timestamp();
    println!("start_time: {}", start_time);

    match client.get_k_lines(symbol, interval, Some(start_time), None).await {
        Ok(klines) => {
            println!("Raw klines response: {:?}", klines);
            
            // 提取收盘价
            let closes: Vec<f64> = klines.iter()
                .filter_map(|k| k.close.as_ref()
                    .map(|c| c.to_string().parse::<f64>().unwrap_or(0.0)))
                .collect();

            if closes.is_empty() {
                eprintln!("No valid price data found");
                return;
            }

            // 计算快速和慢速均线
            let fast_period = 10;
            let slow_period = 20;
            let fast_sma = calculate_sma(&closes, fast_period);
            let slow_sma = calculate_sma(&closes, slow_period);

            if fast_sma.is_empty() || slow_sma.is_empty() {
                eprintln!("Not enough data to calculate moving averages");
                return;
            }

            // 生成交易信号
            let signals = generate_signals(&fast_sma, &slow_sma);

            // 打印最近的信号
            println!("最近5个交易信号:");
            for i in (0..signals.len()).rev().take(5) {
                let signal = signals[i];
                let price = closes[i + slow_period - 1];
                let signal_str = match signal {
                    1 => "买入",
                    -1 => "卖出",
                    _ => "持有",
                };
                println!("价格: {:.2}, 信号: {}", price, signal_str);
            }

            // 计算策略收益
            let mut position = 0.0;  // 0表示空仓，1表示满仓
            let mut balance = 1000.0; // 初始资金
            let mut shares = 0.0;     // 持有数量

            for i in 0..signals.len() {
                let price = closes[i + slow_period - 1];
                match signals[i] {
                    1 => {  // 买入信号
                        if position == 0.0 {
                            shares = balance / price;
                            position = 1.0;
                            println!("买入: 价格={:.2}, 数量={:.4}", price, shares);
                        }
                    },
                    -1 => { // 卖出信号
                        if position == 1.0 {
                            balance = shares * price;
                            position = 0.0;
                            println!("卖出: 价格={:.2}, 余额={:.2}", price, balance);
                        }
                    },
                    _ => {}
                }
            }

            // 如果最后还持有仓位，按最新价格计算收益
            if position == 1.0 {
                let last_price = closes.last().unwrap();
                balance = shares * last_price;
            }

            println!("\n策略回测结果:");
            println!("初始资金: 1000.0");
            println!("最终资金: {:.2}", balance);
            println!("收益率: {:.2}%", (balance - 1000.0) / 10.0);
        },
        Err(err) => eprintln!("Error: {:?}", err),
    }
} 
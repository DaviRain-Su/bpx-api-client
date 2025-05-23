use bpx_api_client::{BpxClient, BACKPACK_API_BASE_URL};
use std::env;
use chrono::Local;
use serde_json::json;
use std::fs::File;
use std::io::Write;

#[tokio::main]
async fn main() {
    let base_url = env::var("BASE_URL").unwrap_or_else(|_| BACKPACK_API_BASE_URL.to_string());
    let secret = env::var("SECRET").expect("Missing SECRET environment variable");

    let current_time = Local::now();
    println!("=== Backpack Exchange 市场数据查询 ===");
    println!("时间: {}", current_time.format("%Y-%m-%d %H:%M:%S"));

    let client = BpxClient::init(base_url, &secret, None)
        .expect("Failed to initialize Backpack API client");

    // 用于存储JSON数据的结构
    let mut markets_data = Vec::new();
    let mut tickers_data = Vec::new();

    // 获取所有市场信息
    match client.get_markets().await {
        Ok(markets) => {
            println!("\n=== 支持的市场列表 ===");
            println!("总市场数量: {}", markets.len());
            
            // 按市场类型分组统计
            let mut spot_count = 0;
            let mut futures_count = 0;
            
            for market in &markets {
                if market.symbol.ends_with("-PERP") {
                    futures_count += 1;
                } else {
                    spot_count += 1;
                }
            }
            
            println!("现货市场数量: {}", spot_count);
            println!("期货市场数量: {}", futures_count);
            
            // 打印详细市场信息
            println!("\n=== 市场详细信息 ===");
            for market in &markets {
                println!("\n交易对: {}", market.symbol);
                println!("基础资产: {}", market.base_symbol);
                println!("计价资产: {}", market.quote_symbol);
                println!("市场类型: {}", if market.symbol.ends_with("-PERP") { "期货" } else { "现货" });
                println!("价格精度: {}", market.price_decimal_places());
                println!("数量精度: {}", market.quantity_decimal_places());

                // 添加到JSON数据
                markets_data.push(json!({
                    "symbol": market.symbol,
                    "base_symbol": market.base_symbol,
                    "quote_symbol": market.quote_symbol,
                    "market_type": if market.symbol.ends_with("-PERP") { "futures" } else { "spot" },
                    "price_precision": market.price_decimal_places(),
                    "quantity_precision": market.quantity_decimal_places()
                }));
            }
        },
        Err(e) => eprintln!("获取市场数据失败: {:?}", e),
    }

    // 获取所有交易对的实时行情
    match client.get_tickers().await {
        Ok(tickers) => {
            println!("\n=== 实时行情数据 ===");
            println!("总交易对数量: {}", tickers.len());
            
            // 打印每个交易对的最新价格和24小时变化
            for ticker in &tickers {
                println!("\n交易对: {}", ticker.symbol);
                println!("最新价格: {}", ticker.last_price);
                println!("开盘价格: {}", ticker.first_price);
                println!("价格变化: {}", ticker.price_change);
                println!("价格变化百分比: {:.2}%", ticker.price_change_percent);
                println!("24小时最高: {}", ticker.high);
                println!("24小时最低: {}", ticker.low);
                println!("24小时成交量: {}", ticker.volume);
                println!("24小时成交笔数: {}", ticker.trades);

                // 添加到JSON数据
                tickers_data.push(json!({
                    "symbol": ticker.symbol,
                    "last_price": ticker.last_price,
                    "first_price": ticker.first_price,
                    "price_change": ticker.price_change,
                    "price_change_percent": ticker.price_change_percent,
                    "high": ticker.high,
                    "low": ticker.low,
                    "volume": ticker.volume,
                    "trades": ticker.trades
                }));
            }
        },
        Err(e) => eprintln!("获取行情数据失败: {:?}", e),
    }

    // 创建完整的JSON数据结构
    let json_data = json!({
        "timestamp": current_time.format("%Y-%m-%d %H:%M:%S").to_string(),
        "markets": markets_data,
        "tickers": tickers_data
    });

    // 生成文件名（使用当前日期）
    let filename = format!("backpack_markets_{}.json", current_time.format("%Y%m%d"));
    
    // 写入JSON文件
    match File::create(&filename) {
        Ok(mut file) => {
            if let Err(e) = writeln!(file, "{}", serde_json::to_string_pretty(&json_data).unwrap()) {
                eprintln!("写入JSON文件失败: {:?}", e);
            } else {
                println!("\n数据已导出到文件: {}", filename);
            }
        },
        Err(e) => eprintln!("创建JSON文件失败: {:?}", e),
    }
}
use anyhow::{Context, Result};
use bpx_api_client::{BpxClient, BACKPACK_API_BASE_URL};
use bpx_api_types::markets::KlineInterval;
use bpx_api_types::order::{ExecuteOrderPayload, LimitOrder, Order, OrderType, Side};
use chrono::Local;
use rust_decimal::prelude::*;
use rust_decimal::RoundingStrategy;
use rust_decimal_macros::dec;
use std::collections::{HashMap, VecDeque};
use std::env;
use std::str::FromStr;
use std::time::Duration;
use tokio::time::sleep;

// 格式化价格到指定精度
fn format_price(price: f64, precision: u32) -> Option<Decimal> {
    if !price.is_finite() || price <= 0.0 {
        return None;
    }
    let multiplier = 10u64.checked_pow(precision).unwrap_or(1) as f64;
    let rounded = (price * multiplier).round() / multiplier;
    Decimal::from_f64(rounded)
}

// 格式化数量到指定精度
fn format_quantity(quantity: Decimal, precision: u32) -> Option<Decimal> {
    if quantity <= Decimal::ZERO {
        return None;
    }
    Some(quantity.round_dp_with_strategy(precision, RoundingStrategy::ToZero))
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

fn parse_decimal_str(value: &str) -> Option<Decimal> {
    Decimal::from_str(value).ok()
}

#[derive(Clone, Debug)]
struct StrategyParams {
    price_precision: u32,
    quantity_precision: u32,
    trade_amount: Decimal,
    max_position: Decimal,
    min_price_increment: Decimal,
    round_fee_rate: Decimal,
    profit_buffer: Decimal,
}

#[derive(Clone, Debug, Default)]
struct PendingLevels {
    total_quantity: Decimal,
    levels: HashMap<Decimal, Vec<LimitOrder>>,
}

impl PendingLevels {
    fn from_orders(orders: Vec<LimitOrder>, precision: u32) -> Self {
        let mut map: HashMap<Decimal, Vec<LimitOrder>> = HashMap::new();
        let mut total = Decimal::ZERO;
        for order in orders {
            let remaining = order.quantity - order.executed_quantity;
            if remaining <= Decimal::ZERO {
                continue;
            }
            total += remaining;
            let key = order.price.round_dp(precision);
            map.entry(key).or_default().push(order);
        }
        PendingLevels {
            total_quantity: total,
            levels: map,
        }
    }
}

#[derive(Clone, Debug)]
struct MarketSnapshot {
    avg_positive: f64,
    avg_negative: f64,
    current_price: f64,
    buy_levels: PendingLevels,
    sell_levels: PendingLevels,
    position_quantity: Decimal,
    position_equity: Decimal,
    entry_price: Option<Decimal>,
}

#[derive(Clone, Debug)]
struct OrderPlan {
    price: Decimal,
    quantity: Decimal,
}

#[derive(Clone, Debug, Default)]
struct StrategyPlan {
    bids: Vec<OrderPlan>,
    asks: Vec<OrderPlan>,
}

trait TradingStrategy {
    fn plan(&self, snapshot: &MarketSnapshot, params: &StrategyParams) -> StrategyPlan;
}

struct GridStrategy {
    grid_count: u32,
    bid_amp_factor: f64,
    ask_amp_factor: f64,
    step_multiplier: f64,
}

impl GridStrategy {
    fn new(grid_count: u32) -> Self {
        GridStrategy {
            grid_count,
            bid_amp_factor: 0.75,
            ask_amp_factor: 0.75,
            step_multiplier: 0.25,
        }
    }
}

impl TradingStrategy for GridStrategy {
    fn plan(&self, snapshot: &MarketSnapshot, params: &StrategyParams) -> StrategyPlan {
        let mut bids = Vec::new();
        let mut asks = Vec::new();

        let buy_threshold = snapshot.avg_negative * self.bid_amp_factor;
        let sell_threshold = snapshot.avg_positive * self.ask_amp_factor;

        let min_gap = params.min_price_increment;
        let entry_price = snapshot.entry_price.unwrap_or(Decimal::ZERO);
        let margin = params.round_fee_rate + params.profit_buffer;
        let min_sell_price = if snapshot.position_quantity > Decimal::ZERO && entry_price > Decimal::ZERO {
            Some(entry_price * (Decimal::ONE + margin))
        } else {
            None
        };

        let max_cover_price = if snapshot.position_quantity < Decimal::ZERO && entry_price > Decimal::ZERO {
            let factor = (Decimal::ONE - margin).max(Decimal::ZERO);
            Some(entry_price * factor)
        } else {
            None
        };

        let mut last_bid_price: Option<Decimal> = None;
        let mut last_ask_price: Option<Decimal> = None;

        for i in 0..self.grid_count {
            let bid_price = snapshot.current_price
                * (1.0 - buy_threshold - i as f64 * self.step_multiplier * snapshot.avg_negative);
            if let Some(price) = format_price(bid_price, params.price_precision) {
                if let Some(quantity) = format_quantity(params.trade_amount / price, params.quantity_precision) {
                    if let Some(limit) = max_cover_price {
                        if price > limit {
                            continue;
                        }
                    }
                    if let Some(prev) = last_bid_price {
                        if (prev - price).abs() < min_gap {
                            continue;
                        }
                    }
                    bids.push(OrderPlan { price, quantity });
                    last_bid_price = Some(price);
                }
            }

            let ask_price = snapshot.current_price
                * (1.0 + sell_threshold + i as f64 * self.step_multiplier * snapshot.avg_positive);
            if let Some(price) = format_price(ask_price, params.price_precision) {
                if let Some(quantity) = format_quantity(params.trade_amount / price, params.quantity_precision) {
                    if let Some(limit) = min_sell_price {
                        if price < limit {
                            continue;
                        }
                    }
                    if let Some(prev) = last_ask_price {
                        if (price - prev).abs() < min_gap {
                            continue;
                        }
                    }
                    asks.push(OrderPlan { price, quantity });
                    last_ask_price = Some(price);
                }
            }
        }

        if let (Some(best_bid), Some(best_ask)) = (bids.first(), asks.first()) {
            if best_ask.price - best_bid.price < min_gap {
                asks.retain(|o| o.price - best_bid.price >= min_gap);
            }
        }

        StrategyPlan { bids, asks }
    }
}

struct RiskManager {
    max_drawdown: Decimal,
    min_reference: Decimal,
    initial_equity: Option<Decimal>,
    max_equity: Decimal,
}

impl RiskManager {
    fn new(max_drawdown: Decimal, min_reference: Decimal) -> Self {
        RiskManager {
            max_drawdown,
            min_reference,
            initial_equity: None,
            max_equity: Decimal::ZERO,
        }
    }

    fn evaluate(&mut self, equity: Decimal) -> RiskDecision {
        if self.initial_equity.is_none() {
            if equity >= self.min_reference {
                self.initial_equity = Some(equity);
                self.max_equity = equity;
            }
            return RiskDecision { flatten: false };
        }

        if let Some(initial) = self.initial_equity {
            if equity > self.max_equity {
                self.max_equity = equity;
            }

            let reference = self.max_equity.max(initial);
            if reference > self.min_reference {
                let threshold = reference * (Decimal::ONE - self.max_drawdown);
                if equity < threshold {
                    return RiskDecision { flatten: true };
                }
            }
        }

        RiskDecision { flatten: false }
    }

    fn max_equity(&self) -> Decimal {
        self.max_equity.max(self.initial_equity.unwrap_or(Decimal::ZERO))
    }

    fn reset(&mut self) {
        self.initial_equity = None;
        self.max_equity = Decimal::ZERO;
    }
}

struct RiskDecision {
    flatten: bool,
}

struct OrderExecutor<'a> {
    client: &'a BpxClient,
    symbol: &'a str,
    params: &'a StrategyParams,
}

impl<'a> OrderExecutor<'a> {
    fn new(client: &'a BpxClient, symbol: &'a str, params: &'a StrategyParams) -> Self {
        OrderExecutor { client, symbol, params }
    }

    async fn flatten_position(&self, quantity: Decimal) -> Result<()> {
        if quantity == Decimal::ZERO {
            return Ok(());
        }

        let side = if quantity.is_sign_positive() {
            Side::Ask
        } else {
            Side::Bid
        };
        let order = ExecuteOrderPayload {
            symbol: self.symbol.to_string(),
            side,
            order_type: OrderType::Market,
            quantity: Some(quantity.abs()),
            ..Default::default()
        };

        self.client
            .execute_order(order)
            .await
            .with_context(|| "平仓失败".to_string())?;

        Ok(())
    }

    async fn cancel_orders(&self, orders: Vec<LimitOrder>) {
        for order in orders {
            if let Err(err) = self
                .client
                .cancel_order(self.symbol, Some(order.id.as_str()), None)
                .await
            {
                if !err.to_string().contains("Order not found") {
                    eprintln!("取消订单失败: {:?}", err);
                }
            }
        }
    }

    async fn reconcile(&self, snapshot: &MarketSnapshot, plan: &StrategyPlan) -> Result<(Decimal, Decimal)> {
        let mut buy_levels: HashMap<Decimal, VecDeque<LimitOrder>> = snapshot
            .buy_levels
            .levels
            .iter()
            .map(|(price, orders)| (price.clone(), orders.clone().into()))
            .collect();
        let mut sell_levels: HashMap<Decimal, VecDeque<LimitOrder>> = snapshot
            .sell_levels
            .levels
            .iter()
            .map(|(price, orders)| (price.clone(), orders.clone().into()))
            .collect();

        let mut pending_long = snapshot.buy_levels.total_quantity;
        let mut pending_short = snapshot.sell_levels.total_quantity;

        for order_plan in &plan.bids {
            let mut reuse = false;
            if let Some(queue) = buy_levels.get_mut(&order_plan.price) {
                if let Some(existing) = queue.pop_front() {
                    println!("保留买单: {} @ {}", existing.id, order_plan.price);
                    reuse = true;
                }
                if queue.is_empty() {
                    buy_levels.remove(&order_plan.price);
                }
            }

            if reuse {
                continue;
            }

            let estimated_long = snapshot.position_quantity.max(Decimal::ZERO) + pending_long;
            if estimated_long >= self.params.max_position {
                println!("已达到最大多头持仓限制，不再下额外买单");
                break;
            }

            let order = ExecuteOrderPayload {
                symbol: self.symbol.to_string(),
                side: Side::Bid,
                order_type: OrderType::Limit,
                quantity: Some(order_plan.quantity),
                price: Some(order_plan.price),
                ..Default::default()
            };

            match self.client.execute_order(order).await {
                Ok(Order::Limit(limit_order)) => {
                    println!(
                        "买单已提交: ID={}, 价格={}, 数量={}",
                        limit_order.id, order_plan.price, order_plan.quantity
                    );
                    pending_long += order_plan.quantity;
                }
                Ok(_) => {}
                Err(e) => eprintln!("买单失败: {:?}", e),
            }
        }

        for order_plan in &plan.asks {
            let mut reuse = false;
            if let Some(queue) = sell_levels.get_mut(&order_plan.price) {
                if let Some(existing) = queue.pop_front() {
                    println!("保留卖单: {} @ {}", existing.id, order_plan.price);
                    reuse = true;
                }
                if queue.is_empty() {
                    sell_levels.remove(&order_plan.price);
                }
            }

            if reuse {
                continue;
            }

            let estimated_short = snapshot.position_quantity.min(Decimal::ZERO).abs() + pending_short;
            if estimated_short >= self.params.max_position {
                println!("已达到最大空头持仓限制，不再下额外卖单");
                break;
            }

            let order = ExecuteOrderPayload {
                symbol: self.symbol.to_string(),
                side: Side::Ask,
                order_type: OrderType::Limit,
                quantity: Some(order_plan.quantity),
                price: Some(order_plan.price),
                ..Default::default()
            };

            match self.client.execute_order(order).await {
                Ok(Order::Limit(limit_order)) => {
                    println!(
                        "卖单已提交: ID={}, 价格={}, 数量={}",
                        limit_order.id, order_plan.price, order_plan.quantity
                    );
                    pending_short += order_plan.quantity;
                }
                Ok(_) => {}
                Err(e) => eprintln!("卖单失败: {:?}", e),
            }
        }

        let remaining_buy = buy_levels.into_values().flatten().collect::<Vec<LimitOrder>>();
        if !remaining_buy.is_empty() {
            println!("取消过期买单: {} 条", remaining_buy.len());
            self.cancel_orders(remaining_buy).await;
        }

        let remaining_sell = sell_levels.into_values().flatten().collect::<Vec<LimitOrder>>();
        if !remaining_sell.is_empty() {
            println!("取消过期卖单: {} 条", remaining_sell.len());
            self.cancel_orders(remaining_sell).await;
        }

        Ok((pending_long, pending_short))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let base_url = env::var("BASE_URL").unwrap_or_else(|_| BACKPACK_API_BASE_URL.to_string());
    let secret = env::var("SECRET").expect("Missing SECRET environment variable");

    println!("=== 网格交易策略启动 ===");
    println!("时间: {}", Local::now().format("%Y-%m-%d %H:%M:%S"));

    let client = BpxClient::init(base_url, &secret, None).expect("Failed to initialize Backpack API client");

    // 交易参数设置
    let symbol = "FARTCOIN_USDC_PERP"; // 期货合约交易对
    let _leverage = 5; // 杠杆倍数（暂时未使用）
    let grid_count: u32 = 3; // 网格数量
    let trade_amount = dec!(100); // 每个网格的交易金额（USDC）
    let max_position = dec!(300); // 最大持仓量
    let max_drawdown = dec!(0.02); // 最大回撤 2%
    let price_precision = 2; // 价格精度（小数位数）
    let quantity_precision = 1; // 数量精度（小数位数）

    let tick_size = Decimal::from_i128_with_scale(1, price_precision);
    let min_price_increment = tick_size * Decimal::from(2); // 至少保持两个最小价位差
    let round_fee_rate = dec!(0.001); // 预估建平仓合计手续费率（0.1%）
    let profit_buffer = dec!(0.0005); // 额外利润缓冲（0.05%）

    let params = StrategyParams {
        price_precision,
        quantity_precision,
        trade_amount,
        max_position,
        min_price_increment,
        round_fee_rate,
        profit_buffer,
    };
    let strategy = GridStrategy::new(grid_count);
    let min_reference = dec!(1); // 当权益达到 1 USDC 以上才启动回撤保护
    let mut risk_manager = RiskManager::new(max_drawdown, min_reference);
    let mut last_price: Option<f64> = None;

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
                    sleep(Duration::from_secs(5)).await;
                    continue;
                }

                // 计算振幅
                let (avg_positive, avg_negative) = calculate_amplitude(&closes);
                let current_price = *closes.last().unwrap();

                if let Some(last) = last_price {
                    let price_change = ((current_price - last) / last) * 100.0;
                    println!(
                        "价格变化: {:.2}% (从 {:.2} 到 {:.2})",
                        price_change, last, current_price
                    );
                }
                last_price = Some(current_price);

                let open_orders = match client.get_open_orders(Some(symbol)).await {
                    Ok(orders) => orders,
                    Err(e) => {
                        eprintln!("获取未完成订单失败: {:?}", e);
                        sleep(Duration::from_secs(5)).await;
                        continue;
                    }
                };

                let mut buy_orders_raw = Vec::new();
                let mut sell_orders_raw = Vec::new();
                for order in open_orders {
                    if let Order::Limit(limit_order) = order {
                        match limit_order.side {
                            Side::Bid => buy_orders_raw.push(limit_order),
                            Side::Ask => sell_orders_raw.push(limit_order),
                        }
                    }
                }

                let buy_levels = PendingLevels::from_orders(buy_orders_raw.clone(), params.price_precision);
                let sell_levels = PendingLevels::from_orders(sell_orders_raw.clone(), params.price_precision);

                let (position_quantity, position_equity, entry_price) = match client.get_open_future_positions().await {
                    Ok(positions) => positions
                        .into_iter()
                        .find(|pos| pos.symbol == symbol)
                        .map(|pos| {
                            let qty = parse_decimal_str(&pos.net_quantity).unwrap_or(Decimal::ZERO);
                            let realized = parse_decimal_str(&pos.pnl_realized).unwrap_or(Decimal::ZERO);
                            let unrealized = parse_decimal_str(&pos.pnl_unrealized).unwrap_or(Decimal::ZERO);
                            let entry = parse_decimal_str(&pos.entry_price).filter(|p| *p > Decimal::ZERO);
                            (qty, realized + unrealized, entry)
                        })
                        .unwrap_or((Decimal::ZERO, Decimal::ZERO, None)),
                    Err(e) => {
                        eprintln!("获取持仓信息失败: {:?}", e);
                        (Decimal::ZERO, Decimal::ZERO, None)
                    }
                };

                let snapshot = MarketSnapshot {
                    avg_positive,
                    avg_negative,
                    current_price,
                    buy_levels,
                    sell_levels,
                    position_quantity,
                    position_equity,
                    entry_price,
                };

                let risk_decision = risk_manager.evaluate(snapshot.position_equity);
                if risk_decision.flatten {
                    println!("触发最大回撤保护，执行清仓");
                    let executor = OrderExecutor::new(&client, symbol, &params);
                    if !buy_orders_raw.is_empty() {
                        executor.cancel_orders(buy_orders_raw).await;
                    }
                    if !sell_orders_raw.is_empty() {
                        executor.cancel_orders(sell_orders_raw).await;
                    }
                    executor.flatten_position(snapshot.position_quantity).await?;
                    risk_manager.reset();
                    println!("已执行清仓，进入冷却...");
                    sleep(Duration::from_secs(5)).await;
                    continue;
                }

                let plan = strategy.plan(&snapshot, &params);
                let executor = OrderExecutor::new(&client, symbol, &params);
                let (pending_long, pending_short) = executor.reconcile(&snapshot, &plan).await?;

                println!("\n=== 当前状态 ===");
                println!("净持仓: {}", snapshot.position_quantity);
                println!("在途买单合计数量: {}", pending_long);
                println!("在途卖单合计数量: {}", pending_short);
                println!("最大权益: {}", risk_manager.max_equity());
                println!("当前权益: {}", snapshot.position_equity);
                println!("平均正向振幅: {:.4}%", snapshot.avg_positive * 100.0);
                println!("平均负向振幅: {:.4}%", snapshot.avg_negative * 100.0);
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
        sleep(Duration::from_secs(5)).await;
    }
    Ok(())
}

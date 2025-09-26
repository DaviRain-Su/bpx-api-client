use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use bpx_api_client::{BpxClient, BACKPACK_API_BASE_URL};
use bpx_api_types::markets::{Kline, KlineInterval};
use bpx_api_types::order::{
    ExecuteOrderPayload, LimitOrder, MarketOrder, Order, OrderStatus, OrderType, SelfTradePrevention, Side, TimeInForce,
};
use chrono::Local;
use good_lp::solvers::microlp::microlp;
use good_lp::{constraint, variable, variables, Solution, SolverModel};
use nalgebra::DVector;
use rust_decimal::prelude::*;
use rust_decimal::RoundingStrategy;
use rust_decimal_macros::dec;
use std::collections::{HashMap, VecDeque};
use std::env;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

#[cfg(feature = "hyperliquid")]
use hyperliquid::{HyperliquidConfig, HyperliquidExchange};

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

fn decimal_from_env(key: &str, default: Decimal) -> Decimal {
    env::var(key)
        .ok()
        .and_then(|raw| Decimal::from_str(&raw).ok())
        .unwrap_or(default)
}

fn u32_from_env(key: &str, default: u32) -> u32 {
    env::var(key)
        .ok()
        .and_then(|raw| raw.parse::<u32>().ok())
        .unwrap_or(default)
}

#[cfg(feature = "hyperliquid")]
fn f64_from_env(key: &str, default: f64) -> f64 {
    env::var(key)
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
        .unwrap_or(default)
}

fn optimize_quantities(base_quantities: &[Decimal], weights: &[f64], capacity: Decimal) -> Vec<Decimal> {
    if base_quantities.is_empty() || weights.is_empty() || capacity <= Decimal::ZERO {
        return vec![Decimal::ZERO; base_quantities.len()];
    }

    let len = base_quantities.len().min(weights.len());
    let cap_f = capacity.to_f64().unwrap_or(0.0).max(0.0);
    if cap_f <= f64::EPSILON {
        return vec![Decimal::ZERO; len];
    }

    let weights_vec = DVector::from_iterator(len, weights.iter().cloned());
    if weights_vec.iter().all(|w| *w <= 0.0) {
        return vec![Decimal::ZERO; len];
    }

    let mut vars = variables!();
    let mut decision_vars = Vec::with_capacity(len);
    for &max_qty in base_quantities.iter().take(len) {
        let upper = max_qty.to_f64().unwrap_or(0.0).max(0.0);
        let var = vars.add(variable().min(0.0).max(upper));
        decision_vars.push(var);
    }

    let mut objective = 0.0 * decision_vars[0];
    for (coeff, var) in weights_vec.iter().zip(decision_vars.iter()) {
        objective = objective + (*coeff) * (*var);
    }

    let mut sum_expr = 0.0 * decision_vars[0];
    for var in &decision_vars {
        sum_expr = sum_expr + 1.0 * (*var);
    }

    let solution = vars
        .maximise(objective)
        .using(microlp)
        .with(constraint!(sum_expr <= cap_f))
        .solve();

    match solution {
        Ok(sol) => decision_vars
            .iter()
            .map(|var| Decimal::from_f64(sol.value(*var)).unwrap_or(Decimal::ZERO))
            .collect(),
        Err(_) => base_quantities.iter().take(len).cloned().collect(),
    }
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

#[derive(Clone, Debug, Default)]
struct PositionSnapshot {
    symbol: String,
    net_quantity: Decimal,
    pnl_realized: Decimal,
    pnl_unrealized: Decimal,
    entry_price: Option<Decimal>,
}

impl PositionSnapshot {
    fn total_pnl(&self) -> Decimal {
        self.pnl_realized + self.pnl_unrealized
    }
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
    exchange: Arc<dyn ExchangeClient>,
    symbol: &'a str,
    params: &'a StrategyParams,
}

impl<'a> OrderExecutor<'a> {
    fn new(exchange: Arc<dyn ExchangeClient>, symbol: &'a str, params: &'a StrategyParams) -> Self {
        OrderExecutor {
            exchange,
            symbol,
            params,
        }
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

        self.exchange
            .submit_order(order)
            .await
            .with_context(|| "平仓失败".to_string())?;

        Ok(())
    }

    async fn cancel_orders(&self, orders: Vec<LimitOrder>) {
        for order in orders {
            if let Err(err) = self.exchange.cancel_order(self.symbol, order.id.as_str()).await {
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

        let available_long = (self.params.max_position - snapshot.position_quantity.max(Decimal::ZERO) - pending_long)
            .max(Decimal::ZERO);
        let available_short =
            (self.params.max_position - snapshot.position_quantity.min(Decimal::ZERO).abs() - pending_short)
                .max(Decimal::ZERO);

        let bid_weights: Vec<f64> = plan
            .bids
            .iter()
            .map(|order| {
                let price = order.price.to_f64().unwrap_or(snapshot.current_price);
                if snapshot.current_price <= 0.0 {
                    0.0
                } else {
                    ((snapshot.current_price - price) / snapshot.current_price).max(0.0)
                }
            })
            .collect();
        let bid_base: Vec<Decimal> = plan.bids.iter().map(|o| o.quantity).collect();
        let optimized_bids = optimize_quantities(&bid_base, &bid_weights, available_long);

        let ask_weights: Vec<f64> = plan
            .asks
            .iter()
            .map(|order| {
                let price = order.price.to_f64().unwrap_or(snapshot.current_price);
                if snapshot.current_price <= 0.0 {
                    0.0
                } else {
                    ((price - snapshot.current_price) / snapshot.current_price).max(0.0)
                }
            })
            .collect();
        let ask_base: Vec<Decimal> = plan.asks.iter().map(|o| o.quantity).collect();
        let optimized_asks = optimize_quantities(&ask_base, &ask_weights, available_short);

        let tolerance = Decimal::from_f64(0.0001).unwrap_or(Decimal::ZERO);

        for (idx, order_plan) in plan.bids.iter().enumerate() {
            let target_quantity = optimized_bids.get(idx).copied().unwrap_or(Decimal::ZERO);
            if target_quantity <= Decimal::ZERO {
                continue;
            }

            let mut reuse = false;
            if let Some(queue) = buy_levels.get_mut(&order_plan.price) {
                if let Some(existing) = queue.front() {
                    let existing_qty = existing.quantity - existing.executed_quantity;
                    if (existing_qty - target_quantity).abs() <= tolerance {
                        let reused_order = queue.pop_front().unwrap();
                        println!("保留买单: {} @ {}", reused_order.id, order_plan.price);
                        reuse = true;
                    }
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
                quantity: Some(target_quantity),
                price: Some(order_plan.price),
                ..Default::default()
            };

            match self.exchange.submit_order(order).await {
                Ok(Order::Limit(limit_order)) => {
                    println!(
                        "买单已提交: ID={}, 价格={}, 数量={}",
                        limit_order.id, order_plan.price, target_quantity
                    );
                    pending_long += target_quantity;
                }
                Ok(_) => {}
                Err(e) => eprintln!("买单失败: {:?}", e),
            }
        }

        for (idx, order_plan) in plan.asks.iter().enumerate() {
            let target_quantity = optimized_asks.get(idx).copied().unwrap_or(Decimal::ZERO);
            if target_quantity <= Decimal::ZERO {
                continue;
            }

            let mut reuse = false;
            if let Some(queue) = sell_levels.get_mut(&order_plan.price) {
                if let Some(existing) = queue.front() {
                    let existing_qty = existing.quantity - existing.executed_quantity;
                    if (existing_qty - target_quantity).abs() <= tolerance {
                        let reused_order = queue.pop_front().unwrap();
                        println!("保留卖单: {} @ {}", reused_order.id, order_plan.price);
                        reuse = true;
                    }
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
                quantity: Some(target_quantity),
                price: Some(order_plan.price),
                ..Default::default()
            };

            match self.exchange.submit_order(order).await {
                Ok(Order::Limit(limit_order)) => {
                    println!(
                        "卖单已提交: ID={}, 价格={}, 数量={}",
                        limit_order.id, order_plan.price, target_quantity
                    );
                    pending_short += target_quantity;
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
    println!("=== 网格交易策略启动 ===");
    println!("时间: {}", Local::now().format("%Y-%m-%d %H:%M:%S"));

    // 解析策略与风险参数（允许通过环境变量覆盖默认值）
    let exchange_label = env::var("EXCHANGE")
        .unwrap_or_else(|_| "backpack".to_string())
        .to_lowercase();
    let grid_count = u32_from_env("GRID_COUNT", 3);
    let trade_amount = decimal_from_env("TRADE_AMOUNT_USD", dec!(100));
    let max_position = decimal_from_env("MAX_POSITION", dec!(300));
    let max_drawdown = decimal_from_env("MAX_DRAWDOWN", dec!(0.02));
    let price_precision = u32_from_env("PRICE_PRECISION", 2);
    let quantity_precision = u32_from_env("QUANTITY_PRECISION", 1);
    let loss_cut = decimal_from_env("LOSS_CUT_USD", dec!(1));
    let take_profit = decimal_from_env("TAKE_PROFIT_USD", dec!(2));
    let round_fee_rate = decimal_from_env("ROUND_FEE_RATE", dec!(0.001));
    let profit_buffer = decimal_from_env("PROFIT_BUFFER", dec!(0.0005));
    let min_reference = decimal_from_env("RISK_MIN_REFERENCE", dec!(1));
    #[cfg(feature = "hyperliquid")]
    let hyper_slippage = f64_from_env("HYPERLIQUID_MAX_SLIPPAGE", 0.05);

    let tick_size = Decimal::from_i128_with_scale(1, price_precision);
    let min_price_increment = tick_size * Decimal::from(2); // 保持至少两个最小价位差

    // 根据交易所类型构建对应客户端
    let (symbol, exchange): (String, Arc<dyn ExchangeClient>) = if exchange_label == "hyperliquid" {
        #[cfg(feature = "hyperliquid")]
        {
            let display_symbol = env::var("SYMBOL").unwrap_or_else(|_| "ETH_PERP".to_string());
            let default_coin = display_symbol.trim_end_matches("_PERP").to_string();
            let coin = env::var("HYPERLIQUID_COIN").unwrap_or(default_coin);
            let private_key =
                env::var("HYPERLIQUID_PRIVATE_KEY").context("缺少 Hyperliquid 私钥: 请设置 HYPERLIQUID_PRIVATE_KEY")?;
            let vault_address = env::var("HYPERLIQUID_VAULT_ADDRESS").ok();

            let base_env = env::var("HYPERLIQUID_ENV").unwrap_or_else(|_| "testnet".to_string());
            let base_url = match base_env.to_lowercase().as_str() {
                "mainnet" => hyperliquid_rust_sdk::BaseUrl::Mainnet,
                "local" | "localhost" => hyperliquid_rust_sdk::BaseUrl::Localhost,
                _ => hyperliquid_rust_sdk::BaseUrl::Testnet,
            };

            let config = HyperliquidConfig {
                base_url,
                coin: coin.clone(),
                display_symbol: display_symbol.clone(),
                private_key,
                vault_address,
                max_slippage: hyper_slippage,
            };
            let exchange = HyperliquidExchange::new(config).await?;
            (display_symbol, Arc::new(exchange) as Arc<dyn ExchangeClient>)
        }
        #[cfg(not(feature = "hyperliquid"))]
        {
            bail!("当前可执行文件未启用 `hyperliquid` 功能，请在 Cargo 特性中打开后重新构建");
        }
    } else {
        let base_url = env::var("BASE_URL").unwrap_or_else(|_| BACKPACK_API_BASE_URL.to_string());
        let secret = env::var("SECRET").context("缺少 Backpack SECRET 环境变量")?;
        let client = BpxClient::init(base_url, &secret, None).context("初始化 Backpack API 客户端失败")?;
        let symbol = env::var("SYMBOL").unwrap_or_else(|_| "FARTCOIN_USDC_PERP".to_string());
        (symbol, Arc::new(BpxExchange::new(client)) as Arc<dyn ExchangeClient>)
    };

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

        match exchange
            .fetch_k_lines(&symbol, KlineInterval::OneMinute, Some(start_time), None)
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

                let open_orders = match exchange.fetch_open_orders(&symbol).await {
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

                let (position_quantity, position_equity, entry_price) = match exchange.fetch_positions().await {
                    Ok(positions) => positions
                        .into_iter()
                        .find(|pos| pos.symbol == symbol)
                        .map(|pos| (pos.net_quantity, pos.total_pnl(), pos.entry_price))
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

                if snapshot.position_equity <= -loss_cut {
                    println!("亏损达到 {:.2} USDC，执行止损平仓", snapshot.position_equity);
                    let executor = OrderExecutor::new(exchange.clone(), &symbol, &params);
                    if !buy_orders_raw.is_empty() {
                        executor.cancel_orders(buy_orders_raw).await;
                    }
                    if !sell_orders_raw.is_empty() {
                        executor.cancel_orders(sell_orders_raw).await;
                    }
                    executor.flatten_position(snapshot.position_quantity).await?;
                    risk_manager.reset();
                    sleep(Duration::from_secs(3)).await;
                    continue;
                }

                if snapshot.position_equity >= take_profit {
                    println!("收益达到 {:.2} USDC，执行止盈平仓", snapshot.position_equity);
                    let executor = OrderExecutor::new(exchange.clone(), &symbol, &params);
                    if !buy_orders_raw.is_empty() {
                        executor.cancel_orders(buy_orders_raw).await;
                    }
                    if !sell_orders_raw.is_empty() {
                        executor.cancel_orders(sell_orders_raw).await;
                    }
                    executor.flatten_position(snapshot.position_quantity).await?;
                    risk_manager.reset();
                    sleep(Duration::from_secs(3)).await;
                    continue;
                }

                let risk_decision = risk_manager.evaluate(snapshot.position_equity);
                if risk_decision.flatten {
                    println!("触发最大回撤保护，执行清仓");
                    let executor = OrderExecutor::new(exchange.clone(), &symbol, &params);
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
                let executor = OrderExecutor::new(exchange.clone(), &symbol, &params);
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
#[async_trait]
trait ExchangeClient: Send + Sync {
    async fn fetch_k_lines(
        &self,
        symbol: &str,
        interval: KlineInterval,
        start: Option<i64>,
        end: Option<i64>,
    ) -> Result<Vec<Kline>>;

    async fn fetch_open_orders(&self, symbol: &str) -> Result<Vec<Order>>;

    async fn fetch_positions(&self) -> Result<Vec<PositionSnapshot>>;

    async fn submit_order(&self, payload: ExecuteOrderPayload) -> Result<Order>;

    async fn cancel_order(&self, symbol: &str, order_id: &str) -> Result<()>;
}

struct BpxExchange {
    client: BpxClient,
}

impl BpxExchange {
    fn new(client: BpxClient) -> Self {
        Self { client }
    }
}

#[async_trait]
impl ExchangeClient for BpxExchange {
    async fn fetch_k_lines(
        &self,
        symbol: &str,
        interval: KlineInterval,
        start: Option<i64>,
        end: Option<i64>,
    ) -> Result<Vec<Kline>> {
        self.client
            .get_k_lines(symbol, interval, start, end)
            .await
            .map_err(anyhow::Error::from)
    }

    async fn fetch_open_orders(&self, symbol: &str) -> Result<Vec<Order>> {
        self.client
            .get_open_orders(Some(symbol))
            .await
            .map_err(anyhow::Error::from)
    }

    async fn fetch_positions(&self) -> Result<Vec<PositionSnapshot>> {
        let raw = self
            .client
            .get_open_future_positions()
            .await
            .map_err(anyhow::Error::from)?;

        Ok(raw
            .into_iter()
            .map(|pos| PositionSnapshot {
                symbol: pos.symbol.clone(),
                net_quantity: parse_decimal_str(&pos.net_quantity).unwrap_or(Decimal::ZERO),
                pnl_realized: parse_decimal_str(&pos.pnl_realized).unwrap_or(Decimal::ZERO),
                pnl_unrealized: parse_decimal_str(&pos.pnl_unrealized).unwrap_or(Decimal::ZERO),
                entry_price: parse_decimal_str(&pos.entry_price).filter(|p| *p > Decimal::ZERO),
            })
            .collect())
    }

    async fn submit_order(&self, payload: ExecuteOrderPayload) -> Result<Order> {
        self.client.execute_order(payload).await.map_err(anyhow::Error::from)
    }

    async fn cancel_order(&self, symbol: &str, order_id: &str) -> Result<()> {
        self.client
            .cancel_order(symbol, Some(order_id), None)
            .await
            .map(|_| ())
            .map_err(anyhow::Error::from)
    }
}

#[cfg(feature = "hyperliquid")]
mod hyperliquid {
    use super::*;
    use alloy::{primitives::Address, signers::local::PrivateKeySigner};
    use chrono::Utc;
    use hyperliquid_rust_sdk::{
        AssetPosition, ClientCancelRequest, ClientLimit, ClientOrder, ClientOrderRequest,
        ExchangeClient as HyperExchangeClient, ExchangeDataStatus, ExchangeResponseStatus, InfoClient,
        MarketCloseParams,
    };
    use tokio::sync::Mutex;

    #[derive(Clone)]
    pub(super) struct HyperliquidConfig {
        pub base_url: hyperliquid_rust_sdk::BaseUrl,
        pub coin: String,
        pub display_symbol: String,
        pub private_key: String,
        pub vault_address: Option<String>,
        pub max_slippage: f64,
    }

    pub(super) struct HyperliquidExchange {
        info_client: InfoClient,
        exchange_client: HyperExchangeClient,
        address: Address,
        coin: String,
        display_symbol: String,
        baseline_account_value: Mutex<Option<Decimal>>,
        max_slippage: f64,
    }

    impl HyperliquidExchange {
        pub async fn new(config: HyperliquidConfig) -> Result<Self> {
            let wallet: PrivateKeySigner = config.private_key.parse().context("Hyperliquid 私钥格式不正确")?;
            let address = wallet.address();
            let vault_address = if let Some(vault) = config.vault_address {
                Some(vault.parse().context("Hyperliquid 金库地址格式不正确")?)
            } else {
                None
            };

            let info_client = InfoClient::new(None, Some(config.base_url)).await?;
            let exchange_client =
                HyperExchangeClient::new(None, wallet.clone(), Some(config.base_url), None, vault_address).await?;

            Ok(HyperliquidExchange {
                info_client,
                exchange_client,
                address,
                coin: config.coin,
                display_symbol: config.display_symbol,
                baseline_account_value: Mutex::new(None),
                max_slippage: config.max_slippage,
            })
        }

        fn interval_to_str(interval: KlineInterval) -> &'static str {
            match interval {
                KlineInterval::OneMinute => "1m",
                KlineInterval::ThreeMinutes => "3m",
                KlineInterval::FiveMinutes => "5m",
                KlineInterval::FifteenMinutes => "15m",
                KlineInterval::ThirtyMinutes => "30m",
                KlineInterval::OneHour => "1h",
                KlineInterval::TwoHours => "2h",
                KlineInterval::FourHours => "4h",
                KlineInterval::SixHours => "6h",
                KlineInterval::EightHours => "8h",
                KlineInterval::TwelveHours => "12h",
                KlineInterval::OneDay => "1d",
                KlineInterval::ThreeDays => "3d",
                KlineInterval::OneWeek => "1w",
                KlineInterval::OneMonth => "1M",
            }
        }

        fn parse_decimal(value: &str) -> Decimal {
            Decimal::from_str(value).unwrap_or(Decimal::ZERO)
        }

        fn is_buy_str(value: &str) -> bool {
            matches!(value.to_ascii_uppercase().as_str(), "B" | "BUY")
        }

        fn side_to_bool(side: Side) -> bool {
            matches!(side, Side::Bid)
        }

        async fn map_limit_response(
            &self,
            statuses: &[ExchangeDataStatus],
            side: Side,
            price: Decimal,
            quantity: Decimal,
        ) -> Result<LimitOrder> {
            let mut order_id: Option<u64> = None;
            let mut executed_quantity = Decimal::ZERO;
            let mut exec_price = price;
            let mut status = OrderStatus::New;

            for entry in statuses {
                match entry {
                    ExchangeDataStatus::Resting(resting) => {
                        order_id = Some(resting.oid);
                        status = OrderStatus::New;
                    }
                    ExchangeDataStatus::Filled(filled) => {
                        order_id = Some(filled.oid);
                        executed_quantity = Self::parse_decimal(&filled.total_sz);
                        exec_price = Self::parse_decimal(&filled.avg_px);
                        status = OrderStatus::Filled;
                    }
                    ExchangeDataStatus::Success
                    | ExchangeDataStatus::WaitingForFill
                    | ExchangeDataStatus::WaitingForTrigger => {}
                    ExchangeDataStatus::Error(err) => {
                        bail!("Hyperliquid 返回错误: {err}");
                    }
                }
            }

            if order_id.is_none() {
                let open_orders = self.info_client.open_orders(self.address).await?;
                order_id = open_orders
                    .into_iter()
                    .filter(|o| o.coin.eq_ignore_ascii_case(&self.coin))
                    .find(|o| {
                        let price_match = (Self::parse_decimal(&o.limit_px) - price).abs()
                            <= Decimal::from_str("0.0000001").unwrap_or(Decimal::ZERO);
                        price_match && (Self::is_buy_str(&o.side) == Self::side_to_bool(side))
                    })
                    .map(|o| o.oid);
            }

            let order_id = order_id.context("未能获取 Hyperliquid 订单 ID")?;
            let executed_quote_quantity = exec_price * executed_quantity;

            Ok(LimitOrder {
                id: order_id.to_string(),
                client_id: None,
                symbol: self.display_symbol.clone(),
                side,
                quantity,
                executed_quantity,
                executed_quote_quantity,
                price,
                trigger_price: None,
                time_in_force: TimeInForce::GTC,
                self_trade_prevention: SelfTradePrevention::RejectBoth,
                post_only: false,
                status,
                created_at: Utc::now().timestamp_millis(),
            })
        }

        fn map_market_response(
            &self,
            statuses: &[ExchangeDataStatus],
            side: Side,
            requested_quantity: Decimal,
        ) -> Result<MarketOrder> {
            let mut order_id: Option<u64> = None;
            let mut executed_quantity = Decimal::ZERO;
            let mut exec_price = Decimal::ZERO;

            for entry in statuses {
                match entry {
                    ExchangeDataStatus::Filled(filled) => {
                        order_id = Some(filled.oid);
                        executed_quantity = Self::parse_decimal(&filled.total_sz);
                        exec_price = Self::parse_decimal(&filled.avg_px);
                    }
                    ExchangeDataStatus::Success
                    | ExchangeDataStatus::WaitingForFill
                    | ExchangeDataStatus::WaitingForTrigger => {}
                    ExchangeDataStatus::Resting(resting) => {
                        order_id = Some(resting.oid);
                    }
                    ExchangeDataStatus::Error(err) => {
                        bail!("Hyperliquid 市价单返回错误: {err}");
                    }
                }
            }

            let order_id = order_id.context("未能获取 Hyperliquid 市价单 ID")?;
            let executed_quantity = if executed_quantity > Decimal::ZERO {
                executed_quantity
            } else {
                requested_quantity
            };
            let executed_quote_quantity = exec_price * executed_quantity;

            Ok(MarketOrder {
                id: order_id.to_string(),
                client_id: None,
                symbol: self.display_symbol.clone(),
                side,
                quantity: Some(requested_quantity),
                executed_quantity,
                quote_quantity: Some(exec_price * requested_quantity),
                executed_quote_quantity,
                trigger_price: None,
                time_in_force: TimeInForce::IOC,
                self_trade_prevention: SelfTradePrevention::RejectBoth,
                status: OrderStatus::Filled,
                created_at: Utc::now().timestamp_millis(),
            })
        }
    }

    #[async_trait]
    impl ExchangeClient for HyperliquidExchange {
        async fn fetch_k_lines(
            &self,
            _symbol: &str,
            interval: KlineInterval,
            start: Option<i64>,
            end: Option<i64>,
        ) -> Result<Vec<Kline>> {
            let end_ts = end.unwrap_or_else(|| chrono::Utc::now().timestamp());
            let start_ts = start.unwrap_or(end_ts.saturating_sub(3600));
            let start_ms = (start_ts.max(0) as u64) * 1000;
            let end_ms = (end_ts.max(0) as u64) * 1000;

            let candles = self
                .info_client
                .candles_snapshot(
                    self.coin.clone(),
                    Self::interval_to_str(interval).to_string(),
                    start_ms,
                    end_ms,
                )
                .await?;

            Ok(candles
                .into_iter()
                .map(|candle| Kline {
                    start: candle.time_open.to_string(),
                    open: Some(Self::parse_decimal(&candle.open)),
                    high: Some(Self::parse_decimal(&candle.high)),
                    low: Some(Self::parse_decimal(&candle.low)),
                    close: Some(Self::parse_decimal(&candle.close)),
                    end: Some(candle.time_close.to_string()),
                    volume: Self::parse_decimal(&candle.vlm),
                    trades: candle.num_trades.to_string(),
                })
                .collect())
        }

        async fn fetch_open_orders(&self, _symbol: &str) -> Result<Vec<Order>> {
            let orders = self.info_client.open_orders(self.address).await?;

            Ok(orders
                .into_iter()
                .filter(|order| order.coin.eq_ignore_ascii_case(&self.coin))
                .map(|order| {
                    let side = if Self::is_buy_str(&order.side) {
                        Side::Bid
                    } else {
                        Side::Ask
                    };
                    let price = Self::parse_decimal(&order.limit_px);
                    let quantity = Self::parse_decimal(&order.sz);
                    Order::Limit(LimitOrder {
                        id: order.oid.to_string(),
                        client_id: None,
                        symbol: self.display_symbol.clone(),
                        side,
                        quantity,
                        executed_quantity: Decimal::ZERO,
                        executed_quote_quantity: Decimal::ZERO,
                        price,
                        trigger_price: None,
                        time_in_force: TimeInForce::GTC,
                        self_trade_prevention: SelfTradePrevention::RejectBoth,
                        post_only: false,
                        status: OrderStatus::New,
                        created_at: order.timestamp as i64,
                    })
                })
                .collect())
        }

        async fn fetch_positions(&self) -> Result<Vec<PositionSnapshot>> {
            let state = self.info_client.user_state(self.address).await?;

            let account_value = Self::parse_decimal(&state.margin_summary.account_value);
            let mut baseline = self.baseline_account_value.lock().await;
            let baseline_value = if let Some(value) = *baseline {
                value
            } else {
                *baseline = Some(account_value);
                account_value
            };

            let mut matching_positions: Vec<(&AssetPosition, Decimal)> = Vec::new();
            let mut total_unrealized = Decimal::ZERO;
            for asset in &state.asset_positions {
                if asset.position.coin.eq_ignore_ascii_case(&self.coin) {
                    let unrealized = Self::parse_decimal(&asset.position.unrealized_pnl);
                    total_unrealized += unrealized;
                    matching_positions.push((asset, unrealized));
                }
            }
            drop(baseline);

            let realized_component = (account_value - baseline_value) - total_unrealized;

            if matching_positions.is_empty() {
                return Ok(vec![PositionSnapshot {
                    symbol: self.display_symbol.clone(),
                    net_quantity: Decimal::ZERO,
                    pnl_realized: realized_component,
                    pnl_unrealized: Decimal::ZERO,
                    entry_price: None,
                }]);
            }

            let mut snapshots: Vec<PositionSnapshot> = matching_positions
                .into_iter()
                .map(|(asset, unrealized)| PositionSnapshot {
                    symbol: self.display_symbol.clone(),
                    net_quantity: Self::parse_decimal(&asset.position.szi),
                    pnl_realized: Decimal::ZERO,
                    pnl_unrealized: unrealized,
                    entry_price: asset
                        .position
                        .entry_px
                        .as_ref()
                        .and_then(|px| Decimal::from_str(px).ok()),
                })
                .collect();

            if let Some(first) = snapshots.first_mut() {
                first.pnl_realized += realized_component;
            }

            Ok(snapshots)
        }

        async fn submit_order(&self, payload: ExecuteOrderPayload) -> Result<Order> {
            match payload.order_type {
                OrderType::Limit => {
                    let price = payload.price.context("Hyperliquid 限价单缺少价格")?;
                    let quantity = payload.quantity.context("Hyperliquid 限价单缺少数量")?;
                    let request = ClientOrderRequest {
                        asset: self.coin.clone(),
                        is_buy: Self::side_to_bool(payload.side),
                        reduce_only: false,
                        limit_px: price.to_f64().context("无法将价格转换为浮点数以提交给 Hyperliquid")?,
                        sz: quantity
                            .to_f64()
                            .context("无法将数量转换为浮点数以提交给 Hyperliquid")?,
                        cloid: None,
                        order_type: ClientOrder::Limit(ClientLimit { tif: "Gtc".to_string() }),
                    };

                    let response = self.exchange_client.order(request, None).await?;
                    if let ExchangeResponseStatus::Ok(resp) = response {
                        if let Some(data) = resp.data {
                            let order = self
                                .map_limit_response(&data.statuses, payload.side, price, quantity)
                                .await?;
                            Ok(Order::Limit(order))
                        } else {
                            bail!("Hyperliquid 限价单返回空数据");
                        }
                    } else if let ExchangeResponseStatus::Err(err) = response {
                        bail!("Hyperliquid 限价单被拒绝: {err}");
                    } else {
                        bail!("Hyperliquid 限价单返回未知响应");
                    }
                }
                OrderType::Market => {
                    let quantity = payload.quantity.context("Hyperliquid 市价单缺少数量")?;

                    let params = MarketCloseParams {
                        asset: &self.coin,
                        sz: Some(quantity.to_f64().context("无法将市价单数量转换为浮点数")?),
                        px: payload.price.and_then(|p| p.to_f64()),
                        slippage: Some(self.max_slippage),
                        cloid: None,
                        wallet: None,
                    };

                    let response = self.exchange_client.market_close(params).await?;
                    if let ExchangeResponseStatus::Ok(resp) = response {
                        if let Some(data) = resp.data {
                            let order = self.map_market_response(&data.statuses, payload.side, quantity)?;
                            Ok(Order::Market(order))
                        } else {
                            bail!("Hyperliquid 市价单返回空数据");
                        }
                    } else if let ExchangeResponseStatus::Err(err) = response {
                        bail!("Hyperliquid 市价单被拒绝: {err}");
                    } else {
                        bail!("Hyperliquid 市价单返回未知响应");
                    }
                }
            }
        }

        async fn cancel_order(&self, _symbol: &str, order_id: &str) -> Result<()> {
            let oid = order_id.parse::<u64>().context("无法解析 Hyperliquid 订单 ID")?;
            let response = self
                .exchange_client
                .cancel(
                    ClientCancelRequest {
                        asset: self.coin.clone(),
                        oid,
                    },
                    None,
                )
                .await?;

            match response {
                ExchangeResponseStatus::Ok(_) => Ok(()),
                ExchangeResponseStatus::Err(err) => {
                    bail!("Hyperliquid 撤单失败: {err}");
                }
            }
        }
    }
}

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};

use crate::Blockchain;

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Asset {
    symbol: String,
    tokens: Vec<Token>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Ticker {
    pub symbol: String,
    pub first_price: Decimal,
    pub last_price: Decimal,
    pub price_change: Decimal,
    pub price_change_percent: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub volume: Decimal,
    pub trades: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TickerUpdate {
    /// Event type
    #[serde(rename = "e")]
    pub event_type: String,

    /// Event timestamp in microseconds
    #[serde(rename = "E")]
    pub event_time: i64,

    /// Symbol
    #[serde(rename = "s")]
    pub symbol: String,

    #[serde(rename = "a")]
    pub ask_price: Decimal,

    #[serde(rename = "A")]
    pub ask_quantity: Decimal,

    #[serde(rename = "b")]
    pub bid_price: Decimal,

    #[serde(rename = "B")]
    pub bid_quantity: Decimal,

    /// Update ID of event
    #[serde(rename = "u")]
    pub update_id: u64,

    /// Engine timestamp in microseconds
    #[serde(rename = "T")]
    pub timestamp: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Market {
    pub symbol: String,
    pub base_symbol: String,
    pub quote_symbol: String,
    pub filters: MarketFilters,
}

impl Market {
    /// Returns the decimal places this market supports on the price.
    /// We error if a price with more decimal places is provided.
    /// `Price decimal too long`
    pub fn price_decimal_places(&self) -> u32 {
        self.filters.price.tick_size.scale()
    }

    /// Returns the decimal places this market supports on the quantity.
    /// if you provide a more precise quantity you will get an error
    /// `Quantity decimal too long`
    pub fn quantity_decimal_places(&self) -> u32 {
        self.filters.quantity.step_size.scale()
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MarketFilters {
    price: PriceFilters,
    quantity: QuantityFilters,
    leverage: Option<LeverageFilters>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PriceFilters {
    min_price: Decimal,
    max_price: Option<Decimal>,
    tick_size: Decimal,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuantityFilters {
    min_quantity: Decimal,
    max_quantity: Option<Decimal>,
    step_size: Decimal,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeverageFilters {
    min_leverage: Decimal,
    max_leverage: Decimal,
    step_size: Decimal,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Token {
    pub blockchain: Blockchain,
    pub deposit_enabled: bool,
    pub minimum_deposit: Decimal,
    pub withdraw_enabled: bool,
    pub minimum_withdrawal: Decimal,
    pub maximum_withdrawal: Option<Decimal>,
    pub withdrawal_fee: Decimal,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrderBookDepth {
    pub asks: Vec<(Decimal, Decimal)>,
    pub bids: Vec<(Decimal, Decimal)>,
    pub last_update_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrderBookDepthUpdate {
    /// Event type
    #[serde(rename = "e")]
    pub event_type: String,

    /// Event timestamp in microseconds
    #[serde(rename = "E")]
    pub event_time: i64,

    /// Symbol
    #[serde(rename = "s")]
    pub symbol: String,

    /// Engine timestamp in microseconds
    #[serde(rename = "T")]
    pub timestamp: i64,

    /// First update ID in event
    #[serde(rename = "U")]
    pub first_update_id: u64,

    /// Last update ID in event
    #[serde(rename = "u")]
    pub last_update_id: u64,

    /// Asks
    #[serde(rename = "a")]
    pub asks: Vec<(Decimal, Decimal)>,

    /// Bids
    #[serde(rename = "b")]
    pub bids: Vec<(Decimal, Decimal)>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Kline {
    pub start: String,
    pub open: Option<Decimal>,
    pub high: Option<Decimal>,
    pub low: Option<Decimal>,
    pub close: Option<Decimal>,
    pub end: Option<String>,
    pub volume: Decimal,
    pub trades: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FundingRate {
    pub symbol: String,
    pub interval_end_timestamp: String,
    pub funding_rate: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MarkPrice {
    pub symbol: String,
    pub funding_rate: Decimal,
    pub index_price: Decimal,
    pub mark_price: Decimal,
    pub next_funding_timestamp: u64,
}

#[derive(Debug, Display, Clone, Copy, Serialize, Deserialize, Default, EnumString, PartialEq, Eq, Hash)]
#[strum(serialize_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum KlineInterval {
    #[default]
    #[strum(serialize = "1m")]
    OneMinute,
    #[strum(serialize = "3m")]
    ThreeMinutes,
    #[strum(serialize = "5m")]
    FiveMinutes,
    #[strum(serialize = "15m")]
    FifteenMinutes,
    #[strum(serialize = "30m")]
    ThirtyMinutes,
    #[strum(serialize = "1h")]
    OneHour,
    #[strum(serialize = "2h")]
    TwoHours,
    #[strum(serialize = "4h")]
    FourHours,
    #[strum(serialize = "6h")]
    SixHours,
    #[strum(serialize = "8h")]
    EightHours,
    #[strum(serialize = "12h")]
    TwelveHours,
    #[strum(serialize = "1d")]
    OneDay,
    #[strum(serialize = "3d")]
    ThreeDays,
    #[strum(serialize = "1w")]
    OneWeek,
    #[strum(serialize = "1M")]
    OneMonth,
}

#[cfg(test)]
mod test {
    use rust_decimal_macros::dec;

    use crate::markets::{Market, QuantityFilters};

    use super::PriceFilters;

    fn get_test_market() -> Market {
        Market {
            symbol: "TEST_MARKET".to_string(),
            base_symbol: "TEST".to_string(),
            quote_symbol: "MARKET".to_string(),
            filters: super::MarketFilters {
                price: PriceFilters {
                    min_price: dec!(0.0001),
                    max_price: None,
                    tick_size: dec!(0.0001),
                },
                quantity: QuantityFilters {
                    min_quantity: dec!(0.01),
                    max_quantity: None,
                    step_size: dec!(0.01),
                },
                leverage: None,
            },
        }
    }

    #[test]
    fn test_decimal_places_on_price_filters_4() {
        let market = get_test_market();
        assert_eq!(market.price_decimal_places(), 4);
    }

    #[test]
    fn test_decimal_places_on_quantity_filters() {
        let market = get_test_market();
        assert_eq!(market.quantity_decimal_places(), 2);
    }
}

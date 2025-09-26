# 仓库指南

## 项目结构与模块组织
- `client/` 包含核心 crate；`src/lib.rs` 负责整合 `BpxClient`、`routes/` 以及可选的 `ws/` 支持。
- `types/` 存放在各 crate 之间复用的 Backpack 数据模型。
- `examples/` 在 `src/` 下提供 API 流程示例。
- 工作区工具位于根目录的 `Cargo.toml` 附近，构建产物输出到 `target/`。

## 构建、测试与开发命令
- `cargo build --all-targets` 或 `just build` 以调试模式编译全部 crate。
- `cargo test` 或 `just test` 运行工作区的单元与文档测试。
- `just check` 依次执行 `cargo check`、`cargo +nightly fmt --check`、`cargo clippy`。
- `just fix` 使用 nightly `rustfmt` 并应用可自动修复的 Clippy 建议。
- 启用 Hyperliquid 客户端时使用 `cargo run --features hyperliquid --bin grid_trading`，需提前配置相关环境变量。

## 代码风格与命名约定
- 遵循 Rust 2021 规则：四空格缩进、`snake_case` 项、`UpperCamelCase` 类型、`SCREAMING_SNAKE_CASE` 常量。
- 遵守 `rustfmt.toml`（120 列宽、导入排序），提交 PR 前运行 `cargo +nightly fmt --all`。
- 处理 `cargo clippy --all-targets --all-features` 的警告，非必要不要屏蔽。

## 测试指南
- 在模块内部使用 `#[cfg(test)]` 编写单元测试；端到端用例放在 `client/tests/`。
- 涉及 HTTP 或 WebSocket 时优先选用 `tokio::test` 等异步测试框架。
- 为新的 WebSocket 覆盖启用 `ws` 功能开关。

## 提交与 PR 指南
- 参考现有历史：主题行保持简短、使用祈使句（如“Add order client”），尽量控制在 50 个字符。
- 保持提交聚焦；规模较大的生成文件建议拆分提交。
- PR 需说明变更、列出执行的测试命令并关联 Backpack Exchange 任务或 Issue；如 API 行为变更请附截图或示例响应。

## 最新策略调整
- `EXCHANGE` 环境变量控制交易所选择：`backpack` 使用默认 `BpxExchange`，`hyperliquid` 需配合 `--features hyperliquid` 启动 Hyperliquid 客户端。
- `HyperliquidExchange` 基于官方 SDK 实现蜡烛、订单、仓位与市价平仓接口，需提供 `HYPERLIQUID_PRIVATE_KEY` 等环境变量完成签名。
- 引入 `PositionSnapshot` 结构，统一抽取净仓、已实现/未实现盈亏，使风控逻辑在多交易所间保持一致。
- 风控新增 `TAKE_PROFIT_USD`（默认 2 USDC）与 `LOSS_CUT_USD`（默认 1 USDC）双阈值：触发止盈/止损后先撤单再市价平仓，并重置高水位与冷却计时。
- `RISK_MIN_REFERENCE`（默认 1 USDC）用作高水位启动线，防止刚开盘的轻微波动触发回撤保护。
- 网格仍由 LP 线性规划分配数量，搭配最小价差、手续费缓冲与盈利余量，确保挂单价差覆盖成本。
- `LP_DEVIATION_PENALTY`（默认 0.25）为线性规划加入偏差惩罚，使仓位在多个网格价位间铺开而非集中在单点。
- 通过 `SYMBOLS`（逗号分隔）可同时运行多资产网格，策略会逐一拉取行情、执行 LP 优化并维护各自的风控状态；Hyperliquid 模式仍限制为单一交易对。
- 热身与循环阶段继续使用 `tokio::time::sleep` 节流，所有 I/O 均通过 `ExchangeClient` trait 统一调度，便于后续扩展新策略。

### 架构概览（ASCII）
```text
┌────────────────────────────┐
│ main (grid_trading.rs)     │
│ ├─ 初始化环境/客户端          │
│ ├─ StrategyParams & 风控配置 │
│ ├─ 创建 GridStrategy        │
│ ├─ 创建 RiskManager         │
│ ├─ 循环:                   │
│ │   ┌─ 拉取行情与订单快照     │
│ │   ├─ 组装 MarketSnapshot   │
│ │   ├─ RiskManager.evaluate │
│ │   ├─ strategy.plan        │
│ │   └─ OrderExecutor.reconcile│
│ └─ 结束: 平仓/退出           │
└────────────────────────────┘

┌────────────────────────────┐
│ TradingStrategy            │
│ ├─ trait plan(snapshot,    │
│ │        params) -> plan  │
│ └─ GridStrategy 实现       │
│    └─ 根据振幅生成网格       │
└────────────────────────────┘

┌────────────────────────────┐
│ MarketSnapshot             │
│ ├─ 振幅/现价                 │
│ ├─ buy_levels/sell_levels  │
│ └─ 持仓/权益                 │
└────────────────────────────┘

┌────────────────────────────┐
│ RiskManager                │
│ ├─ 维护初始权益/高水位         │
│ ├─ min_reference 阈值        │
│ └─ 判断是否需要清仓           │
└────────────────────────────┘

┌────────────────────────────┐
│ OrderExecutor              │
│ ├─ cancel_orders           │
│ ├─ flatten_position        │
│ └─ reconcile ->
│     ├─ 复用已有挂单           │
│     ├─ 下新单               │
│     └─ 撤销过期挂单           │
└────────────────────────────┘

┌────────────────────────────┐
│ ExchangeClient (trait)     │
│ ├─ fetch_k_lines           │
│ ├─ fetch_open_orders       │
│ ├─ fetch_position          │
│ ├─ place_limit/market      │
│ └─ cancel_order            │
└────────────────────────────┘

┌────────────────────────────┐
│ 实现:                       │
│ ├─ BpxExchange (默认)        │
│ └─ HyperliquidExchange      │
│     (启用 `hyperliquid`)     │
└────────────────────────────┘
```

### 架构说明
- 主循环负责数据采集与调度，依次调用风控、策略和执行模块；策略可通过实现 `TradingStrategy` trait 插拔更换。
- `MarketSnapshot` 把行情、挂单、持仓等信息整合成纯数据结构，既方便策略读取，也便于测试。
- `RiskManager` 在收到最新权益后先更新高水位，再根据指定最小参考值与最大回撤比率判断是否需要全局清仓。
- `OrderExecutor` 统一封装与交易所交互的逻辑，处理挂单复用、下单与撤单，策略层只需提供目标价格/数量。
- 当前策略为 `GridStrategy`，它根据振幅和参数生成对称网格计划；未来如需添加均线或动量策略，仅需实现新的 `plan` 方法并复用同一执行框架。
- `ExchangeClient` + `PositionSnapshot` 将交易所细节与风控解耦，Hyperliquid 适配器复用同一抽象即可读取/平掉仓位。

### 策略核心流程（伪代码）
```text
exchange = select_exchange(EXCHANGE)        # 根据环境变量选择交易所
symbols = parse_symbols()                   # 解析 SYMBOL / SYMBOLS
loop every 5 seconds:                       # 主循环：每 5 秒执行一次
    for symbol in symbols:                  # 逐一处理每个交易对
        fetch recent klines -> closes       # 获取最新 K 线并提取收盘价
        if insufficient data: continue      # 数据不足则跳过本轮

        (avg_up, avg_down) = amplitude(closes)  # 计算正负振幅均值
        open_orders = get_open_orders(symbol)   # 拉取当前未成交限价单
        pending_long, pending_short = aggregate(open_orders)
        position_qty, position_pnl = get_positions(symbol)  # 读取持仓/盈亏

        if position_pnl >= take_profit or drawdown_exceeded(position_pnl):
            cancel_all(open_orders)
            close_position(position_qty)
            reset_risk(symbol)
            continue

        target_bids = grid_prices(price, avg_down, count)
        target_asks = grid_prices(price, avg_up, count)

        optimized_qty = solve_lp(weight_vector, base_quantities, LP_DEVIATION_PENALTY)
        maintain_levels(target_bids, Side::Bid, pending_long, max_position)
        maintain_levels(target_asks, Side::Ask, pending_short, max_position)

        log(symbol, position_qty, pending_long, pending_short, position_pnl)
```

## 安全与配置提示
- 切勿硬编码真实 API 密钥；运行示例前通过环境变量注入。
- 使用内置的 `BACKPACK_API_BASE_URL` 与 `BACKPACK_WS_URL` 常量，避免混用不同环境。
- Hyperliquid 的私钥、金库地址等敏感数据必须通过环境变量传入，并避免输出在日志中。

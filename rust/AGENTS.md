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
- `examples/src/bin/grid_trading.rs` 已在热身逻辑与主循环中改用 `tokio::time::sleep`，避免阻塞 Tokio 运行时。
- 网格挂单逻辑通过 `get_open_orders` 读取实时订单，只复用价格匹配的层位、撤销过期层，并在估算净敞口与在途数量后遵守 `max_position`。
- 回撤保护依赖 `get_open_future_positions` 的盈亏数据；触发阈值时先撤单，再按仓位方向执行市价平仓。
- 价格与数量工具会过滤非有限值并向零截断，P&L 解析对异常字符串进行容错，防止运行时 panic。
- 新增 `StrategyParams`、`TradingStrategy`、`OrderExecutor`、`RiskManager` 等抽象组件，方便在同一执行框架下插入不同策略或拓展风控逻辑。
- 抽象出 `ExchangeClient` 接口，当前通过 `BpxExchange` 适配 Backpack，后续可接入其他交易所客户端而无需改动策略核心。
- 调整 `RiskManager` 的初始权益与高水位记录逻辑，仅在权益为正时建立基准，并基于高水位计算回撤阈值，避免启动阶段误触清仓。
- 追加最小权益阈值（默认 1 USDC），只有当累计盈亏达到该水平后才启用回撤保护，过滤掉初期的微小浮动噪声。
- 网格价格生成新增最小价差、手续费缓冲和收益阈值：避开重复价位、强制留出手续费空间，仅在满足 `entry_price*(1±margin)` 后才继续补单。
- 风控在触发清仓后会重置基准并进入短暂冷却，允许策略在风险解除后继续自动建仓。

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
```

### 架构说明
- 主循环负责数据采集与调度，依次调用风控、策略和执行模块；策略可通过实现 `TradingStrategy` trait 插拔更换。
- `MarketSnapshot` 把行情、挂单、持仓等信息整合成纯数据结构，既方便策略读取，也便于测试。
- `RiskManager` 在收到最新权益后先更新高水位，再根据指定最小参考值与最大回撤比率判断是否需要全局清仓。
- `OrderExecutor` 统一封装与交易所交互的逻辑，处理挂单复用、下单与撤单，策略层只需提供目标价格/数量。
- 当前策略为 `GridStrategy`，它根据振幅和参数生成对称网格计划；未来如需添加均线或动量策略，仅需实现新的 `plan` 方法并复用同一执行框架。

### 策略核心流程（伪代码）
```text
loop every 5 seconds:                       # 主循环：每 5 秒执行一次
    fetch recent klines -> closes           # 获取最新 K 线并提取收盘价
    if insufficient data: sleep and continue  # 数据不足则等待再试

    (avg_up, avg_down) = amplitude(closes)  # 计算正负振幅均值
    open_orders = get_open_orders(symbol)   # 拉取当前未成交限价单
    pending_long, pending_short = aggregate(open_orders)  # 汇总在途买卖数量
    position_qty, position_pnl = get_open_future_positions(symbol)  # 读取期货持仓与盈亏

    if drawdown_exceeded(position_pnl):       # 若达到最大回撤
        cancel_all(open_orders)               # 撤销剩余挂单
        close_position(position_qty)          # 市价平掉仓位
        risk_manager.reset()                  # 重置基准并冷却
        sleep(cooldown)
        continue                              # 继续监控

    target_bids = grid_prices(price, avg_down, count)       # 依据振幅计算买入网格价
    target_asks = grid_prices(price, avg_up, count)         # 依据振幅计算卖出网格价

    maintain_levels(target_bids, Side::Bid, pending_long, max_position)     # 维护买单层位
    maintain_levels(target_asks, Side::Ask, pending_short, max_position)    # 维护卖单层位

    log(position_qty, pending_long, pending_short, position_pnl)            # 打印状态
```

## 安全与配置提示
- 切勿硬编码真实 API 密钥；运行示例前通过环境变量注入。
- 使用内置的 `BACKPACK_API_BASE_URL` 与 `BACKPACK_WS_URL` 常量，避免混用不同环境。

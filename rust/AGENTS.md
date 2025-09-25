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
        break                                 # 退出循环

    target_bids = grid_prices(price, avg_down, count)       # 依据振幅计算买入网格价
    target_asks = grid_prices(price, avg_up, count)         # 依据振幅计算卖出网格价

    maintain_levels(target_bids, Side::Bid, pending_long, max_position)     # 维护买单层位
    maintain_levels(target_asks, Side::Ask, pending_short, max_position)    # 维护卖单层位

    log(position_qty, pending_long, pending_short, position_pnl)            # 打印状态
```

## 安全与配置提示
- 切勿硬编码真实 API 密钥；运行示例前通过环境变量注入。
- 使用内置的 `BACKPACK_API_BASE_URL` 与 `BACKPACK_WS_URL` 常量，避免混用不同环境。

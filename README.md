# 2026 世界杯赛果预测

用 React 浏览赛程、比分和积分榜，并根据球队 Elo 和 Poisson 进球分布估算主胜、平局与客胜概率。网页启动后会尝试从 ESPN 更新未标记为完赛的比分，内置数据可用于离线浏览。

## 本地运行

```bash
npm ci
npm start
```

浏览器打开 [localhost:3000](http://localhost:3000)。执行 `npm test -- --watchAll=false` 运行测试，执行 `npm run build` 生成静态站点。

## 预测与回测

- [`src/data.js`](src/data.js) 保存内置赛程、赛果和 Elo。
- [`src/App.js`](src/App.js) 计算进球分布和胜平负概率，并渲染赛程与自定义预测。
- [`evaluate_and_improve.py`](evaluate_and_improve.py) 是单独的 Python 评估入口。前端直接使用 JavaScript 计算预测，运行网页无需 Python。

赛程页的「回测准确率」以已结束的内置小组赛为样本，用当前 Elo 和预测逻辑重新计算胜平负，再与已记录的结果比较。页面同时显示命中场次和样本量。要评估赛前表现，需要使用开赛前保存的预测和评级快照。

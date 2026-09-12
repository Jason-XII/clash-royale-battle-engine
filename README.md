# Clash Royale Battle Engine

这个仓库允许你把皇室战争的原生战斗引擎当成 API 来调用，极大方便了 AI 训练。同时，为了方便你自己配置这个仓库，我在开发过程中把 agent 总结的经验蒸馏成了 3 个 skill，放在 `.agents/skills` 目录下。

本仓库是对 [FirstLight-CR](https://gitlab.com/firstlight3/FirstLight_CR) 逆向工作加以改进后的产物，极大简化了项目结构和通信接口，把代码从上万行压缩到几百行。如果你也希望训练 AI 来玩皇室战争，这个项目提供的干净接口非常适合作为你的训练起点。

完成项目配置之后，最后运行：

```bash
python bootstrap_emulator.py
```

测试战斗引擎的速度（下面的数据是在我的笔记本电脑运行的结果，只运行1个战斗所以速度较慢）：
```
python -m clash-royale-battle-engine.benchmark

result=1 crowns=(0, 1) final_tick=3835
attempts=62 executions=62 wire_bytes=475842
reset=0.035s rollout=0.376s total=0.411s
simulated=187.2s speed=497.9x real-time
ticks_per_second=9957.5
```
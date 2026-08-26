---
name: minimax-h3-orchestrator
description: 操作 MiniMax-H3 orchestrator（Kaggle GPU 池 + Cloudflare 隧道跑 ComfyUI 视频生成）。当需要提交 FL2VA / R2V / Multishot 视频任务、重启 controller、重启 Kaggle kernel、调试 worker 注册、或排查生成失败/超时时使用。包含 T4 实测速度配方和所有踩过的坑。
---

# MiniMax-H3 Orchestrator 操作手册

这是一个用免费 Kaggle GPU 池 + Cloudflare 隧道跑 MiniMax-H3 视频生成的调度系统。controller 是本机 FastAPI 服务，worker 是 Kaggle notebook 里的 ComfyUI 实例，通过 Cloudflare 隧道回连。

## 架构速览

```
Client → controller (本机 :8001, FastAPI) → Kaggle notebook × 2 worker (ComfyUI)
                ↑ Cloudflare 隧道回连        （每 notebook 2 张 T4 = 2 worker）
```

- **controller**：本机 `uvicorn controller.main:app --port 8001`
- **worker**：Kaggle notebook 启动时 clone 本 repo，起 2 个 ComfyUI（gpu0/gpu1），开隧道回注册
- **公网入口**：`https://controller.jayapp.cn` → cloudflared → `localhost:8001`

## 快速开始

```bash
cd ~/minimax-h3-orchestrator
source .venv/bin/activate

# 启动 controller（注意是 8001，不是 8000）
nohup uvicorn controller.main:app --host 0.0.0.0 --port 8001 > /tmp/minimax-controller.log 2>&1 & disown

# 验证
curl -sS http://127.0.0.1:8001/api/v1/system/status
```

## 三种 workflow

### 1. FL2VA（默认，图生视频单镜头）

```bash
curl -sS -X POST http://127.0.0.1:8001/api/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"...", "duration":5.0, "first_frame":"img.png"}'
```

### 2. R2V + turbo（带参考图锁角色的单镜头）

```bash
curl -sS -X POST http://127.0.0.1:8001/api/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{"workflow":"minimax-h3-r2v", "prompt":"...", "duration":5.0,
       "width":832, "height":480, "turbo":true, "ref_images":["char.png"]}'
```

### 3. Multishot（多镜头无缝链式，可带参考图）★ 重点

```bash
curl -sS -X POST http://127.0.0.1:8001/api/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{"workflow":"minimax-h3-multishot",
       "script":"shot1 prompt\n---\nshot2 prompt",
       "shot_count":2, "frames_per_shot":124,
       "width":832, "height":480,
       "start_image":"first.png",          # 可选：首帧种第一镜
       "reference_images":["char.png"]}     # 可选：角色参考图贯穿全链
```

- `script`：每镜头一个 prompt，用 `---` 单独一行分隔。**必须遵守 Multishot 官方边界规则**（见下方"脚本规则"）
- `shot_count`：总镜头数，0 = 按 `---` 块数
- `frames_per_shot`：124≈5.1s / 243≈10.1s / 362≈15.1s（17k+5 网格）
- 参考图需先放到 `storage/uploads/`

## T4 实测速度配方（★ 必读，否则会超时）

**最优解：int8 ref2va + turbo LoRA 4步 + 832×480**

| 配置 | 2镜头链式耗时 |
|------|--------------|
| 20步无turbo | >60min 超时 ❌ |
| 8步+turbo | ~60min 超时 ❌ |
| **4步+turbo** | **~30min** ✅ |
| 4步+turbo+参考图 | ~49min（参考token贯穿每步，慢1.6x）|

**两个加速实验都否决了，别再试：**
- `--enable-triton-backend`：T4 是 Turing 老架构，triton 优化 Ampere/Hopper，加了反而更慢（39min）
- GGUF Q4_0：4bit 精度让 turbo 4步不收敛，48min 比 int8 还慢

## 脚本规则（Multishot 接缝不崩的关键）

写 multishot 脚本时，每个镜头 prompt 必须遵守：

1. **airlock**：第2镜起，开头先承接上一镜的定格画面 + 静默约2秒（这段会被丢弃）
2. **给"静止"加点动作**：纯静止渲染成 freeze，写个呼吸/重心转移/视线变化
3. **落定（land settled）**：每镜结尾回到稳定画面，台词说完留2秒
4. **台词不跨镜**：一句台词要么整句放一个镜头，要么移到下个镜头。**124帧（5秒）装不下台词**，必须无对白
5. **描述逐字重复**：角色外观 + 房间/光线的描述在每个镜头 verbatim 重复，不许改写
6. **每镜有物理变化**：动作要不可逆（撕纸>皱眉），否则镜头3变成镜头2的复制

详细规则见 [PROMPTING.md](https://huggingface.co/joeygambino/MiniMax-H3-Multishot-Workflow/blob/main/PROMPTING.md)

## 运维要点（★ 踩过的坑）

### controller 必须重启才加载新代码

改完代码 + push 后，**controller 进程不会自动重载**。必须：

```bash
cd ~/minimax-h3-orchestrator
source .venv/bin/activate
kill $(pgrep -f 'uvicorn controller.main' | head -1) 2>/dev/null
sleep 2
nohup uvicorn controller.main:app --host 0.0.0.0 --port 8001 > /tmp/minimax-controller.log 2>&1 & disown
```

### worker 不热更新（★ 最重要的坑）

Kaggle worker 在 **notebook 启动时 clone 一次 repo**，之后 push 的代码变更**不会生效**。改 worker 侧代码（`worker/` 目录）必须重启 Kaggle kernel：

```bash
cd ~/minimax-h3-orchestrator
source .venv/bin/activate
python - <<'EOF'
from controller import db
from controller.constants import NotebookStatus, WorkerStatus
from controller.kaggle_manager import KaggleManager
from controller.accounts import AccountManager
from controller.notebook_builder import build_notebook
from controller.config import settings

store = db.Store()
am = AccountManager(store); am.sync_from_config()
ac = am.credential('kaggle-account-1')
mgr = KaggleManager(ac)

# 1. 重置 DB 状态（否则 scheduler 不重新 provision）
with store.session() as s:
    for j in s.query(db.Job).filter(db.Job.status.in_(["QUEUED","RUNNING","STARTING"])).all():
        j.status = "CANCELLED"
    nb = s.query(db.Notebook).filter(db.Notebook.id=='nb-kaggle-account-1').first()
    if nb:
        nb.status = NotebookStatus.NOTEBOOK_STOPPED.value
        nb.kaggle_kernel_slug = None
    for w in s.query(db.Worker).all():
        w.status = WorkerStatus.UNREGISTERED.value
        w.tunnel_url = None
        w.current_job_id = None

# 2. push 新 notebook（触发 Kaggle 新会话，重新 clone + 重下模型）
nb = build_notebook(
    notebook_id='nb-kaggle-account-1',
    controller_public_url=settings.controller_public_url,
    worker_auth_secret=settings.worker_auth_secret,
    repo_url=settings.orchestrator_repo_url,
    template_path=settings.notebook_path,
)
ok = mgr.ensure_notebook('wxujie/nb-kaggle-account-1', nb)
print('push:', ok)
EOF
```

重启后 worker 要重新下载全部模型（20-40 分钟），期间 `ready_workers=0` 是正常的。

### 端口

- controller 跑 **8001**（8000 被无关的 node 进程占用）
- cloudflared 把 `controller.jayapp.cn → localhost:8001`

### 检查状态

```bash
curl -sS http://127.0.0.1:8001/api/v1/system/status   # ready_workers / queued_jobs
curl -sS http://127.0.0.1:8001/api/v1/workers          # worker 状态
curl -sS http://127.0.0.1:8001/api/v1/jobs             # 任务列表（含 input 参数）
```

### 查询单个任务（GET /jobs/{id} 不含 input，用列表端点）

```bash
curl -sS http://127.0.0.1:8001/api/v1/jobs | python3 -c "
import sys,json
for j in json.load(sys.stdin):
    if j['job_id']=='JOB_ID':
        print(json.dumps(j, ensure_ascii=False, indent=2))
"
```

### 清理僵尸 BUSY worker

任务取消后 worker 可能卡在 BUSY（`current_job_id` 指向已取消任务），调度器不会再派给它：

```bash
cd ~/minimax-h3-orchestrator
source .venv/bin/activate
python -c "
from controller import db
from controller.constants import WorkerStatus
s = db.Store()
with s.session() as ss:
    for w in ss.query(db.Worker).all():
        if w.status == WorkerStatus.WORKER_BUSY.value and w.current_job_id:
            w.status = WorkerStatus.WORKER_READY.value
            w.current_job_id = None
    ss.flush()
print('cleaned')
"
```

## 调试技巧

### 任务失败看真实错误

```bash
curl -sS http://127.0.0.1:8001/api/v1/jobs | python3 -c "
import sys,json
for j in json.load(sys.stdin):
    if j['status']=='FAILED':
        print(j['job_id'][:18], '|', j.get('error','')[:200])
"
```

常见错误：
- `timed out after 3600s` → 步数太多/没挂 turbo，降到 4 步
- `no video output produced` → 分辨率太高 OOM，降到 832×480
- `Invalid image file: xxx.png` → 参考图没上传到 ComfyUI input/ 目录（参考图必须走 worker 上传链路）
- `first_frame not provided` → workflow 字段丢了（controller 跑旧代码没重启）

### 手动派发卡住的任务

如果任务 QUEUED 但 worker READY，可以手动 tick 一次：

```bash
cd ~/minimax-h3-orchestrator
source .venv/bin/activate
python -c "
from controller import db
from controller.jobs import JobManager
from controller.workers import WorkerManager
from controller.accounts import AccountManager
from controller.scheduler import Scheduler
from controller.main import KaggleProvider
s = db.Store()
sched = Scheduler(s, JobManager(s), WorkerManager(s), AccountManager(s), KaggleProvider())
sched.tick()
"
```

## 参考：本地脚本生成器

`~/.openclaw/workspace/multishot/script_writer.py` 用可配置 LLM 生成 Multishot 分镜脚本（内置官方 6 条边界规则）：

```bash
python3 ~/.openclaw/workspace/multishot/script_writer.py \
  --premise "故事梗概" --shots 12 --frames 124 --dialogue no
```

默认调本机 3000 端口（deepseek-pro），`--base-url` / `--api-key` / `--model` 可覆盖。

## 测试

```bash
cd ~/minimax-h3-orchestrator
source .venv/bin/activate
python -m pytest tests/ -q   # 62 个测试，全 in-process，无网络/GPU
```
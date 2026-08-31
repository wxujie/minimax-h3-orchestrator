---
name: minimax-h3-orchestrator
description: 操作 MiniMax-H3 orchestrator（Kaggle/Colab GPU 池 + Cloudflare 隧道跑 ComfyUI 视频生成）。当需要提交 FL2VA / R2V / Multishot 视频任务、重启 controller、重启 Kaggle kernel、登录 Colab 账号、调试 worker 注册、或排查生成失败/超时时使用。包含 T4 实测速度配方和所有踩过的坑。
---

# MiniMax-H3 Orchestrator 操作手册

这是一个用免费 Kaggle GPU 池 + Google Colab GPU + Cloudflare 隧道跑 MiniMax-H3 视频生成的调度系统。controller 是本机 FastAPI 服务，worker 是 Kaggle notebook / Colab session 里的 ComfyUI 实例，通过 Cloudflare 隧道回连。

## 架构速览

```
Client → controller (本机 :8001, FastAPI) → Kaggle notebook × 2 worker (ComfyUI)
        ↑ Cloudflare 隧道回连        └──── Colab session × 1 worker (ComfyUI)
```

- **controller**：本机 `uvicorn controller.main:app --port 8001`
- **worker（Kaggle）**：Kaggle notebook 启动时 clone 本 repo，起 2 个 ComfyUI（gpu0/gpu1），开隧道回注册
- **worker（Colab）**：Colab session 启动时 clone 本 repo，起 1 个 ComfyUI（单卡 T4），开隧道回注册
- **公网入口**：`https://controller.jayapp.cn` → cloudflared → `localhost:8001`
- **多账号**：Kaggle 用 `KAGGLE_ACCOUNT_<N>_*`，Colab 用 `COLAB_ACCOUNT_<N>_*`（OAuth 本地登录态）

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
- `timeout_s`：**任务级渲染超时（秒）**，可选。不传则回落全局 `JOB_TIMEOUT_S`（`.env`，默认 7200）→ worker 的 `spec.job_timeout_s`。多镜头/参考图任务建议显式传大一点，短任务可传小一点提前失败重试。

## T4 实测速度配方（★ 必读，否则会超时）

**最优解：int8 ref2va + turbo LoRA 4步 + 832×480**

| 配置 | 耗时 |
|------|------|
| 20步无turbo | >60min 超时 ❌ |
| 8步+turbo | ~60min 超时 ❌ |
| **4步+turbo** | **~30min** ✅ |
| **4步+turbo+TeaCache** | **20min（5s）/ 23min（10s多镜头）** ✅✅ |
| 4步+turbo+参考图 | ~49min（参考token贯穿每步，慢1.6x）|
| 阿里 PDD 8步（Ref2VA） | >60min 超时 ❌（head bank 在 T4 int8 换页下 thrash）|

**平台实测（2026-08-28）：**

| 平台 | 配置 | 内容 | 耗时 |
|------|------|------|------|
| Kaggle | 4步turbo+TeaCache | 5s单镜 | 20min |
| Colab | 4步turbo 无TeaCache | 5s单镜 | 23min |
| Colab | 4步turbo+TeaCache | **10s多镜头(2镜)** | **23min** ✅ |

**平台实测（2026-08-29，Colab 双账号并行）：**

| 配置 | 内容 | 结果 |
|------|------|------|
| 4步turbo+TeaCache | **2×5s（2镜×124帧）** | ✅ **纯渲染 ~11-12min**（shot1 初始化~5分半 + shot2 初始化~6分；每镜重新初始化模型是大头，采样只要几秒）|
| 4步turbo+TeaCache | **8s单镜（192帧）** | ⚠️ 采样4步全跑完、进入VAE解码，但 VM 被回收掉线（跑两次都被回收，未出片）|
| 4步turbo+TeaCache | **10s单镜（238帧）** | ❌ **OOM**：峰值显存 13.1GB，109秒即触发 CUDA OOM，不是 3600s 超时 |

**关键结论：**
- **TeaCache（UC_MiniMaxH3Cache）是 T4 上的核心提速件**——内容翻倍耗时几乎不变
- **10秒单镜头在 T4 上必然 OOM**：峰值显存 13.1GB（T4 只有 15GB），约 109 秒就爆，不是慢慢超时。要 10 秒就 5s×2 链式
- **8秒单镜（192帧）采样阶段能跑完**，但 Colab 免费档 VM 在重负载 ~25 分钟后会掉线，长任务风险很高
- **2×5s 是 T4 上最稳定的配置**，纯渲染约 11-12 分钟
- **阿里 PDD Acc LoRA 在小显存卡上不可行**，放弃
- **Colab 免费档 VM 不稳定**：实测跑 2×5s 完成后 ~25min 掉线；8s 单镜两次都在 VAE 解码阶段被回收。长任务（>20min）建议 Kaggle 或降低时长

**两个加速实验都否决了，别再试：**
- `--enable-triton-backend`：T4 是 Turing 老架构，triton 优化 Ampere/Hopper，加了反而更慢（39min）
- GGUF Q4_0：4bit 精度让 turbo 4步不收敛，48min 比 int8 还慢

### TeaCache 接入要点（★ 必读）

TeaCache 用 `UC_MiniMaxH3Cache` 节点（来自 `ComfyUI-UtilsCollection`）。

- **依赖**：`opencv-python` + `typing-extensions` + `unifiedefficientloader>=0.5.3`。**`comfy_api` 是 ComfyUI 内置模块，不是 PyPI 包，不要 pip install**
- 接入位置：`H3LoraStack → UC_MiniMaxH3Cache → H3MultishotSampler`
- 参数：`reuse_threshold=0.15`（默认0.05），`device=cpu`（T4 显存紧，residual 放 CPU）
- API 请求：`use_teacache=true, teacache_thresh=0.15`

### ⚠️ ComfyUI 0.34+ 与 TeaCache 的兼容性 bug（2026-08-29 踩坑）

ComfyUI 升级到 **0.34.0** 后，`comfy/ldm/minimax/model.py` 的
`FinalLayer.forward()` 从 4 个参数变成 7 个必需参数：

```python
# 旧签名（UtilsCollection patch 还在用这个）
def forward(self, x, t_emb, video_seg, audio_seg): ...
# 新签名（0.34+，多了 PDD head bank 用的 sigma/sample_sigmas/shifts）
def forward(self, x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts): ...
```

`ComfyUI-UtilsCollection`（TeaCache 的 `UC_MiniMaxH3Cache`）的
`patcher_helpers.py::minimax_h3_block_patch_forward` 还在按旧签名调用
`self.final_layer(...)`，导致一开 TeaCache 就：

```text
TypeError: FinalLayer.forward() missing 3 required positional arguments:
'sigma', 'sample_sigmas', and 'shifts'
```

**症状**：任务派上去约 5 分钟就 FAILED，`last_error="no video output produced"`，
ComfyUI 日志里是这个 TypeError。**不是 OOM 也不是超时**。

**修复**：已写进 notebook 的 cell 15（clone 完 UtilsCollection 后探测签名并原地 patch）：

```python
# 把旧调用
self.final_layer(hidden_states, timestep_embedding, video_seg, audio_seg)
# 改成（sigma_v / transformer_options / shift_v / shift_a 都在 patch 函数作用域内）
self.final_layer(hidden_states, timestep_embedding, video_seg, audio_seg,
                 sigma_v, transformer_options.get("sample_sigmas"), (shift_v, shift_a))
```

上游 `silveroxides/ComfyUI-UtilsCollection` 的 main 分支（截至 2026-08-29）**也还没修**，
所以不能靠升级 UtilsCollection 解决——要么等上游跟进，要么保持这个 notebook 补丁。

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

### Colab worker（★ 第二 GPU 池）

Colab 是 Kaggle 的替代/补充 worker 池，单卡 T4，通过同一套隧道机制回连。

**登录账号（一次性，每个账号一条命令）：**

```bash
bash scripts/colab-login.sh colab-account-1
bash scripts/colab-login.sh colab-account-2
# 会弹出 Google OAuth 链接，浏览器授权后粘回 code
```

token 落到 `~/.colab-accounts/<id>/.config/colab-cli/token.json`，多账号 HOME 隔离互不覆盖。

**手动操作某个账号（关键：token 靠 `$HOME` 隔离，不是 `--config`）：**

```bash
cd ~/minimax-h3-orchestrator
# 操作 1 号账号（wxujie666@gmail.com）
HOME=/home/jieubuntu26/.colab-accounts/colab-account-1 uv run colab whoami
HOME=/home/jieubuntu26/.colab-accounts/colab-account-1 uv run colab status
HOME=/home/jieubuntu26/.colab-accounts/colab-account-1 uv run colab sessions

# 操作 2 号账号（wxingxing2026@gmail.com）
HOME=/home/jieubuntu26/.colab-accounts/colab-account-2 uv run colab status

# ⚠️ 只传 --config 不会切 token！实测：
#   uv run colab --config .../colab-account-2/.../sessions.json whoami
#   仍然读默认 HOME 的 token（读到 1 号账号），必须同时设 HOME。
# --config 只是显式指定 sessions.json，默认 HOME 下等价于
# ~/.config/colab-cli/sessions.json。
```

`uv run colab --help` 完整输出（命令全集）：

```text
Usage: colab [OPTIONS] COMMAND [ARGS]...

 Colab CLI

Options:
  --client-oauth-config -c <str>  Path to client OAuth config JSON file
                                  [default: /home/jieubuntu26/.colab-cli-oauth-config.json]
  --config            <str>       Path to session state file
                                  (~/.config/colab-cli/sessions.json)
  --logtostderr                   Log all output to stderr
  --auth  <oauth2|adc>            Authentication strategy: 'oauth2' (public
                                  InstalledAppFlow) or 'adc' (Application
                                  Default Credentials). [default: oauth2]
  --install-completion            Install completion for the current shell.
  --show-completion               Show completion for the current shell.
  --help  -h                      Show this message and exit.

Commands:
  console         Connect to raw TTY console
  download        Download a file from a session
  drivemount      Mount Google Drive at path
  edit            Edit a file on a running Colab session
  exec            Execute code in a session
  help            Show help for a command.
  install         Install python packages on the VM
  log             Manage and view session history logs
  ls              List files in a session
  new             Create a new session
  pay             Open the Colab signup page to manage compute units
  readme          Print the bundled README.md file
  repl            Start an interactive REPL
  restart-kernel  Restart a session's kernel
  rm              Remove a remote file
  run             Run a Python script on a fresh Colab VM, then release the VM
  sessions        List all active sessions
  skill           Print the bundled COLAB_SKILL.md file
  status          Show session status
  stop            Stop a session
  update          Check for latest version and print if an update is available
  upload          Upload a file to a session
  url             Print a browser URL that connects to an existing session.
  version         Show the version of the Colab CLI
```

常用子命令（都需前缀 `HOME=<账号 HOME> uv run colab`）：
- `new -s <name> --gpu T4`：开一个带 T4 的 session
- `exec -s <name> -f <本地脚本.py>`：把本地脚本传到 session 里跑（`--timeout` 设秒数）
- `status` / `sessions`：查 session 状态 / 列表
- `stop -s <name>`：终止 session（Kaggle 没有的 terminate）
- `download -s <name> <远端路径> <本地路径>`：从 session 拉文件（**不走 kernel，比 exec 稳**）
- `upload -s <name> <本地> <远端>`：传文件（**不走 kernel，比 exec 稳**）
- `ls -s <name> <远端目录>`：列目录
- `restart-kernel -s <name>`：重启 kernel（⚠️ 可能触发 Colab 回收 VM）

**启用账号（`.env`）：**

```bash
COLAB_ACCOUNT_1_ID=colab-account-1
COLAB_ACCOUNT_1_ENABLED=true
```

**关键差异 vs Kaggle：**
- 单卡 T4（gpu_count=1），Kaggle 双卡
- 能主动 `stop`（Kaggle 无 terminate API）
- CLI 自带 keep-alive，但**免费档长任务仍可能被回收**（实测跑一半 session 没了）
- `colab exec` 输出是**结束后一次性返回**，跑长任务时看不到实时进度

**⚠️ 依赖版本坑：**
- `jupyter-kernel-client` 必须 `<1.0`（1.0.2 把 `KernelClient` 改名成 `JupyterKernelClient`，colab CLI 0.6.0 会崩）
- OAuth scope 校验 bug：登录脚本已内置 `OAUTHLIB_RELAX_TOKEN_SCOPE=1`

### Cloudflare named 隧道（★ 固定域名回连，替代 quick tunnel）

worker 默认 `TUNNEL_MODE=quick`（随机 trycloudflare URL），生产用 `named` 固定域名
`https://<worker-id>.<TUNNEL_DOMAIN>`。用的是 **locally-managed 隧道**
（`cloudflared tunnel create` 的 credentials + config.yml），**不是** remotely-managed `--token`。

建隧道统一用脚本（幂等，`--force` 强制重建，自动配一级子域 + proxied=false + 写回 .env）：

```bash
cd ~/minimax-h3-orchestrator
bash scripts/create-worker-tunnel.sh <account_id> <notebook_name> <gpu_count> <agent_port_base> jayapp.cn [--force]
# 例：bash scripts/create-worker-tunnel.sh kaggle-account-1 nb-kaggle-account-1 1 8000 jayapp.cn --force
```

三条硬性要求（**每一条都实测踩过坑**）：

1. **TUNNEL_DOMAIN 必须是一级子域**（如 `jayapp.cn`），hostname 形如 `<worker-id>.jayapp.cn`。
   Cloudflare 通用证书只覆盖 `*.jayapp.cn`，**二级子域**（`*.tunnel.jayapp.cn`）不在证书里，
   导致 TLS handshake failure（alert 40 / no peer certificate）。
2. **DNS 记录必须 proxied=false（灰云）**。`cloudflared tunnel route dns` 默认建 `proxied=true`
   （橙云），橙云让边缘尝试 SSL 代理，与隧道 QUIC 通道冲突，同样 TLS handshake failure。
   脚本已自动用 cert.pem 里解码的 apiToken + zoneID 调 Cloudflare API 改 false。
3. **credentials-file 路径**：写 `/tmp/cloudflared-tunnel/credentials.json`（不能用 `/tmp/cloudflared`，
   那是 cloudflared 二进制文件，会 FileExistsError）。

**cloudflared 参数顺序坑**：`--config` 和 `--no-autoupdate` 都必须放 `run` 之前：
`cloudflared tunnel --config <path> --no-autoupdate run`（放后面会 "exited early" 死循环）。

**隧道被删的坑**：Cloudflare 控制台里隧道若被删，.env 里的 credentials 会失效，
cloudflared 日志报 `error="Unauthorized: Tunnel not found"` + `Register tunnel error`，
worker 永远 READY 不了。诊断：`cloudflared tunnel token <name>` 报 "neither the ID nor the
name of any of your tunnels"。修复：`--force` 重建脚本重新 create + 配 DNS + 写回 .env。

**worker agent 还暴露 `/debug` 端点**（Bearer WORKER_AUTH_SECRET），返回 threads /
comfy_queue / comfy_log_tail 三字段，卡死排查用：
```bash
curl -sS --noproxy '*' -H "Authorization: Bearer $WORKER_AUTH_SECRET" <worker_tunnel_url>/debug
```

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
- `timed out after 3600s` → 步数太多/没挂 turbo，降到 4 步，或挂 TeaCache
- `no video output produced` → 分辨率太高 OOM，降到 832×480
- `Invalid image file: xxx.png` → 参考图没上传到 ComfyUI input/ 目录（参考图必须走 worker 上传链路）
- `first_frame not provided` → workflow 字段丢了（controller 跑旧代码没重启）
- `missing_node_type: 'UC_MiniMaxH3Cache'` → 任务派给了没装 UtilsCollection 的旧 worker（Kaggle 旧 worker 没装 TeaCache）。把旧 worker 下线，或重启它拉新代码
- `SSL: UNEXPECTED_EOF`（轮询时）→ 本机代理（127.0.0.1:7890）拦截了 controller→worker 的 Cloudflare 请求。已用 trust_env=False 修复，但旧代码需重启 controller / worker 才生效

### 本机代理坑（★ 今晚最隐蔽的）

controller 进程会继承 gateway 的 `https_proxy=http://127.0.0.1:7890`。httpx 默认 `trust_env=True` 会走这个代理，代理转发 Cloudflare 的 TLS 握手会坏掉，表现成 worker 永远连不上（SSL EOF）。

**已修复**：`controller/worker_client.py` 和 `worker/runner.py` 的所有 httpx 调用都加了 `trust_env=False`。

验证方法：
```bash
# 不走代理直连 worker（应该 200）
curl -sS --noproxy '*' <worker_tunnel_url>/health
# 走代理（会 SSL EOF）
curl -sS <worker_tunnel_url>/health
```

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
python -m pytest tests/ -q   # 75 个测试，全 in-process，无网络/GPU
```
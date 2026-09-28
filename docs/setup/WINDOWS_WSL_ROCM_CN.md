# Windows 迁移与 WSL2 + ROCm 训练手册

本文用于把本项目迁移到装有 AMD Radeon RX 7800 XT 的 Windows 机器，并继续训练
VisDrone。当前项目已验证的路线是：

```text
Windows 11 + AMD WSL 驱动
  -> WSL2 Ubuntu 24.04
  -> ROCm 7.2.3 官方 PyTorch Docker 镜像
  -> PyTorch 2.9.1 + torchvision 0.24.0
  -> RT-DETR + VisDrone
```

原生 Windows Python/ROCm 不属于本项目的已验证环境。为了减少环境差异，迁移后仍
建议使用 WSL2 + Docker，不要直接在 PowerShell 里运行训练脚本。

本文中的命令分为两类：标有“PowerShell”的命令在 Windows Terminal/PowerShell 中
执行；标有“WSL”的命令在 Ubuntu 终端中执行。

## 1. 已验证基线

截至 2026-07-20，本项目验证通过的组合如下：

| 项目 | 版本或值 |
| --- | --- |
| GPU | AMD Radeon RX 7800 XT，`gfx1101` |
| WSL 发行版 | Ubuntu 24.04 |
| ROCm 镜像 | `rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.9.1` |
| Python | 3.12 |
| PyTorch | `2.9.1+rocm7.2.3` |
| torchvision | `0.24.0+rocm7.2.3` |
| VisDrone train | 6471 张图片，343204 个有效目标 |
| VisDrone val | 548 张图片，38759 个有效目标 |

AMD 的 ROCm 7.2 WSL 支持矩阵列出了 RX 7800 XT，并将 PyTorch 2.9.1 标为生产支持；
nightly 版本没有经过同等程度的测试。迁移时应保留镜像自带的 torch/torchvision，
不要再运行会覆盖它们的 nightly 安装命令。

官方参考：

- [ROCm 7.2 WSL 支持矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html)
- [在 WSL 安装 Radeon Software 与 ROCm](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/wsl/install-radeon.html)
- [AMD WSL PyTorch 安装说明](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/legacywsl/install-pytorch.html)

## 2. 搬运前必须知道的内容

以下目录被 `.gitignore` 忽略，只复制或克隆 Git 仓库不会带到新机器：

| 路径 | 内容 | 是否必须搬运 |
| --- | --- | --- |
| `rtdetr_pytorch/configs/dataset/VisDrone/` | VisDrone 原始图片和 TXT 标注，约 1.9 GB | 必须 |
| `rtdetr_pytorch/configs/dataset/visdrone_annotations/` | 转换后的 COCO JSON，约 45 MB | 可重新生成 |
| `rtdetr_pytorch/output/` | checkpoint、日志和评估结果 | 需要续训时必须 |
| `rtdetr_pytorch/configs/dataset/VisDrone.previous/` | 旧的不完整数据备份 | 不要搬运 |

推荐把项目放在 WSL 的 Linux 文件系统，例如
`/home/<WSL用户名>/rocm-work/my_lab/bachelor paper`。不要长期从 `/mnt/c` 或 `/mnt/d`
直接训练；大量小图片的读取速度通常更差。

### 方案 A：整体打包（最不容易漏文件）

在旧机器的 WSL 中执行：

```bash
cd "/home/lan-ubuntu/rocm-work/my_lab"
tar \
  --exclude='bachelor paper/rtdetr_pytorch/configs/dataset/VisDrone.previous' \
  -czf /mnt/c/Users/<Windows用户名>/Desktop/rtdetr_visdrone_bundle.tar.gz \
  'bachelor paper'
```

这个归档包含代码、当前 VisDrone、COCO JSON 和已有输出。将归档复制到新 Windows
机器后，在新机器的 WSL 中执行：

```bash
mkdir -p "/home/<WSL用户名>/rocm-work/my_lab"
tar -xzf \
  /mnt/c/Users/<Windows用户名>/Desktop/rtdetr_visdrone_bundle.tar.gz \
  -C "/home/<WSL用户名>/rocm-work/my_lab"
```

把命令中的两个用户名占位符替换成新机器的实际用户名。解压后应存在：

```text
/home/<WSL用户名>/rocm-work/my_lab/bachelor paper/rtdetr_pytorch
```

### 方案 B：Git + 单独复制数据

如果代码通过 Git 获取，仍须单独复制 `VisDrone`；需要续训时也要复制 `output`。
`visdrone_annotations` 可以不复制，之后运行转换器重新生成。

## 3. Windows 和 WSL2 前置环境

### 3.1 安装或更新 WSL2

以管理员身份打开 PowerShell：

```powershell
wsl --install -d Ubuntu-24.04
wsl --update
wsl --shutdown
```

重启 Windows 后检查：

```powershell
wsl --version
wsl -l -v
```

Ubuntu 所在行的 `VERSION` 应为 `2`。如果不是：

```powershell
wsl --set-version Ubuntu-24.04 2
```

### 3.2 安装 AMD 的 WSL 兼容驱动

从 AMD 当前 ROCm WSL 安装页选择支持矩阵要求的 Adrenalin WSL2 驱动，并在安装后
重启 Windows。不要只根据普通游戏驱动是否最新来判断兼容性；以 AMD 当前支持矩阵
为准。

### 3.3 在 Ubuntu 中安装 ROCm WSL 组件

以下是 ROCm 7.2、Ubuntu 24.04 的官方安装方式。在 WSL 中执行：

```bash
sudo apt update
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb
sudo apt install ./amdgpu-install_7.2.70200-1_all.deb
sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms
```

安装完成后重新打开 WSL，验证：

```bash
test -e /dev/dxg && echo '/dev/dxg OK'
test -e /usr/lib/wsl/lib/libdxcore.so && echo 'libdxcore OK'
test -e /opt/rocm/lib/libhsa-runtime64.so.1 && echo 'HSA runtime OK'
test -e /opt/rocm/lib/librocdxg.so && echo 'librocdxg OK'
rocminfo | grep -E 'Name:|Marketing Name:'
```

必须能看到 `gfx1101` 或 `AMD Radeon RX 7800 XT`，再继续创建容器。

### 3.4 准备 Docker

可以使用 Docker Desktop 的 WSL 集成，也可以在 Ubuntu 中安装 Docker Engine。无论
选择哪种方式，以下命令必须在 WSL 中成功：

```bash
docker version
docker run --rm hello-world
```

## 4. 创建稳定训练容器

先进入项目并确认路径。下例假设 WSL 用户名是 `<WSL用户名>`：

```bash
cd "/home/<WSL用户名>/rocm-work/my_lab/bachelor paper/rtdetr_pytorch"
pwd
```

创建模型缓存目录：

```bash
mkdir -p "/home/<WSL用户名>/models"
```

然后创建容器。三个 WSL 运行库挂载都要保留：

```bash
docker run -dit --name rocm723_visdrone \
  --device=/dev/dxg \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host --shm-size 32g \
  -e HSA_ENABLE_DXG_DETECTION=1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib \
  -v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so:ro \
  -v /opt/rocm/lib/libhsa-runtime64.so.1:/opt/rocm/lib/libhsa-runtime64.so.1:ro \
  -v /opt/rocm/lib/librocdxg.so:/usr/lib/librocdxg.so:ro \
  -v "/home/<WSL用户名>/rocm-work:/workspace" \
  -v "/home/<WSL用户名>/models:/models" \
  -w "/workspace/my_lab/bachelor paper/rtdetr_pytorch" \
  rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.9.1 \
  bash
```

安装项目依赖。`requirements.txt` 已故意不包含 torch 和 torchvision：

```bash
docker exec rocm723_visdrone python3 -m pip install -r requirements.txt
```

容器已经存在但处于停止状态时，不要重复 `docker run`。启动并直接打开 Bash 主终端：

```bash
docker start -ai rocm723_visdrone
```

也可以在项目根目录运行 `./start_visdrone_container.sh`；它会根据状态执行
`docker start -ai` 或 `docker attach`。看到类似
`root@容器ID:/workspace/my_lab/bachelor paper/rtdetr_pytorch#` 的提示符即表示成功。
只想离开终端并保持容器运行时，按 `Ctrl+P`，再按 `Ctrl+Q`。输入 `exit` 会结束作为
主进程的 Bash，因此容器也会停止。

## 5. 迁移后的强制自检

### 5.1 固定框架版本并测试 GPU NMS

```bash
docker exec rocm723_visdrone python3 -c \
  "import torch, torchvision; from torchvision.ops import nms; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('HIP:', torch.version.hip); print('GPU:', torch.cuda.is_available()); print('name:', torch.cuda.get_device_name(0)); print('NMS:', nms(torch.tensor([[0.,0.,10.,10.]], device='cuda'), torch.tensor([0.9], device='cuda'), 0.5))"
```

预期重点：

```text
torch: 2.9.1+rocm7.2.3...
torchvision: 0.24.0+rocm7.2.3...
GPU: True
name: AMD Radeon RX 7800 XT
NMS: tensor([0], device='cuda:0')
```

### 5.2 检查数据并重新生成 COCO JSON

```bash
docker exec rocm723_visdrone sh -lc \
  "find configs/dataset/VisDrone/VisDrone2019-DET-train/images -type f | wc -l; find configs/dataset/VisDrone/VisDrone2019-DET-val/images -type f | wc -l"

docker exec rocm723_visdrone \
  python3 tools/convert_visdrone_to_coco.py
```

图片数应依次为 `6471` 和 `548`。转换器会检查图片与 TXT 标注是否一一配对，并输出
train/val 的图片数和有效目标数。

## 6. 训练、停止、查看日志和恢复

### 6.1 前台训练（首次验证推荐）

```bash
docker exec -it rocm723_visdrone \
  python3 tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_visdrone.yml
```

看到以下阶段表示环境和数据已经跑通：

```text
Start training
Load PResNet18 state_dict
loading annotations into memory...
number of params: 20094584
```

ROCm/MIOpen 第一次运行可能进行算子调优，首批明显慢于后续批次。

### 6.2 后台训练并保存控制台日志

```bash
docker exec rocm723_visdrone \
  mkdir -p output/rtdetr_r18vd_6x_visdrone

docker exec -d rocm723_visdrone sh -lc \
  "PYTHONUNBUFFERED=1 python3 tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_visdrone.yml > output/rtdetr_r18vd_6x_visdrone/train_console.log 2>&1"
```

查看日志：

```bash
docker exec -it rocm723_visdrone \
  tail -f output/rtdetr_r18vd_6x_visdrone/train_console.log
```

按 `Ctrl+C` 只会退出 `tail`，不会停止后台训练。

### 6.3 检查或停止训练

检查：

```bash
docker exec rocm723_visdrone sh -lc \
  "ps -eo pid,etime,pcpu,pmem,args | grep '[p]ython3 tools/train.py'"
```

停止：

```bash
docker exec rocm723_visdrone pkill -TERM -f \
  "python3 tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_visdrone.yml"
```

### 6.4 从 checkpoint 恢复

每个完整 epoch 结束后，项目会更新：

```text
rtdetr_pytorch/output/rtdetr_r18vd_6x_visdrone/checkpoint.pth
```

恢复训练：

```bash
docker exec -it rocm723_visdrone \
  python3 tools/train.py \
  -c configs/rtdetr/rtdetr_r18vd_6x_visdrone.yml \
  -r output/rtdetr_r18vd_6x_visdrone/checkpoint.pth
```

如果在第一个 epoch 尚未完成时停止，不会生成 checkpoint，只能重新从 epoch 0 开始。

## 7. 常见问题

### `RuntimeError: operator torchvision::nms does not exist`

torch 与 torchvision 二进制不匹配。不要通过安装 nightly 单独修补。删除并重建容器，
使用本手册固定的官方镜像，然后只安装 `requirements.txt`。

### 出现 `rocprofiler agents ... Aborted (core dumped)`

这是本项目曾安装 nightly 后遇到的崩溃。重建稳定容器，不要在已污染的虚拟环境中继续
叠加安装不同版本。

### `torch.cuda.is_available()` 为 `False`

依次检查：

1. Windows 使用 AMD 当前支持的 WSL2 驱动并已重启。
2. WSL 中存在 `/dev/dxg`。
3. `rocminfo` 能看到 `gfx1101`。
4. Docker 命令同时挂载了 `libdxcore.so`、`libhsa-runtime64.so.1` 和
   `librocdxg.so`。
5. 容器中的 torch/torchvision 仍是官方镜像自带的配套版本。

### `amd-smi` 或 `rocm-smi` 报 `amdgpu not found in modules`

WSL 使用 `/dev/dxg`，这些工具可能因为没有传统 Linux `amdgpu` 内核模块而失败。
这不等于 PyTorch 没有使用 GPU。以 `torch.cuda.is_available()`、GPU 名称、GPU NMS
测试和训练进程持有 `/dev/dxg` 为准。

### 容器中找不到数据或项目

检查 WSL 用户名和 `-v` 左侧宿主路径。Windows 的 `D:\...` 路径不能直接写进 Linux
容器挂载命令；在 WSL 中对应为 `/mnt/d/...`。为了训练性能，仍建议先复制到 WSL 的
`/home/<WSL用户名>/...`。

### Docker 提示容器名已存在

先检查：

```bash
docker ps -a --filter name=rocm723_visdrone
```

如果只是停止了，运行 `docker start -ai rocm723_visdrone`，或使用项目根目录中的
`./start_visdrone_container.sh`。只有确认旧容器不再需要时才删除并重建。

## 8. 迁移完成检查表

- [ ] Windows 驱动与 ROCm WSL 支持矩阵匹配。
- [ ] Ubuntu 使用 WSL2，不是 WSL1。
- [ ] `/dev/dxg` 和三个 WSL 运行库都存在。
- [ ] `rocminfo` 能看到 RX 7800 XT / gfx1101。
- [ ] 项目位于 WSL Linux 文件系统，不是长期从 `/mnt/c` 训练。
- [ ] VisDrone train/val 图片数为 6471/548。
- [ ] torch/torchvision 为 2.9.1/0.24.0 的 ROCm 7.2.3 配套版本。
- [ ] GPU NMS 返回 `tensor([0], device='cuda:0')`。
- [ ] 需要续训时，`output/.../checkpoint.pth` 已单独搬运。

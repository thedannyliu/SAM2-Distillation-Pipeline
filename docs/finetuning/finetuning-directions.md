先講結論：以下把你說的「一個 SAB」理解成「只能使用一個 SA‑V subset/shard」。如果其實是 SA‑1B 靜態影像，memory module 無法只靠它有效學到影片 propagation。

你的上一個配置——凍結 mask decoder、memory，只用 task mask loss 更新 image encoder——剛好是最難穩定的方向。對原生 SAM2.1 checkpoint，官方做法是全模型、極小 differential LR；對你這種把 Hiera 換成 TinyViT 的 student，則應先做 feature/interface alignment，再訓 memory，最後才短暫低 LR E2E。

## 最重要的判斷

### 原生 SAM2.1

直接把官方 MOSE fine-tune 當 baseline：

| 設定                  |                 官方值 |
| ------------------- | ------------------: |
| Resolution          |                1024 |
| Frames/clip         |                   8 |
| Objects/clip        |                最多 3 |
| Global batch        |   8 clips，8 GPU × 1 |
| Image encoder LR    |              `3e-6` |
| 其他模組 LR             |              `5e-6` |
| Optimizer           |               AdamW |
| Precision           |                BF16 |
| Weight decay        | `0.1`；bias/norm 為 0 |
| Encoder layer decay |               `0.9` |
| Global grad clip    |               `0.1` |
| LR schedule         |  cosine 到原 LR 的 0.1 |
| Trainable modules   |               預設全模型 |

這份 config strict-load SAM2.1 checkpoint，並沒有凍結 decoder/memory。[官方 MOSE config](https://github.com/facebookresearch/sam2/blob/main/sam2/configs/sam2.1_training/sam2.1_hiera_b%2B_MOSE_finetune.yaml)；官方 training README 報告該 recipe 約可達 MOSE 79.4 J&F。[SAM2 training README](https://github.com/facebookresearch/sam2/blob/main/training/README.md)

### TinyViT student／替換 encoder

不能直接套上面。TinyViT 輸出必須先和 frozen SAM2 decoder/memory 的介面相容，尤其是：

* stride-4 high-resolution feature
* stride-8 high-resolution feature
* stride-16 image/memory feature
* memory-attention output feature

SAM2 decoder 同時使用三個尺度，不是只吃最後一層。如果你之前只 distill 最後一個 embedding，即使 feature loss 很低，stride-4/8 skip 仍可能完全不匹配。[SAM2 v2 architecture](https://arxiv.org/html/2408.00714v2)

[EdgeTAM](https://arxiv.org/html/2501.07256v1) 是最接近你情境的證據：它同時保留 task loss、image feature MSE 與 memory-output MSE，加入 distillation 後，SA‑V val/test J&F 分別提升約 1.3/3.3；不是只靠 GT mask task loss硬拉 encoder。[EdgeTAM code](https://github.com/facebookresearch/EdgeTAM)

## 為什麼你之前可能完全沒有學到

按優先級檢查：

1. **Frozen downstream 被包在 `no_grad()`**

   正確的 freeze：

   ```python
   for p in decoder.parameters():
       p.requires_grad_(False)

   decoder.eval()
   pred = decoder(student_features, prompts)  # 保留 autograd
   ```

   `requires_grad=False` 只是不更新 decoder 參數，task loss 仍可經 decoder Jacobian 回到 encoder。若把 decoder/memory forward 放進 `torch.no_grad()` 或 `inference_mode()`，encoder 的 task gradient 就被切斷。

   `eval()` 不會切斷 gradient；它只固定 Dropout/BN 行為。

2. **GT mask prompt bypass decoder**

   官方設定包含 `use_mask_input_as_output_without_sam: true`。如果 T=1 時把 GT mask 同時當 prompt，該 frame 可能直接輸出輸入 mask，幾乎不經過 decoder。因此單幀 interface training 必須用 point/box prompt；GT mask prompt留到多幀 propagation 的 conditioning frame。

3. **你在訓「錯的一側」**

   ImageNet/TinyViT feature 與 frozen SAM2 decoder/memory 不相容時，只要求 encoder 自己扭曲輸出來配合所有 frozen downstream，是高方差、病態的 optimization。應先用 KD 固定介面，再讓 memory/decoder逐步適配。

4. **LR 可能大了 10–100 倍**

   大規模 SAM2 pretraining 的 `6e-5/3e-4` 不適合單一小資料 fine-tune。官方 downstream 是 `3e-6/5e-6`，且 grad clip `0.1`。

5. **未標註 frame 被當成 empty mask**

   SA‑V mask是 6 FPS annotation。24 FPS 影片中的其他 frame「未標註」不等於「物件不存在」。未標註 frame應 ignore supervised loss；只有已標註且確認物件消失的 frame，才監督 object-presence/occlusion。

6. **Raw loss 本來就不平滑**

   每 step 的物件數、大小、prompt種類、correction次數、occlusion與 multimask token都不同，因此 raw total loss必然 jagged。真正判斷不收斂要看 fixed-validation J&F、100/500-step EMA，以及固定 mini-set 能否 overfit。

## 我建議你的完整流程

你已有的 TinyViT image-feature distillation checkpoint可作為起點。

| 階段                    | Clip／資料                                              | Trainable                                             | Frozen                             |       建議步數 |
| --------------------- | ---------------------------------------------------- | ----------------------------------------------------- | ---------------------------------- | ---------: |
| D0 Debug              | 固定 8–16 clips、T=2、1 object、box prompt、無 augmentation | neck/projection + TinyViT last stage                  | decoder/memory/prompt，但保留 autograd |    200–500 |
| S1 Interface          | T=1，point/box各 50%                                   | neck/projection、IoU/presence heads、TinyViT last stage | core decoder、memory、prompt         |         2k |
| S2 Memory calibration | T=4、1–2 objects、0→3 corrections                      | neck、memory encoder/attention、object pointer、heads    | TinyViT trunk、prompt、core decoder  |         3k |
| S3 Main E2E           | T=8、最多 3 objects、完整 prompt simulation                | neck、memory、decoder、TinyViT逐步全解凍                      | prompt encoder、BN running stats    |      8–10k |
| S4 Long clip          | T=16、top 50% hard clips                              | memory、decoder、heads                                  | 整個 image encoder/neck、prompt       | 約 S3 的 1/3 |

D0 必須能把固定小資料 overfit 到至少約 0.9 J&F，且每個預期 trainable module 都有非零 gradient。做不到就先修 pipeline，不要調 batch size。

S4 直接有官方先例：SAM2 v2 使用 16 frames、top 50% most-edited masklets、原 schedule 的 1/3、half LR，並凍結 image encoder。[SAM2 v2 long-clip fine-tuning](https://arxiv.org/html/2408.00714v2)

### 各模組的建議 LR

| Parameter group         |             LR 起點 |
| ----------------------- | ----------------: |
| 新初始化 projection/adapter |       `3e-5–1e-4` |
| 已完成 distillation 的 neck |            `1e-5` |
| Memory、heads、decoder    |            `5e-6` |
| TinyViT 最後 stage        |            `3e-6` |
| TinyViT 更早 stages       | layer decay `0.8` |
| S4 temporal modules     |          `2.5e-6` |

共同設定：

* AdamW，betas `(0.9, 0.999)`
* BF16 autocast，loss/reduction 使用 FP32
* weight decay `0.1`；bias、BN、LayerNorm、LayerNorm2d 不 decay
* global gradient clip `0.1`
* 每次新增解凍模組後，重建 optimizer/scheduler/DDP
* 每個 stage warm-up 約 5%，再 cosine 到 0.1×
* TinyViT drop-path 先設為 0
* `find_unused_parameters=True`

## Loss 與 rollout 必須怎麼做

基礎 loss直接沿用官方：

[
L_{\text{task}}
=20L_{\text{focal}}
+L_{\text{dice}}
+L_{\text{IoU-L1}}
+L_{\text{presence-CE}}
]

再加入：

[
L=L_{\text{task}}
+\lambda_{\text{img}}L_{\text{image-KD}}
+\lambda_{\text{mem}}L_{\text{memory-KD}}
]

建議起點：

| Stage |  Image KD | Memory KD |
| ----- | --------: | --------: |
| S1    |       1.0 |         0 |
| S2    |   0.5–1.0 |       1.0 |
| S3    | 0.5 → 0.1 | 0.5 → 0.1 |
| S4    |         0 |     0–0.1 |

重要細節：

* GT mask是所有 annotated frames 的 target，不是每一 frame 的輸入。
* GT mask只在 conditioning/correction frame當 prompt。
* 後續 frame寫入 memory 的應是 model prediction，不是 GT mask，否則有 train/inference mismatch。
* Multimask時，focal+dice只監督 segmentation loss最低的 token；IoU heads可全部監督。
* 物件不在畫面時不算 mask/dice/IoU，只算合法的 presence/occlusion loss。
* Teacher與student必須收到完全相同的 frame、augmentation與prompt。
* Teacher用 `inference_mode()`；但 student的 frozen downstream不可用。
* KD應在實際送入 decoder/memory 的 post-neck feature上計算。

S3 的官方 prompt simulation：

* 初始 GT mask 50%、positive point 25%、box 25%
* 最多 2 個 conditioning/correction frames
* 最多 7 correction clicks
* 90% correction從 prediction FP/FN error取點
* 10%直接從 GT取點
* temporal reverse probability `0.5`

可以用 curriculum：S1無 iterative correction；S2先 0–1，再到3；S3才完整7次。[SAM2 training details](https://arxiv.org/html/2408.00714v2)

如果你的實際 deployment 永遠是「第一幀 GT mask → 自動 propagation」，則應提高 mask-prompt比例，例如 70–80%，而不是盲目完全複製官方互動式比例。

## 單一 SA‑V subset 的資料效率

1. **按 video切 train/val/test**，不能按 frame切，避免嚴重 leakage。

2. 建議 sampling mixture：

   * 50% uniform
   * 30% hard masklets
   * 20% small-object、fast-motion、occlusion/reappearance

   官方用 50k most-edited masklets時，SA‑V val約 66.2，明顯高於 50k random的63.7，證明小資料應優先 quality/hardness，而非純隨機。[SAM2 data-quality ablation](https://arxiv.org/html/2408.00714v2)

3. 沒有 edit-count metadata時，用以下 proxy：

   * teacher/student J&F最低
   * boundary F最低
   * disappearance/reappearance
   * mask area劇烈變化
   * identity switch／相似干擾物
   * 小物件與快速運動

   不要100%只訓最高 loss，因為其中會混入 annotation error。

4. 不增加儲存量的 image replay：

   * 20–25% updates：從同一 SA‑V抽 T=1 annotated frame做 point/box segmentation
   * 75–80% updates：T=4/T=8 video clips

   這不能完全取代 SA‑1B 的多樣性，但能減少 decoder/prompt capability遺忘。

5. Temporal sampling使用 stride 1/2/4，並確保你現有16-frame cache保留同一影片的順序、frame index與annotation status。若16張是彼此獨立隨機frame，就不能直接拿來做 memory training。

6. 空間 augmentation必須整段 clip一致；mask resize使用 nearest-neighbor。可用 hflip、degrees 25/shear 20 affine、mild color jitter、grayscale 0.05。不要對每 frame獨立 crop/rotate。

## 4／8 GPU DDP 與 batch size

有效 batch應以 clips計算：

[
B_{\mathrm{global}}
=N_{\mathrm{GPU}}\times B_{\mathrm{clip/GPU}}\times N_{\mathrm{accum}}
]

第一個可靠 baseline先固定 global batch 8：

| GPU         | Micro batch/GPU | Accumulation | Global clips |
| ----------- | --------------: | -----------: | -----------: |
| 4 GPU       |               2 |            1 |            8 |
| 4 GPU，若 OOM |               1 |            2 |            8 |
| 8 GPU       |               1 |            1 |            8 |

穩定後再測 global batch 16：

| GPU   | Micro batch/GPU | Accumulation |
| ----- | --------------: | -----------: |
| 4 GPU |               2 |            2 |
| 8 GPU |               2 |            1 |

從 batch 8增到16時，第一個實驗保持 LR不變。若確認 under-update，再測 (\sqrt{2}) LR scaling；不要直接linear ×2。小資料 fine-tuning中，更大的 batch會降低每 epoch的optimizer updates，不一定提升最終品質。

Gradient accumulation時：

* loss除以 accumulation steps
* 前幾個 microstep使用 DDP `no_sync()`
* scheduler只在 optimizer update後 step
* clipping在完整 accumulated、DDP-reduced gradient後執行

DDP使用 one process/GPU、NCCL、`DistributedSampler.set_epoch()`與 `drop_last=True`。[PyTorch DDP文件](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)

效能上優先確保：

* image encoder對每個 frame只算一次，不要對每個 object/correction重算
* 將 B×T frames一起送進 encoder
* BF16、fused AdamW、SDPA/Flash Attention
* pinned memory、persistent workers
* teacher image feature cache重用
* activation checkpointing只在能把 microbatch 1提升到2時使用
* correctness穩定後，只 compile固定shape的 image encoder；不要先 compile動態 prompt/memory loop

GPU使用率和VRAM占滿不是目標；應比較 clips/sec、frames/sec與 validation J&F/wall-clock。

## TinyViT BatchNorm

原生 Hiera主要是 LayerNorm，因此官方SAM2沒有小batch BN問題；但TinyViT包含 BatchNorm。

預設做法：

* BN running mean/variance全程 frozen
* 第一版連 affine gamma/beta也 frozen
* 每次 `model.train()`後重新把 BN設回 `eval()`
* gradient accumulation不會改善BN統計，因為BN只看當前microbatch
* 不把SyncBN當預設：每rank只有1–2個高度相關clips，且會增加communication

只有在 domain shift非常大時，才單獨測試：

* SyncBN，或
* 訓練結束後用未augmentation的train frames做一次BN recalibration

不要把這個和主實驗一起變動。[PyTorch SyncBatchNorm文件](https://docs.pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html)

## 最小、最有資訊量的 ablation

所有 pilot使用相同資料順序、T=4、global batch 8、2k optimizer updates：

| Run | Encoder          | Memory      | Decoder      | KD              | 用途                    |
| --- | ---------------- | ----------- | ------------ | --------------- | --------------------- |
| A   | last stage train | frozen      | frozen       | 無               | 重現舊方法                 |
| B   | last stage train | frozen      | frozen       | image KD        | 驗證interface anchor    |
| C   | frozen           | train       | heads only   | image+memory KD | 驗證temporal bottleneck |
| D   | train，小LR        | train       | train，小LR    | image+memory KD | 建議主E2E                |
| E   | encoder LoRA     | memory LoRA | frozen/heads | image+memory KD | low-data PEFT         |
| F   | frozen           | train，T=16  | train        | 少量/無            | long-clip refinement  |

我預期 D 的 in-domain上限最高，但E最省參數、最不容易破壞原本能力。

[FS‑SAM2](https://arxiv.org/html/2509.12105v1) 的 low-data消融中，encoder-r4 + memory-r32 LoRA以約0.75M參數取得55.3 mIoU，優於全量更新memory的51.8；再加decoder LoRA沒有提升。它不是autoregressive VOS，因此只能作為PEFT方向證據，但很適合放進你的ablation。[FS-SAM2 code](https://github.com/fornib/FS-SAM2)

更直接的影片證據是 [RobustPVOS/MoGA](https://openaccess.thecvf.com/content/CVPR2026/papers/Lee_Robust_Promptable_Video_Object_Segmentation_CVPR_2026_paper.pdf)：只訓1.1M個memory-aware adapter參數得到71.8 J&F，略勝80.9M參數 full fine-tune的71.5，training memory也從25GB降至22GB；但它專注於影片corruption，且目前尚無公開code，不能直接視為一般SA‑V的唯一最優解。

## 應該怎麼判斷「真的穩定」

每次記錄：

* focal、dice、IoU、presence、image KD、memory KD各自的loss
* valid object-frame normalized loss
* 100與500-step EMA
* 各模組gradient norm與parameter update norm
* grad clip觸發比例
* prompt type、object數、mask area、ignored/absent frame比例
* multimask被選中的token
* 固定validation的 mask-prompt J&F、box mIoU、1/3-click mIoU
* small object、occlusion/reappearance、長時間距離的J&F

若raw loss震盪，但fixed validation J&F上升，就是正常；若tiny-set不能overfit、EMA不降且validation也不動，才是真正pipeline或optimization failure。

最值得優先看的程式與論文是：[SAM2官方訓練碼](https://github.com/facebookresearch/sam2)、[SAM2 v2 paper](https://arxiv.org/html/2408.00714v2)、[EdgeTAM](https://github.com/facebookresearch/EdgeTAM)、[EfficientTAM](https://github.com/yformer/EfficientTAM)、[FS-SAM2](https://github.com/fornib/FS-SAM2)、[XMem training curriculum](https://github.com/hkchengrex/XMem/blob/main/docs/TRAINING.md) 與 [Cutie training curriculum](https://github.com/hkchengrex/Cutie/blob/main/docs/TRAINING.md)。

如果只選一個第一輪 production run，我會選：**S2 T=4 3k steps → S3 T=8 10k steps → S4 T=16 3k steps；global batch 8；encoder `3e-6`、memory/decoder `5e-6`、clip `0.1`、BN frozen、image+memory KD；最後16-frame階段freeze encoder並half LR。**

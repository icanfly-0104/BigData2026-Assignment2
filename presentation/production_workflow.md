# Slide 與 Video 製作流程（學號 315551143）

要繳交的三個檔案 + 一個連結：

| 項目 | 檔名 / 位置 | 狀態 |
|---|---|---|
| 預測結果 | `output_315551143.csv`（repo 根目錄） | ✅ 已產生，官方 evaluate.py 格式檢查通過 |
| 投影片 | `Asg2_315551143.pdf` | ⬜ 依下方步驟 2–3 製作 |
| 影片 | `Asg2_315551143.mp4` | ⬜ 依下方步驟 4–6 錄製 |
| 程式碼 | public GitHub repo（連結要放在投影片第 1 頁） | ⬜ 步驟 1 |

---

## 步驟 1：建立 public GitHub repo（約 10 分鐘，先做，因為投影片要放連結）

1. 到 GitHub 新增一個 **Public** repository，例如 `BigData2026-Assignment2`，不要勾選自動產生 README。
2. 在專案資料夾執行（`.gitignore` 已排除 `data/`、`raw_filtered/`、`work/`、`.venv/`、官方 `assignment2/`）：

```bash
git init
```

```bash
git add README.md run_all.sh .gitignore src figures presentation
```

```bash
git commit -m "Assignment 2: RDL-GNN + LightGBM for GitHub user activity prediction"
```

```bash
git branch -M main
```

```bash
git remote add origin https://github.com/<你的帳號>/BigData2026-Assignment2.git
```

```bash
git push -u origin main
```

3. 用瀏覽器（無痕視窗）打開 repo 網址，確認不登入也看得到，而且 README 有正常顯示。
4. 把網址填進 `slides_content.md` 第 1 頁。
5. 所有 commit 都要在截止時間前完成。

> `presentation/` 可以不放進 repo（講稿是給你自己看的）；不想放的話，在 `git add` 那行把 `presentation` 拿掉。

---

## 步驟 2：用課程模板做投影片（約 1.5–2 小時）

1. 打開 Assignment2.pdf 第 2 頁的 **Assignment2_report_template** 連結 → 「檔案 → 建立副本」，存到自己的 Google Drive。
2. 依照 `presentation/slides_content.md` 一頁一頁填，**總共 10 頁，不能超過**：

| 頁 | 內容 | 要插入的圖（`figures/`） |
|---|---|---|
| 1 | 標題、姓名、學號、**GitHub 連結**、測試週 | — |
| 2 | 任務、資料、regime shift | `a_regime_shift.png` |
| 3 | Q1 Schema graph | `q1_schema_graph.png`（整頁放大） |
| 4 | Q1 Relational entity graph | `q1_relational_entity_graph.png`（整頁放大） |
| 5 | Q2 跨表關係建模 | 可放 framework 圖裁切的 ④b 部分，或純文字＋數字 |
| 6 | Q3 時間資訊建模 | 可放半衰期比較的小表 |
| 7 | Q4 Framework 與 pipeline | `q4_framework.png`（題目要求一定要有 framework figure） |
| 8 | 結果與消融表 | `b_model_comparison.png` + 結果表 |
| 9 | 額外實驗（prob vs rate、半衰期、特徵重要度） | `c_feature_importance.png` + 小表 |
| 10 | 結論與限制 | — |

3. 插入圖片：「插入 → 圖片 → 上傳電腦中的圖片」，選 `figures/` 裡的 PNG。
4. 排版建議：
   - 每頁文字 ≤ 6 個重點，字級 ≥ 14pt；細節放在講稿裡用說的。
   - 表格可以直接在 Google Slides 用「插入 → 表格」重打，比貼圖清楚。
   - 投影片要用**英文**（影片可以用中文）。
   - **數字要跟 `slides_content.md` 完全一致**，不要自己改或四捨五入成別的值（評分會扣「AI 幻覺／不實結果」）。
5. 自我檢查：
   - [ ] 4 個題目（Q1–Q4）都有對應頁面
   - [ ] 第 1 頁有 public GitHub 連結
   - [ ] 第 7 頁有 framework 圖
   - [ ] ≤ 10 頁

## 步驟 3：輸出 PDF

「檔案 → 下載 → PDF 文件」，重新命名為 **`Asg2_315551143.pdf`**，打開確認圖片清楚、沒有被切掉。

---

## 步驟 4：準備講稿與排練（約 1 小時）

1. 把 `presentation/video_script.md` 改成你自己的說話方式（講稿只是草稿；**不能用 AI 語音或 AI 生成影片，會 0 分**）。
2. 時間目標：**9.5–11 分鐘**（規定 8–12 分鐘，留緩衝）。建議分配：

| 頁 | 時間 |
|---|---|
| 1 開場 | 0:30 |
| 2 任務與資料 | 1:10 |
| 3–4 Q1 | 1:20 |
| 5 Q2 | 1:30 |
| 6 Q3 | 1:30 |
| 7 Q4 | 1:30 |
| 8 結果 | 1:10 |
| 9 額外實驗 | 1:10 |
| 10 結論 | 0:50 |

3. 用碼表完整講一遍；超過 11 分鐘就刪第 2、9 頁的細節，**不要刪 Q2–Q4**。
4. 原則：**只有「投影片和影片都有」的內容才會被評分**，所以每一頁的重點都要講到，講到的重點也要在投影片上看得到。

---

## 步驟 5：錄影（Webex 或 Google Meet 擇一）

錄之前：關掉通知、只開投影片；麥克風先試錄 10 秒聽聽看；鏡頭可開可不開。

### 方式 A：Webex（學校帳號）

1. 開啟 Webex → 從自己的個人會議室開始會議（只有你一個人）。
2. 「共用」→ 選投影片視窗，投影片切到「簡報」模式。
3. 按「錄製」→ 選「錄製在此電腦」（存成本機 MP4），或用雲端錄製，結束後再從 Webex 網站下載 MP4。
4. 講完 → 停止錄製 → 結束會議。

### 方式 B：Google Meet

1. 開一個新會議（需要有錄影權限的帳號，例如學校的 Google Workspace 帳號；個人 Gmail 通常不能錄影）。
2. 「分享螢幕」→ 選投影片分頁，切到簡報模式。
3. 「活動 → 錄製」→ 開始錄製。
4. 結束後，錄影檔會存到 Google Drive 的「Meet Recordings」資料夾（MP4），下載下來。

小技巧：可以分段錄，但規定要用 Webex / Meet 錄，所以盡量一次錄完；講錯了停一下重講那一句就好，不需要追求完美。

---

## 步驟 6：檢查影片

- [ ] 長度在 **8–12 分鐘**之間
- [ ] 格式是 **MP4**，檔名 **`Asg2_315551143.mp4`**
- [ ] 從頭到尾聲音清楚、投影片畫面看得清楚
- [ ] Q1–Q4 都有講到，也有講額外實驗（prob vs rate、半衰期、消融）

---

## 步驟 7：上傳 E3 前的最終檢查

- [ ] `output_315551143.csv` → E3（已通過格式檢查，不要用 Excel 開啟後再存檔，避免格式被改）
- [ ] `Asg2_315551143.pdf` → E3
- [ ] `Asg2_315551143.mp4` → E3
- [ ] GitHub repo 是 public，連結在投影片第 1 頁，所有 commit 都在截止前
- [ ] 截止日期：PDF 寫 11/11，但課程 repo 的 README 已改成測試週 11/22–11/29，**請向助教確認目前的繳交期限**

### （可選）用更新的資料重跑

越接近 11/22 重跑，空窗越短，預測可能越準。重跑後 `work/results.csv` 的數字會變，**投影片和講稿的數字也要一起更新**：

```bash
.venv/Scripts/python src/download_filter.py --start 2026-11-18T10 --end <最新的 UTC 小時，如 2026-11-21T10> --out raw_filtered
```

```bash
.venv/Scripts/python assignment2/scripts/preprocess.py --raw raw_filtered --out data
```

```bash
bash run_all.sh 315551143
```

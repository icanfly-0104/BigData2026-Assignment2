# 影片講稿（中文口說，約 10–11 分鐘，對應 slides_content.md 的 10 頁）

> 提醒：規定 AI 生成的影片 0 分，請**自己錄音/露臉**用 Webex 或 Google Meet 錄製。
> 這份是講稿草稿，請改成你自己的說話方式；數字都來自實際執行結果，不要再自行改動。
> 時間分配：Q2–Q4 與實驗佔約 60%。

---

## Slide 1 — 開場（0:00–0:30，約 30 秒）

大家好，我是＿＿＿，學號 315551143。這次作業是預測 GitHub 使用者在 11/22 到 11/29（台灣時間）這一週，會不會做六種活動：開 PR、開 issue、留言、PR review、PR 行內留言，還有 push。
我的方法是 Relational Deep Learning 的時序異質 GNN，再加上一個 point-in-time 的 LightGBM 做 ensemble。程式碼都放在投影片上的這個 public GitHub repo。

## Slide 2 — 任務、資料與最大的困難（0:30–1:40，約 70 秒）

評分是 macro-F1 跟 micro-F1 的平均，測試使用者有 4,465 位。
資料部分，因為課程 repo 改成要自己下載 GH Archive，我寫了一個串流下載器：每個小時的檔案邊下載邊只保留 cohort 使用者的事件，這跟官方 preprocess.py 的 Step 2 是同一個過濾條件，所以最後再用**官方的 preprocess.py** 建出 8 張表，結果完全一樣，但不用存幾百 GB 的原始檔。
資料範圍是 2025/12/1 到 2026/11/18 早上 9 點 UTC，一共 8,458 個小時、約 239 萬筆 cohort 事件。資料結束的時間離測試週開始還有 3 天 6 小時，這段是看不到的空窗。

這頁最重要的是這張圖：每週六種活動的事件數（對數座標）。可以看到分布非常不穩定，push 從五月每週約 8 萬筆，掉到十一月每週只有約 500 筆；其他活動在七、八月幾乎消失、九月初又突然跳回來。這是 GH Archive 收錄量的改變，不是使用者行為改變。所以我整個方法很大一部分是在處理這種 regime shift。

## Slide 3 — Q1 Schema graph（1:40–2:20，約 40 秒）

第一題的 schema graph：8 張表是 8 個節點，20 個 foreign key 是 20 條邊，箭頭從 FK 指向被參照的 PK。紅色是 primary key、藍色是 foreign key、灰色是 row time 欄位。
可以看到 Users 和 Repositories 是兩個 hub：Users 有 8 個 FK 指進來，例如 author、reviewer、pusher、assignee、merged_by；Repositories 有 6 個 repo_id 指進來。另外 PRReviewComments 有一個自我參照 in_reply_to_id，Issues 用 pull_request_id 連到對應的 PR。

## Slide 4 — Q1 Relational entity graph（2:20–3:00，約 40 秒）

這是兩位指定使用者在 9/15 到 9/17（UTC）的 relational entity graph。題目給的是原始 GitHub id，資料經過匿名化，所以我用官方的 code_int 把它轉成資料裡的 id，對應到 user_id 778 和 1025。
這兩天共有 28 個節點、53 條邊。最有意思的是兩個人透過 Repo 2219 連在一起：28983449 對 22996444 開的 PR 287357 做了 review，也留了行內留言。這種「使用者 → review → PR → 作者」的多跳關係，正是 GNN 可以利用的訊號。

## Slide 5 — Q2 跨表關係的建模（3:00–4:30，約 90 秒）

第二題，我怎麼加強跨表關係的建模，主要有兩層。
第一層是主模型 RDL-GNN。每一個預測時間點，我把資料庫轉成一張異質圖：8 種節點，對應 8 張表；20 個 FK 裡我用了 18 個當邊的型別，每一種都加上反向邊。模型是兩層 HeteroConv，每種邊用 GraphSAGE，hidden 128，有 residual 和 LayerNorm。兩層之後，一個使用者的 embedding 不只看到自己的事件，還會看到這些事件連到的 PR、issue、repo，以及在上面互動的其他使用者，例如「別人在我的 PR 上留的 review」。
第二層是跨表 join 的特徵，同時餵給 LightGBM 和 GNN 的 user 節點。例如：我收到的 review 與留言（review → PR → 作者的 join）、我還沒關的 PR、我 merge 過的 PR 數量（維護者訊號）、被指派的 issue；還有 repo 鄰居的特徵：我參與的 repo 裡所有 cohort 使用者最近的活動量、新開的 PR 數（代表 review 機會）、共同作者數。

實驗上，用同一組使用者特徵，LightGBM 是 37.4，加上 FK 圖上的 message passing 變成 39.2，提升 1.8。純用原始圖、完全不給工程特徵的 GNN 也有 32.6，比 copy last week 的 29.8 好。老實說，手工的關聯特徵在 LightGBM 裡效果不穩定，用機率門檻時 +0.9，用最後的比例規則時反而 -0.3，所以關聯資訊主要是靠 GNN 學到的。

## Slide 6 — Q3 時間資訊的建模（4:30–6:00，約 90 秒）

第三題是時間資訊，我做了四件事。
第一是 point-in-time 正確性：在時間 t 的特徵，只能用 row time 早於 cutoff 的資料，cutoff 是 t 減掉 gap；merged_at、closed_at 這些「之後才知道」的欄位，也只有在 cutoff 之前才使用。而且每一筆訓練樣本都模擬了真實測試時的 3 天 6 小時空窗。
第二是多尺度的時序特徵：每一種活動在 1、3、7、14、28、56、112 天和全部時間的次數、最近一次距今多久、半衰期 7 天的指數衰減、過去 4/8/16 週有活動的比例、趨勢，以及幾點、週末的習慣。
第三是 GNN 裡的相對時間編碼：每個事件節點都有 Δt，也就是 cutoff 減 row time，取 log，再加上週期 1、3.5、7、14、28 天的 sin/cos，以及一天中的時間和星期幾。
第四是處理 regime shift：加入全域的活動量特徵、使用者相對於全域的活動量；訓練時用時間衰減權重，半衰期 3 週；最後用比例式的決策規則，下一頁會講。
實驗上，半衰期的比較：不加權 35.0，12 週 35.6，6 週 35.7，3 週最好 36.2。GNN 加上時間編碼是從 38.9 到 39.2。

## Slide 7 — Q4 Framework 與完整流程（6:00–7:30，約 90 秒）

第四題是整體架構，照圖從左上到右下：
一，串流下載 GH Archive，只留 cohort 的事件，再用官方 preprocess.py 建 8 張表。
二，建立樣本：45 個每週的預測時間點，跟測試週一樣對齊在週六 16:00 UTC，從 1/3 到 11/7；每個點的 cutoff 是 t 減 3 天 6 小時，標籤是 t 之後 7 天內有沒有做這六種活動；9,000 位 cohort 使用者，141 個特徵。
三，依時間切分：最後兩週（10/31 和 11/7）當 validation，在測試使用者上評估；訓練的 42 週，標籤都在 validation cutoff 之前結束。
四，A 分支 LightGBM：每種活動一個二元分類器，時間加權，用 validation early stopping，再用全部週數重新訓練。
五，B 分支 RDL-GNN：45 張 snapshot 圖，六個輸出的 multi-task BCE，AdamW、學習率 5e-4、cosine decay、15 個 epoch，3 個 seed 平均，最後也用全部週數重新訓練。
六，ensemble：0.7 乘 LightGBM 加 0.3 乘 GNN，權重在 validation 上選。
七，決策：每種活動預測分數最高的前 q% 為 1，q 是為了 Score 調出來的；輸出 csv，並用官方 evaluate.py 檢查格式。

## Slide 8 — 結果與消融實驗（7:30–8:40，約 70 秒）

這是在測試使用者、兩個 validation 週上的結果，決策規則在其中一週調，在另一週評分，兩個方向取平均，避免太樂觀。
Copy last week 是 29.8。LightGBM 只用時序特徵 37.7，加上關聯特徵 37.4。純圖 GNN 32.6，GNN 加使用者特徵、沒有時間編碼 38.9，有時間編碼 39.2。最後的 ensemble 是 40.3，macro-F1 40.0，micro-F1 40.7。
各活動中 open_issue 和 pr_review 最好，大約 43；pr_review_comment 最難，35.6，因為它最稀少。用官方 evaluate.py 在 11/7 那一週評分，Score 是 39.80%。

## Slide 9 — 額外實驗與分析（8:40–9:50，約 70 秒）

這頁是我覺得最關鍵的額外實驗：決策門檻怎麼定。
一開始我用機率門檻，但發現 validation 週的 cutoff 還包含十月底以前 push 很多的歷史，測試時的 cutoff 已經在低 push 的 regime，所以測試時模型輸出的機率整體下降：push 的平均機率在 validation 是 0.29 和 0.25，測試時只有 0.06。如果沿用機率門檻，只會預測 0.6% 的人會 push，但實際上 validation 週有大約 6% 的人 push。
因為排序在不同 regime 之間比機率穩定，我改成調「預測為 1 的比例」。表上可以看到，LightGBM 從 32.2 升到 37.4，ensemble 從 35.6 升到 40.3。最後的比例是：open_pr 6.6%、open_issue 4.0%、留言 3.8%、review 6.6%、行內留言 1.6%、push 8.4%。
另外，特徵重要度顯示時間衰減和 recency 類的特徵最重要，接著是相對於全域的 rel_push_28d 和全域活躍人數。

## Slide 10 — 結論與限制（9:50–10:40，約 50 秒）

總結：RDL-GNN 加上 point-in-time 使用者特徵是最好的單一模型，39.2；和 LightGBM ensemble 之後是 40.3，相對 copy last week 的 29.8 提升了 10.5 分。
時間的建模影響最大，包括正確的 point-in-time 特徵、模擬真實的空窗、時間加權，以及對 regime shift 穩健的決策規則；關聯的部分主要是 GNN 在 FK 圖上的 message passing 帶來的提升。
限制是新 regime 只有兩個 validation 週，所以權重和比例的調整有雜訊，而且 ensemble 權重也是在這兩週上選的，40.3 會稍微偏樂觀；另外測試週 GH Archive 的收錄狀況也無法預知。
重現方式都寫在 README，謝謝大家。

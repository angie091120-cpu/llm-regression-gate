# 第二標註者說明（NebulaDesk 客服信件分類）

**任務**：你會拿到 70 封寄到一家 B2B SaaS 客服平台「NebulaDesk」支援信箱的信（寄件人是其他公司的客服團隊），請依下面四個類別的定義，替每一封各選一個類別填進 CSV。

這份說明刻意不附任何範例信件，也不會給你任何現成答案，請直接依定義判斷。

## 怎麼填

用 Excel 或任何試算表軟體打開 `annotator2_sheet.csv`，共四欄：

| 欄位 | 說明 |
|------|------|
| `case_id` | 信件編號，請勿更動 |
| `email_body` | 信件全文，請勿更動 |
| `your_label` | 你的判斷，只填 `billing`／`technical`／`account`／`general` 其中一個（小寫英文） |
| `notes` | 備註，可留空 |

- 每封信只能填一個類別，不複選、不留空。
- 不確定的時候，照你自己的判斷選一個，並在 `notes` 寫一句你猶豫的原因。
- 請憑自己的判斷填，不要用 AI 工具代為分類，也不要和別人討論或詢問任何一題的答案。
- 存檔請維持 CSV：Excel 若問要不要換成 xlsx，選保留 CSV；另存新檔時請選「CSV UTF-8（逗號分隔）」，否則中文會變亂碼。
- 70 封大約 25–35 分鐘。填完把 CSV 傳回給找你標註的人即可。
- 不會記錄你的姓名，紀錄上只寫「真人、非本專案成員」。

## 四個類別的定義

以下逐字引用分類任務的原始定義（英文原文，未改寫）：

```
Classify each email into exactly one category:
  - billing: invoices, payments, refunds, plan/pricing changes, proration,
    currency/exchange-rate questions, receipts, purchase orders.
  - technical: bugs, integration/API/webhook problems, outages, performance,
    data sync issues, exports, mobile app issues.
  - account: login/password, roles and permissions, SSO/2FA setup, seat
    management, account transfer/merge/deletion, team member changes.
  - general: anything that isn't clearly billing/technical/account --
    feature requests, sales/pricing inquiries, compliance/SOC2 questions,
    thank-you notes, vague complaints, onboarding/training requests.
```

參考中譯（非正式，判斷一律以上方英文原文為準）：

- **billing**：發票、付款、退款、方案／定價變更、按比例計費、幣別與匯率問題、收據、採購單。
- **technical**：程式錯誤、整合／API／webhook 問題、服務中斷、效能、資料同步問題、資料匯出、行動 App 問題。
- **account**：登入／密碼、角色與權限、SSO／兩階段驗證設定、席次管理、帳號移轉／合併／刪除、團隊成員異動。
- **general**：不明確屬於 billing／technical／account 的其他信——功能需求、業務／報價詢問、法遵／SOC2 問題、道謝信、含糊的抱怨、導入／教育訓練需求。

## 兩類都像的時候

同樣逐字引用原始規則：

```
If an email could plausibly
fit two categories, pick the one that reflects what the customer needs
actioned first (e.g. "I can't log in because my card was declined and my
account got downgraded" is an account-access problem even though billing
caused it, if regaining access is the more urgent ask -- use judgment).
```

參考中譯（非正式）：一封信若兩個類別都說得通，選「客戶最需要先被處理的那件事」所屬的類別。

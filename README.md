# 🔥 AI Sales & Order Manager: BBQ & Charcoal Wholesale

An AI product for a wholesale sales team: it creates **promotions**, reads **customer LPOs**, checks **warehouse stock** and **credit**, creates **orders and invoices**, handles **returns**, and writes the **monthly business plan** from an ML sales forecast.

> ⚠️ **Demo project only.** All products, customers, LPOs and invoices are made-up data.

## 🔀 The flow
```
🎁 Promotions → 📄 Customer LPO → 💲 Price check → 🏭 Stock check → 💳 Credit check → 🧾 Sales Order · Delivery Note · Invoice → 🔄 Returns → 📈 Forecast & Business Plan
```

| Step | What happens |
|---|---|
| 🎁 **Promotions** | Finds slow-moving / overstocked products, calculates the safe discount (margin stays ≥ 15%), AI writes offers in **English and Arabic**. A guardrail blocks any AI offer above the allowed discount |
| 📄 **LPO Reader** | AI reads messy LPOs (forms, emails) into structured data and matches items to product codes |
| 💲 **LPO checks** | Known customer, LPO not expired, not a duplicate, prices match the price list (incl. approved promos) |
| 🏭 **Stock check** | Full stock → reserve · partial → deliver now + back-order · none → suggest an alternative · reorder alerts for December |
| 💳 **Credit check** | Orders over the customer's available credit go on hold for Finance |
| 🧾 **Documents** | Sales Order → Delivery Note → Invoice (LPO number on every document, VAT 5%) + AI customer email |
| 🔄 **Returns** | AI reads return requests; the return policy decides credit, restocking fee or replacement; resellable items go back to stock |
| 📈 **Forecast & Plan** | Random Forest forecast (tested on unseen months) + AI-written December business plan using only real numbers |

## 🛡️ AI engineering details
- **Structured output** with Pydantic schemas (no free-text parsing)
- **Guardrails:** business rules are decided by code, not by the AI
- **Fallback models:** if one Gemini model is busy, the app tries the next one
- **Usage limit** per visitor to protect the API key
- **Model evaluation:** forecast accuracy measured on months the model never saw

## Tech stack
Python · Gemini API · Pydantic · scikit-learn · pandas · Plotly · Streamlit

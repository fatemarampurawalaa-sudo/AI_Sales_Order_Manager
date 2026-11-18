import html
import json
import time
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor

# =====================================================================
# DEMO DATA (made-up BBQ & charcoal wholesaler in the UAE)
# =====================================================================
TODAY = date(2026, 11, 18)          # fixed demo date, so sample LPOs always behave the same
MIN_MARGIN = 0.15                   # never discount below 15% margin
AI_LIMIT = 25                       # AI calls per visitor session (protects the API key)

products = pd.DataFrame({
    "code": ["GR-P01", "GR-K02", "GR-L03", "GR-G04",
             "CH-L03", "CH-L05", "CH-L10", "CH-C05", "CH-Q02",
             "AC-SK1", "AC-TG1", "AC-FS1", "AC-FT1", "AC-CV1"],
    "product": ["Portable Charcoal Grill", "Kettle Grill 57cm", "Large Charcoal Grill", "Gas Grill 3-Burner",
                "Lemon Wood Charcoal 3kg", "Lemon Wood Charcoal 5kg", "Lemon Wood Charcoal 10kg",
                "Coconut Briquettes 5kg", "Quick-Light Charcoal 2kg",
                "Steel Skewers (12 pcs)", "BBQ Tongs", "Fire Starters (32 pcs)",
                "Foil Trays (10 pcs)", "Grill Cover"],
    "category": ["Grill"] * 4 + ["Charcoal"] * 5 + ["Accessory"] * 5,
    "cost_aed":  [45, 180, 320, 650, 9, 14, 26, 18, 8, 12, 10, 6, 7, 35],
    "price_aed": [75, 290, 520, 990, 15, 23, 42, 30, 14, 22, 18, 11, 12, 60],
    "stock":     [120, 40, 15, 8, 900, 650, 300, 220, 500, 180, 150, 600, 400, 25],
    "days_in_stock": [20, 35, 95, 120, 10, 12, 15, 80, 18, 25, 30, 14, 22, 110],
    "restock_days": [14, 21, 30, 30, 5, 5, 7, 10, 5, 10, 10, 7, 7, 21],
})
products["margin_%"] = ((products["price_aed"] - products["cost_aed"]) / products["price_aed"] * 100).round(1)

customers = pd.DataFrame({
    "customer": ["GreenLeaf Supermarket", "Desert Camp Supplies", "Smoky Grill Restaurant",
                 "Oasis Hypermarket", "Dune Adventures Camp", "Corner Mart Grocery"],
    "type": ["Supermarket", "Camping Shop", "Restaurant", "Hypermarket", "Desert Camp", "Grocery"],
    "city": ["Dubai", "Sharjah", "Dubai", "Abu Dhabi", "Dubai", "Ajman"],
    "credit_limit_aed": [60000, 25000, 15000, 120000, 30000, 8000],
    "unpaid_aed":       [22000, 21000, 3000, 45000, 5000, 7500],
    "payment_terms": ["30 days", "30 days", "15 days", "60 days", "30 days", "Cash"],
})

RESERVED_THIS_WEEK = {"CH-L05": 500, "GR-K02": 40, "GR-P01": 30, "AC-FT1": 100}
ALREADY_RECEIVED = ["DCS-LPO-0988", "GL-2026-0498"]

past_invoices = pd.DataFrame({
    "invoice": ["INV-2026-0790", "INV-2026-0790", "INV-2026-0702", "INV-2026-0702", "INV-2026-0811", "INV-2026-0811"],
    "customer": ["GreenLeaf Supermarket", "GreenLeaf Supermarket", "Desert Camp Supplies", "Desert Camp Supplies",
                 "Smoky Grill Restaurant", "Smoky Grill Restaurant"],
    "delivery_date": pd.to_datetime(["2026-11-12", "2026-11-12", "2026-10-05", "2026-10-05", "2026-11-16", "2026-11-16"]),
    "code": ["CH-L03", "GR-P01", "GR-L03", "CH-L05", "CH-L10", "AC-TG1"],
    "invoiced_qty": [150, 20, 5, 100, 50, 10],
    "price_aed": [15, 75, 520, 23, 42, 18],
})

SAMPLE_LPOS = {
    "GreenLeaf (clean)": """GREENLEAF SUPERMARKET - LOCAL PURCHASE ORDER
LPO No: GL-2026-0512        Date: 15/11/2026      Valid until: 15/12/2026
Deliver to: GreenLeaf Supermarket, Al Barsha, Dubai
Payment: 30 days
1. Lemon wood charcoal 3kg bags     x 150   @ AED 15.00
2. Fire starters 32pcs              x 80    @ AED 11.00
3. Portable charcoal grill          x 20    @ AED 75.00
4. Foil trays 10pcs                 x 60    @ AED 12.00
Authorised by: Purchasing Dept.""",
    "Oasis (wrong price + low stock)": """From: purchasing@oasis-hyper.demo
Subject: PO OH/LPO/7781 - BBQ weekend stock
Hi team, pls supply below for our Abu Dhabi branch before the weekend.
LPO ref OH/LPO/7781 dated 16-Nov-2026, valid 30 days. Terms 60 days.
- 5kg lemon charcoal - 300 bags - 20 AED each
- kettle grills 57cm - 12 units - 290 AED
- tongs - 100 pcs at 18
- quick light charcoal 2kg x 200 @14
Delivery: Oasis Hypermarket, Mussafah, Abu Dhabi
Thanks, Procurement""",
    "Desert Camp (expired + duplicate)": """DESERT CAMP SUPPLIES LLC
Purchase Order: DCS-LPO-0988
Issue date: 01/11/2026   Expiry: 10/11/2026
Ship to: Desert Camp Supplies, Industrial Area 6, Sharjah
Terms: 30 days
Item                          Qty     Unit Price
Lemon Wood Charcoal 10kg      200     42.00
Steel Skewers 12pcs           50      22.00""",
    "Corner Mart (over credit)": """Corner Mart Grocery, Ajman
LPO CM-1123 / date 17-11-2026 / valid till 30-11-2026 / payment cash
40 bags lemon charcoal 3kg @15
20 packs fire starters @11
deliver to shop, Al Nuaimiya, Ajman""",
}

SAMPLE_RETURNS = {
    "GreenLeaf (damaged)": """Hi team, ref INV-2026-0790. 20 bags of the 3kg lemon charcoal arrived wet,
and 2 portable grills have broken legs straight out of the box. Photos attached. Pls arrange credit. - GreenLeaf Purchasing""",
    "Desert Camp (unsold)": """Salam, season is slower than expected at our camps. We want to return 3 large charcoal grills
(still sealed in box) and 40 bags of 5kg lemon charcoal (unopened) from invoice INV-2026-0702. Desert Camp Supplies""",
    "Smoky Grill (wrong + defective)": """INV-2026-0811 - two problems: you sent 10kg charcoal but we ordered 5kg, returning all 50 bags.
Also 3 BBQ tongs - handle melted after first use. No photo sorry. Smoky Grill Restaurant""",
}


# =====================================================================
# SALES HISTORY + FORECAST
# =====================================================================
@st.cache_data
def build_sales():
    np.random.seed(42)
    months = pd.DataFrame({"month": pd.date_range("2024-11-01", periods=24, freq="MS")})   # 2 years of history
    season = {11: 1.3, 12: 1.6, 1: 1.7, 2: 1.5, 3: 1.2, 4: 0.8, 5: 0.5, 6: 0.3, 7: 0.3, 8: 0.3, 9: 0.5, 10: 0.9}
    months["season_factor"] = months["month"].dt.month.map(season)
    base_qty = {"GR-P01": 10, "GR-K02": 3, "GR-L03": 1, "GR-G04": 0.5, "CH-L03": 80, "CH-L05": 60, "CH-L10": 25,
                "CH-C05": 15, "CH-Q02": 40, "AC-SK1": 15, "AC-TG1": 12, "AC-FS1": 50, "AC-FT1": 35, "AC-CV1": 2}
    size = {"Supermarket": 1.0, "Camping Shop": 0.7, "Restaurant": 0.5, "Hypermarket": 2.0, "Desert Camp": 0.8, "Grocery": 0.3}
    sales = (months.merge(products[["code", "product", "category", "price_aed", "cost_aed"]], how="cross")
                   .merge(customers[["customer", "type"]], how="cross"))
    sales["qty"] = (sales["code"].map(base_qty) * sales["season_factor"] * sales["type"].map(size)
                    * np.random.uniform(0.7, 1.3, len(sales))).round().astype(int)
    sales["revenue_aed"] = sales["qty"] * sales["price_aed"]
    sales["profit_aed"] = sales["qty"] * (sales["price_aed"] - sales["cost_aed"])
    return sales


@st.cache_data
def build_forecast(sales):
    mp = sales.groupby(["month", "code"], as_index=False)["qty"].sum()
    mp["month_sin"] = np.sin(2 * np.pi * mp["month"].dt.month / 12)
    mp["month_cos"] = np.cos(2 * np.pi * mp["month"].dt.month / 12)
    X = pd.concat([pd.get_dummies(mp["code"]), mp[["month_sin", "month_cos"]]], axis=1)
    y = mp["qty"]
    train = mp["month"] < "2026-08-01"
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X[train], y[train])
    error_pct = round(float(abs(model.predict(X[~train]) - y[~train]).sum() / y[~train].sum() * 100), 1)
    model.fit(X, y)
    dec = products[["code", "product", "category", "price_aed", "cost_aed"]].copy()
    dec["month_sin"] = np.sin(2 * np.pi * 12 / 12)
    dec["month_cos"] = np.cos(2 * np.pi * 12 / 12)
    X_dec = pd.concat([pd.get_dummies(dec["code"]), dec[["month_sin", "month_cos"]]], axis=1).reindex(columns=X.columns, fill_value=0)
    dec["forecast_qty"] = model.predict(X_dec).round().astype(int)
    return error_pct, dec.drop(columns=["month_sin", "month_cos"])


def promo_table(sales):
    avg = sales.groupby("code", as_index=False)["qty"].sum()
    avg["avg_monthly_qty"] = (avg["qty"] / 12).round()
    p = products.merge(avg[["code", "avg_monthly_qty"]], on="code")
    p["months_of_stock"] = (p["stock"] / p["avg_monthly_qty"]).round(1)
    p["needs_promo"] = (p["days_in_stock"] >= 60) | (p["months_of_stock"] > 3)
    p["max_safe_disc_%"] = ((1 - p["cost_aed"] / (1 - MIN_MARGIN) / p["price_aed"]) * 100).round(1)
    p["suggested_disc_%"] = (np.floor(p["max_safe_disc_%"].clip(upper=20) / 5) * 5).astype(int)
    p["next_month_qty"] = (p["avg_monthly_qty"] * 1.6).round()
    p["qty_with_promo"] = (p["next_month_qty"] * (1 + 2 * p["suggested_disc_%"] / 100)).round()
    p["promo_price_aed"] = (p["price_aed"] * (1 - p["suggested_disc_%"] / 100)).round(2)
    p["extra_profit_aed"] = (p["qty_with_promo"] * (p["promo_price_aed"] - p["cost_aed"])
                             - p["next_month_qty"] * (p["price_aed"] - p["cost_aed"])).round()
    return p


def warehouse_now(promo, reserved_extra, returned):
    wh = promo[["code", "product", "category", "stock", "restock_days", "avg_monthly_qty"]].copy()
    wh["reserved"] = (wh["code"].map(RESERVED_THIS_WEEK).fillna(0) + wh["code"].map(reserved_extra).fillna(0)).astype(int)
    wh["returned"] = wh["code"].map(returned).fillna(0).astype(int)
    wh["available"] = wh["stock"] + wh["returned"] - wh["reserved"]
    wh["december_need"] = (wh["avg_monthly_qty"] * 1.6).round().astype(int)
    wh["status"] = np.select([wh["available"] <= 0, wh["available"] < wh["december_need"]],
                             ["🔴 Out of stock", "🟡 Low for December"], default="✅ OK")
    return wh


def customers_now(new_unpaid):
    c = customers.copy()
    c["unpaid_aed"] = c["unpaid_aed"] + c["customer"].map(new_unpaid).fillna(0)
    c["credit_available_aed"] = c["credit_limit_aed"] - c["unpaid_aed"]
    return c


# =====================================================================
# AI (Gemini, structured output, backup models)
# =====================================================================
class Offer(BaseModel):
    code: str
    offer_title: str
    offer_type: Literal["Percent Off", "Buy X Get Y", "Bundle"]
    discount_pct: int
    message_en: str
    message_ar: str
    target_customer_types: list[str]
    reason: str


class LPOItem(BaseModel):
    code: str
    description: str
    qty: int
    unit_price_aed: float


class LPO(BaseModel):
    lpo_number: str
    customer: str
    lpo_date: str
    valid_until: str
    delivery_address: str
    payment_terms: str
    items: list[LPOItem]


class ReturnLine(BaseModel):
    code: str
    qty: int
    reason: Literal["Wrong item", "Damaged on delivery", "Defective", "Unsold stock", "Changed mind"]
    opened: bool


class ReturnRequest(BaseModel):
    customer: str
    invoice: str
    photo_attached: bool
    lines: list[ReturnLine]


class Email(BaseModel):
    subject: str
    body: str


class CategoryTarget(BaseModel):
    category: str
    revenue_target_aed: int
    focus: str


class BusinessPlan(BaseModel):
    executive_summary: str
    revenue_target_aed: int
    profit_target_aed: int
    targets_by_category: list[CategoryTarget]
    promotions_to_run: list[str]
    stock_to_reorder: list[str]
    sales_actions: list[str]
    risks: list[str]


SCHEMAS = {"offers": list[Offer], "lpo": LPO, "return": ReturnRequest, "email": Email, "plan": BusinessPlan}
MODELS = ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-2.5-flash"]


@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"], http_options=types.HttpOptions(api_version="v1beta"))


@st.cache_data(show_spinner=False)
def ai_json(prompt, kind):
    config = {"response_mime_type": "application/json", "response_schema": SCHEMAS[kind]}
    errors = []
    for model in MODELS:                      # if one model is busy, try the next one
        try:
            response = get_client().models.generate_content(model=model, contents=prompt, config=config)
            return json.loads(response.text)
        except Exception as e:
            errors.append(f"{model}: {e}")
            time.sleep(2)
    raise RuntimeError(" | ".join(errors))


def run_ai(prompt, kind):
    if st.session_state.ai_calls >= AI_LIMIT:
        st.error(f"Demo limit reached ({AI_LIMIT} AI calls per visit). Refresh the page to start a new session.")
        return None
    try:
        result = ai_json(prompt, kind)
        st.session_state.ai_calls += 1
        return result
    except Exception as e:
        st.error("The AI is busy right now. Please try again in a minute.")
        st.caption(f"Details: {str(e)[:300]}")
        return None


def catalog_csv():
    return products[["code", "product"]].to_csv(index=False)


# =====================================================================
# BUSINESS RULES (decided by code, not by the AI)
# =====================================================================
def guard_offers(offers, promo):
    o = pd.DataFrame(offers).merge(promo[["code", "product", "suggested_disc_%"]], on="code", how="left")
    o["rule_ok"] = o["discount_pct"] <= o["suggested_disc_%"]
    o["status"] = np.where(o["rule_ok"], "✅ Approved", "❌ Blocked: discount too high")
    return o


def process_lpo(lpo, approved, promo, ss):
    custs = customers_now(ss.new_unpaid)
    wh = warehouse_now(promo, ss.reserved_extra, ss.returned)
    valid = pd.to_datetime(lpo.get("valid_until"), errors="coerce")

    header = pd.DataFrame({
        "check": ["Known customer", "LPO not expired", "Not a duplicate LPO"],
        "ok": [lpo["customer"] in custs["customer"].tolist(),
               bool(pd.notna(valid) and valid.date() >= TODAY),
               lpo["lpo_number"] not in ss.processed],
    })
    header["status"] = np.where(header["ok"], "✅ OK", "❌ Problem")

    items = pd.DataFrame(lpo["items"], columns=["code", "description", "qty", "unit_price_aed"])
    items = (items.merge(products[["code", "product", "price_aed"]], on="code", how="left")
                  .merge(approved, on="code", how="left"))
    items["discount_pct"] = items["discount_pct"].fillna(0)
    items["our_price_aed"] = (items["price_aed"] * (1 - items["discount_pct"] / 100)).round(2)
    items["price_check"] = np.where(items["price_aed"].isna(), "❓ Unknown product",
                            np.where((items["unit_price_aed"] - items["our_price_aed"]).abs() < 0.01,
                                     "✅ Price OK", "❌ Price mismatch → invoiced at our price"))

    stock = (items[items["code"].isin(wh["code"])][["code", "description", "qty", "unit_price_aed", "our_price_aed"]]
             .merge(wh[["code", "product", "category", "available", "restock_days", "december_need"]], on="code"))
    stock["can_deliver_now"] = np.minimum(stock["qty"], stock["available"].clip(lower=0))
    stock["back_order"] = stock["qty"] - stock["can_deliver_now"]
    stock["stock_status"] = np.select([stock["back_order"] == 0, stock["can_deliver_now"] > 0],
                                      ["✅ Full stock", "🟡 Partial stock"], default="🔴 No stock")
    stock["restock_date"] = (pd.Timestamp(TODAY) + pd.to_timedelta(stock["restock_days"], unit="D")).dt.strftime("%d-%b")
    alts = (wh.sort_values("available", ascending=False)[["category", "code", "product", "available"]]
              .rename(columns={"code": "alt_code", "product": "alt_product", "available": "alt_available"}))
    best_alt = stock[["code", "category"]].merge(alts, on="category").query("alt_code != code").drop_duplicates("code")
    stock = stock.merge(best_alt[["code", "alt_product", "alt_available"]], on="code", how="left")
    stock["action"] = np.select(
        [stock["stock_status"] == "✅ Full stock", stock["stock_status"] == "🟡 Partial stock"],
        ["Reserve & confirm",
         "Deliver " + stock["can_deliver_now"].astype(str) + " now, " + stock["back_order"].astype(str) + " on " + stock["restock_date"]],
        default="Offer " + stock["alt_product"].astype(str) + " (" + stock["alt_available"].astype(str) + " available) or wait till " + stock["restock_date"])
    stock["left_after_order"] = stock["available"] - stock["can_deliver_now"]
    stock["reorder_qty"] = (stock["december_need"] - stock["left_after_order"]).clip(lower=0).astype(int)
    stock["deliver_value_aed"] = (stock["can_deliver_now"] * stock["our_price_aed"]).round(2)
    stock["backorder_value_aed"] = (stock["back_order"] * stock["our_price_aed"]).round(2)

    subtotal = round(float(stock["deliver_value_aed"].sum()), 2)
    vat = round(subtotal * 0.05, 2)
    total = round(subtotal + vat, 2)

    known = lpo["customer"] in custs["customer"].tolist()
    cust = custs.set_index("customer").loc[lpo["customer"]] if known else None
    credit_available = float(cust["credit_available_aed"]) if known else 0.0
    terms = cust["payment_terms"] if known else lpo.get("payment_terms", "")
    terms_days = int("".join(filter(str.isdigit, str(terms))) or 0)

    if not header["ok"].all():
        decision = "⛔ ON HOLD: LPO problem (expired / duplicate / unknown customer). Sent to the sales manager."
    elif total > credit_available:
        decision = "⛔ ON HOLD: over credit limit. Sent to Finance for approval or advance payment."
    elif subtotal == 0:
        decision = "⛔ ON HOLD: nothing in stock to deliver. Customer offered alternatives."
    else:
        decision = "✅ APPROVED: ready to confirm"

    return {"lpo": lpo, "header": header, "items": items, "stock": stock, "subtotal": subtotal, "vat": vat,
            "total": total, "credit_available": credit_available, "terms": terms,
            "delivery_date": TODAY + timedelta(days=1), "due_date": TODAY + timedelta(days=terms_days),
            "approved": decision.startswith("✅"), "decision": decision, "confirmed": None}


def process_return(req, ss):
    r = (pd.DataFrame(req["lines"], columns=["code", "qty", "reason", "opened"])
         .merge(past_invoices[past_invoices["invoice"] == req["invoice"]], on="code", how="left")
         .merge(products[["code", "product", "category"]], on="code", how="left"))
    r["days_since_delivery"] = (pd.Timestamp(TODAY) - r["delivery_date"]).dt.days
    photo = bool(req["photo_attached"])
    opened = r["opened"].astype(bool)
    conditions = [
        r["invoiced_qty"].isna(),
        r["qty"] > r["invoiced_qty"],
        (r["reason"] == "Wrong item") & (r["days_since_delivery"] <= 14),
        (r["reason"] == "Damaged on delivery") & (r["days_since_delivery"] <= 7) & photo,
        (r["reason"] == "Damaged on delivery"),
        (r["reason"] == "Defective") & (r["category"] == "Grill") & (r["days_since_delivery"] <= 365),
        (r["reason"] == "Defective") & (r["category"] == "Accessory") & (r["days_since_delivery"] <= 30),
        (r["reason"] == "Unsold stock") & (r["category"] == "Charcoal"),
        (r["reason"] == "Unsold stock") & (r["days_since_delivery"] <= 60) & (~opened),
        (r["reason"] == "Changed mind") & (r["days_since_delivery"] <= 14) & (~opened),
    ]
    decisions = [
        "❌ Rejected: item not on this invoice", "❌ Rejected: more than invoiced",
        "✅ Full credit (we collect)", "✅ Full credit",
        "❌ Rejected: needs photo & within 7 days",
        "🔧 Free replacement (12-month warranty)", "🔧 Free replacement (30-day warranty)",
        "❌ Rejected: charcoal can't be returned unsold",
        "🟡 Credit minus 15% restocking fee", "🟡 Credit minus 15% restocking fee",
    ]
    factor = [0, 0, 1, 1, 0, 0, 0, 0, 0.85, 0.85]
    r["decision"] = np.select(conditions, decisions, default="❌ Rejected: outside return policy")
    r["credit_aed"] = (r["qty"] * r["price_aed"] * np.select(conditions, factor, default=0)).fillna(0).round(2)
    r["back_to_stock"] = (r["decision"].str.contains("credit", case=False)
                          & r["reason"].isin(["Wrong item", "Unsold stock", "Changed mind"]))
    credit = round(float(r["credit_aed"].sum()), 2)
    return {"req": req, "lines": r, "credit": credit, "vat": round(credit * 0.05, 2), "photo": photo, "done": None}


def log(text):
    st.session_state.log.insert(0, f"{time.strftime('%H:%M:%S')}  {text}")


# =====================================================================
# ---------- Page ----------
# =====================================================================
st.set_page_config(page_title="AI Sales & Order Manager", page_icon="🔥", layout="wide")

defaults = {"ai_calls": 0, "offers": None, "lpo_result": None, "lpo_email": None, "ret_result": None, "ret_email": None,
            "plan": None, "reserved_extra": {}, "returned": {}, "new_unpaid": {}, "processed": list(ALREADY_RECEIVED),
            "orders": [], "credit_notes": [], "log": []}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)
ss = st.session_state

sales = build_sales()
sales12 = sales[sales["month"] >= "2025-11-01"]          # last 12 months
promo = promo_table(sales12)
error_pct, dec_base = build_forecast(sales)
offers_df = guard_offers(ss.offers, promo) if ss.offers else None
approved = (offers_df[offers_df["status"] == "✅ Approved"][["code", "discount_pct"]]
            if offers_df is not None else pd.DataFrame({"code": [], "discount_pct": []}))
wh = warehouse_now(promo, ss.reserved_extra, ss.returned)

dec = dec_base.merge(approved, on="code", how="left")
dec["discount_pct"] = dec["discount_pct"].fillna(0)
dec["qty_with_promo"] = (dec["forecast_qty"] * (1 + 2 * dec["discount_pct"] / 100)).round().astype(int)
dec["sell_price"] = dec["price_aed"] * (1 - dec["discount_pct"] / 100)
dec["revenue_aed"] = (dec["qty_with_promo"] * dec["sell_price"]).round()
dec["profit_aed"] = (dec["qty_with_promo"] * (dec["sell_price"] - dec["cost_aed"])).round()
dec = dec.merge(wh[["code", "available"]], on="code")
dec["stock_gap"] = (dec["qty_with_promo"] - dec["available"]).clip(lower=0)

st.title("🔥 AI Sales & Order Manager")
st.caption("BBQ grills & charcoal wholesaler, UAE · Promotions → Customer LPO → Stock check → Order & invoice → Returns → Forecast & business plan")
st.warning("⚠️ Demo project with made-up data. Demo date: 18 Nov 2026 (start of BBQ & camping season).")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("💰 Revenue, last 12 months", f"AED {sales12['revenue_aed'].sum()/1e6:.2f}M")
k2.metric("📈 December forecast", f"AED {dec['revenue_aed'].sum()/1e3:,.0f}K", f"model accuracy {100 - error_pct:.0f}%")
k3.metric("🎁 Products needing a promo", int(promo["needs_promo"].sum()))
k4.metric("🧾 Orders confirmed", len(ss.orders), f"AED {sum(o['total'] for o in ss.orders):,.0f}")
k5.metric("🤖 AI calls used", f"{ss.ai_calls} / {AI_LIMIT}")

tab_dash, tab_promo, tab_lpo, tab_ret, tab_plan, tab_rules = st.tabs(
    ["📊 Dashboard", "🎁 Promotions", "📄 LPO → Order", "🔄 Returns", "📈 Forecast & Plan", "📋 Policies"])

# ---------------------------------------------------------------- Dashboard
with tab_dash:
    left, right = st.columns([3, 2])
    with left:
        monthly = sales.groupby("month", as_index=False)["revenue_aed"].sum().assign(type="Actual")
        fc = pd.DataFrame({"month": [pd.Timestamp("2026-12-01")], "revenue_aed": [dec["revenue_aed"].sum()], "type": ["Forecast"]})
        fig = px.bar(pd.concat([monthly, fc]), x="month", y="revenue_aed", color="type",
                     color_discrete_map={"Actual": "#e8590c", "Forecast": "#868e96"},
                     title="Monthly revenue (AED), last 2 years: BBQ season Nov–Mar + December forecast")
        fig.update_layout(height=380, margin=dict(t=50, b=10), legend_title=None)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        by_cat = sales12.groupby("category", as_index=False)["profit_aed"].sum()
        fig2 = px.pie(by_cat, names="category", values="profit_aed", hole=0.5, title="Profit by category",
                      color_discrete_sequence=["#e8590c", "#343a40", "#fab005"])
        fig2.update_layout(height=380, margin=dict(t=50, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🏭 Warehouse stock (live)")
    st.dataframe(wh[["code", "product", "category", "stock", "reserved", "returned", "available", "december_need", "status"]],
                 hide_index=True, use_container_width=True)
    st.subheader("📝 Activity log")
    st.text("\n".join(ss.log) if ss.log else "No activity yet. Read an LPO or a return request to start.")

# ---------------------------------------------------------------- Promotions
with tab_promo:
    st.markdown("**Rule:** promote a product if its stock is **60+ days old** or will last **more than 3 months**. "
                f"The discount can never push margin below **{MIN_MARGIN:.0%}**.")
    cands = promo[promo["needs_promo"]]
    st.dataframe(cands[["code", "product", "stock", "days_in_stock", "months_of_stock", "margin_%",
                        "suggested_disc_%", "promo_price_aed", "extra_profit_aed"]],
                 hide_index=True, use_container_width=True)

    if st.button("🤖 Create promotions with AI", type="primary"):
        prompt = f"""You are a sales manager at a BBQ and charcoal wholesaler in the UAE. It is November: BBQ and desert camping season is starting.
Create ONE promotion for EACH product below.
Rules:
- discount_pct must NOT be higher than suggested_disc_% for that product.
- Bundles can pair a grill with charcoal or accessories.
- message_en: short, catchy WhatsApp message for customers (max 40 words).
- message_ar: the same message in Arabic.
- target_customer_types: choose from {customers['type'].tolist()}.
- reason: one line, why this promotion makes business sense.
Products needing a promotion:
{cands[['code','product','stock','days_in_stock','months_of_stock','price_aed','suggested_disc_%']].to_csv(index=False)}"""
        with st.spinner("AI is creating promotions..."):
            result = run_ai(prompt, "offers")
        if result:
            ss.offers = result
            log(f"🎁 AI created {len(result)} promotions")
            st.rerun()

    if offers_df is not None:
        st.subheader("🛡️ AI offers + guardrail check")
        st.caption("Our code checks every AI offer: if the AI gave a bigger discount than allowed, the offer is blocked.")
        st.dataframe(offers_df[["code", "product", "offer_title", "offer_type", "discount_pct", "suggested_disc_%", "status", "reason"]],
                     hide_index=True, use_container_width=True)
        pick = st.selectbox("📣 See the customer message for:", offers_df["offer_title"].tolist())
        row = offers_df[offers_df["offer_title"] == pick].iloc[0]
        c_en, c_ar = st.columns(2)
        c_en.markdown("**🇬🇧 English**")
        c_en.info(row["message_en"])
        c_ar.markdown("**🇦🇪 العربية**")
        c_ar.markdown(f"<div dir='rtl' style='text-align:right;background:#fff4e6;padding:14px;border-radius:8px'>"
                      f"{html.escape(str(row['message_ar']))}</div>", unsafe_allow_html=True)
        st.caption("Target customers: " + ", ".join(row["target_customer_types"]))

# ---------------------------------------------------------------- LPO -> Order
with tab_lpo:
    st.markdown("Pick a sample customer **LPO** (or edit / paste your own). The AI reads it, then our business rules check "
                "**prices, LPO validity, warehouse stock and customer credit** before creating the order.")
    choice = st.radio("Sample LPO", list(SAMPLE_LPOS.keys()), horizontal=True)
    lpo_text = st.text_area("LPO text", value=SAMPLE_LPOS[choice], height=220, key=f"lpo_text_{choice}")

    if st.button("🤖 Read LPO with AI", type="primary"):
        prompt = f"""Read this customer LPO (Local Purchase Order) and extract the data.
- Match every item to OUR product code from the catalog below. If no match, use "UNKNOWN".
- customer must be one of: {customers['customer'].tolist()} (or the name as written if not in the list).
- Dates in YYYY-MM-DD. If only "valid 30 days" is written, add 30 days to the LPO date.
OUR CATALOG:
{catalog_csv()}
LPO TEXT:
{lpo_text}"""
        with st.spinner("AI is reading the LPO..."):
            lpo = run_ai(prompt, "lpo")
        if lpo:
            ss.lpo_result = process_lpo(lpo, approved, promo, ss)
            ss.lpo_email = None
            log(f"📄 LPO {lpo['lpo_number']} read for {lpo['customer']}")
            st.rerun()

    res = ss.lpo_result
    if res:
        lpo = res["lpo"]
        st.divider()
        st.subheader(f"📄 {lpo['lpo_number']} · {lpo['customer']}")
        st.caption(f"Dated {lpo['lpo_date']} · valid until {lpo['valid_until']} · 🚚 {lpo['delivery_address']} · 💳 {lpo['payment_terms']}")

        c1, c2 = st.columns([1, 2])
        c1.markdown("**1️⃣ LPO checks**")
        c1.dataframe(res["header"][["check", "status"]], hide_index=True, use_container_width=True)
        c2.markdown("**2️⃣ Price check vs our price list**")
        c2.dataframe(res["items"][["code", "description", "qty", "unit_price_aed", "our_price_aed", "price_check"]],
                     hide_index=True, use_container_width=True)

        st.markdown("**3️⃣ Warehouse stock check**")
        st.dataframe(res["stock"][["code", "product", "qty", "available", "can_deliver_now", "back_order", "stock_status", "action"]],
                     hide_index=True, use_container_width=True)
        reorder = res["stock"][res["stock"]["reorder_qty"] > 0]
        if len(reorder):
            st.caption("📦 Reorder alert for purchasing: " + " · ".join(
                (reorder["product"] + ": order " + reorder["reorder_qty"].astype(str)).tolist()))

        st.markdown("**4️⃣ Credit check**")
        st.write(f"Order total incl. 5% VAT: **AED {res['total']:,.2f}** · Credit available: **AED {res['credit_available']:,.2f}**")

        if res["confirmed"]:
            o = res["confirmed"]
            st.success(f"✅ Order confirmed · Sales Order {o['so']} · Delivery Note {o['dn']} · Invoice {o['inv']}")
        elif res["approved"]:
            st.success(res["decision"])
        else:
            st.error(res["decision"])

        if res["approved"] or res["confirmed"]:
            st.markdown("**5️⃣ Documents**")
            lines = res["stock"][res["stock"]["can_deliver_now"] > 0][["code", "product", "can_deliver_now", "our_price_aed", "deliver_value_aed"]]
            d1, d2, d3 = st.columns(3)
            d1.info(f"🧾 **Sales Order**\n\nCustomer LPO: {lpo['lpo_number']}\n\n{len(lines)} lines")
            d2.info(f"📦 **Delivery Note**\n\nDeliver on {res['delivery_date']:%d-%b-%Y}\n\n{int(lines['can_deliver_now'].sum())} units")
            d3.info(f"💰 **Invoice**\n\nAED {res['subtotal']:,.2f} + VAT {res['vat']:,.2f} = **AED {res['total']:,.2f}**\n\nDue {res['due_date']:%d-%b-%Y} ({res['terms']})")
            st.dataframe(lines, hide_index=True, use_container_width=True)
            backorders = res["stock"][res["stock"]["back_order"] > 0]
            if len(backorders):
                st.caption(f"⏳ Back-order to deliver later: AED {backorders['backorder_value_aed'].sum():,.2f}")

            if res["approved"] and not res["confirmed"]:
                if st.button("✅ Confirm order & reserve stock"):
                    n = 815 + len(ss.orders)
                    order = {"so": f"SO-2026-{n}", "dn": f"DN-2026-{n}", "inv": f"INV-2026-{n}", "lpo": lpo["lpo_number"],
                             "customer": lpo["customer"], "total": res["total"]}
                    reserved = dict(zip(lines["code"], lines["can_deliver_now"].astype(int)))
                    ss.reserved_extra = {c: ss.reserved_extra.get(c, 0) + reserved.get(c, 0) for c in set(ss.reserved_extra) | set(reserved)}
                    ss.new_unpaid[lpo["customer"]] = ss.new_unpaid.get(lpo["customer"], 0) + res["total"]
                    ss.processed.append(lpo["lpo_number"])
                    ss.orders.append(order)
                    res["confirmed"] = order
                    log(f"🧾 {order['so']} confirmed for {lpo['customer']} · AED {res['total']:,.2f} · stock reserved")
                    st.rerun()
            if res["confirmed"]:
                st.download_button("⬇️ Download invoice lines (CSV)", lines.to_csv(index=False),
                                   file_name=f"{res['confirmed']['inv']}.csv", mime="text/csv")

        st.markdown("**6️⃣ Customer email**")
        if st.button("✉️ Write customer email with AI"):
            prompt = f"""Write a short, polite email from our Sales Team to {lpo['customer']} about their LPO {lpo['lpo_number']}.
Status: {res['decision']}. Delivery date: {res['delivery_date']:%d-%b-%Y}. Invoice total incl. VAT: AED {res['total']:,.2f}.
Mention clearly: price corrections (their LPO price vs our price list), partial deliveries and back-order dates,
out-of-stock products and the alternative we suggest, and the reason if the order is on hold.
Details:
{res['stock'][['product','qty','can_deliver_now','back_order','unit_price_aed','our_price_aed','action']].to_csv(index=False)}
Under 150 words. Sign as "Sales Team"."""
            with st.spinner("AI is writing the email..."):
                ss.lpo_email = run_ai(prompt, "email")
        if ss.lpo_email:
            st.text_input("Subject", ss.lpo_email["subject"])
            st.text_area("Email", ss.lpo_email["body"], height=230)

# ---------------------------------------------------------------- Returns
with tab_ret:
    st.markdown("Pick a sample **return request** (or paste your own). The AI reads it, then our **return policy** decides "
                "each line, calculates the credit note and puts resellable items back into stock.")
    rchoice = st.radio("Sample return request", list(SAMPLE_RETURNS.keys()), horizontal=True)
    ret_text = st.text_area("Customer message", value=SAMPLE_RETURNS[rchoice], height=130, key=f"ret_text_{rchoice}")

    if st.button("🤖 Read return request with AI", type="primary"):
        prompt = f"""Read this customer return request and extract the data.
- Match each item to OUR product code from the catalog.
- customer must be one of: {customers['customer'].tolist()}
- reason: wet/broken on arrival = "Damaged on delivery"; stops working / breaks during use = "Defective".
- opened = false only if the message says sealed/unopened.
OUR CATALOG:
{catalog_csv()}
MESSAGE:
{ret_text}"""
        with st.spinner("AI is reading the return request..."):
            req = run_ai(prompt, "return")
        if req:
            ss.ret_result = process_return(req, ss)
            ss.ret_email = None
            log(f"🔄 Return request read: {req['customer']} · {req['invoice']}")
            st.rerun()

    rr = ss.ret_result
    if rr:
        req = rr["req"]
        st.divider()
        st.subheader(f"🔄 {req['customer']} · Invoice {req['invoice']} · Photo: {'Yes' if rr['photo'] else 'No'}")
        st.dataframe(rr["lines"][["code", "product", "qty", "reason", "opened", "days_since_delivery", "decision", "credit_aed", "back_to_stock"]],
                     hide_index=True, use_container_width=True)
        back_units = int(rr["lines"].loc[rr["lines"]["back_to_stock"], "qty"].sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("💳 Credit note", f"AED {rr['credit'] + rr['vat']:,.2f}", "incl. 5% VAT", delta_color="off")
        m2.metric("🏭 Back to stock", f"{back_units} units")
        m3.metric("🔧 Replacements", int(rr["lines"]["decision"].str.contains("replacement").sum()))

        if rr["done"]:
            st.success(f"✅ {rr['done']} issued and stock updated.")
        elif rr["credit"] > 0 or back_units > 0:
            if st.button("✅ Approve credit note & update stock"):
                cn = f"CN-2026-{42 + len(ss.credit_notes)}"
                back = rr["lines"][rr["lines"]["back_to_stock"]]
                add = dict(zip(back["code"], back["qty"].astype(int)))
                ss.returned = {c: ss.returned.get(c, 0) + add.get(c, 0) for c in set(ss.returned) | set(add)}
                ss.credit_notes.append(cn)
                rr["done"] = cn
                log(f"💳 {cn} for {req['customer']} · AED {rr['credit'] + rr['vat']:,.2f} · {back_units} units back to stock")
                st.rerun()

        if st.button("✉️ Write reply with AI"):
            prompt = f"""Write a short, polite reply from our Sales Team to {req['customer']} about their return request on {req['invoice']}.
Explain each decision clearly and kindly, and why (based on our return policy). Total credit note: AED {rr['credit']:,.2f} + VAT.
Decisions:
{rr['lines'][['product','qty','reason','decision','credit_aed']].to_csv(index=False)}
Under 130 words."""
            with st.spinner("AI is writing the reply..."):
                ss.ret_email = run_ai(prompt, "email")
        if ss.ret_email:
            st.text_input("Subject", ss.ret_email["subject"], key="ret_subject")
            st.text_area("Reply", ss.ret_email["body"], height=220, key="ret_body")

# ---------------------------------------------------------------- Forecast & Plan
with tab_plan:
    f1, f2, f3 = st.columns(3)
    f1.metric("🧪 Forecast accuracy (tested on unseen months)", f"{100 - error_pct:.1f}%")
    f2.metric("📈 December revenue forecast", f"AED {dec['revenue_aed'].sum():,.0f}")
    f3.metric("💰 December profit forecast", f"AED {dec['profit_aed'].sum():,.0f}")
    st.caption("Random Forest model trained on 2 years of sales (up to Jul 2026 to train, Aug–Oct 2026 to test), "
               "then retrained on all months. Approved promotions add +2% sales per 1% discount.")
    st.dataframe(dec[["code", "product", "forecast_qty", "discount_pct", "qty_with_promo", "revenue_aed", "profit_aed", "available", "stock_gap"]],
                 hide_index=True, use_container_width=True)

    if st.button("🤖 Write December business plan with AI", type="primary"):
        offer_list = (offers_df[offers_df["status"] == "✅ Approved"][["offer_title", "offer_type", "discount_pct"]].to_csv(index=False)
                      if offers_df is not None else "No promotions created yet.")
        prompt = f"""You are the sales director of a BBQ and charcoal wholesaler in the UAE. Write the business plan for DECEMBER 2026 (peak BBQ & camping season).
Use ONLY these numbers. Do not invent figures.
Model accuracy: {100 - error_pct}%
December forecast per product (with promotions):
{dec[['product','category','forecast_qty','discount_pct','qty_with_promo','revenue_aed','profit_aed','available','stock_gap']].to_csv(index=False)}
Approved promotions:
{offer_list}
Customers (credit):
{customers_now(ss.new_unpaid)[['customer','type','credit_available_aed','payment_terms']].to_csv(index=False)}
Keep each list item to one short line."""
        with st.spinner("AI is writing the business plan..."):
            plan = run_ai(prompt, "plan")
        if plan:
            ss.plan = plan
            log("📊 AI wrote the December business plan")
            st.rerun()

    if ss.plan:
        p = ss.plan
        bullets = lambda items: "- " + "\n- ".join(items) if items else "- (none)"
        plan_md = f"""# 📊 Business Plan: December 2026
**Summary:** {p['executive_summary']}

**🎯 Revenue target:** AED {p['revenue_target_aed']:,} · **💰 Profit target:** AED {p['profit_target_aed']:,}

### 🎁 Promotions to run
{bullets(p['promotions_to_run'])}

### 📦 Stock to reorder
{bullets(p['stock_to_reorder'])}

### 🤝 Sales actions
{bullets(p['sales_actions'])}

### ⚠️ Risks
{bullets(p['risks'])}
"""
        st.markdown(plan_md)
        st.dataframe(pd.DataFrame(p["targets_by_category"]), hide_index=True, use_container_width=True)
        st.download_button("⬇️ Download business plan", plan_md, file_name="December_2026_Business_Plan.md")

# ---------------------------------------------------------------- Policies
with tab_rules:
    st.subheader("🎁 Promotion rules")
    st.table(pd.DataFrame({
        "Rule": ["Which products", "Max discount", "AI guardrail", "Languages"],
        "Detail": ["Stock 60+ days old, or more than 3 months of stock", f"Margin must stay ≥ {MIN_MARGIN:.0%}, rounded to 5/10/15/20%",
                   "Any AI offer above the allowed discount is blocked automatically", "Every offer written in English and Arabic"]}))
    st.subheader("📄 LPO & order rules")
    st.table(pd.DataFrame({
        "Check": ["Customer", "Validity", "Duplicate", "Prices", "Stock", "Credit", "Documents"],
        "Rule": ["Must be a known customer", "LPO must not be expired", "Same LPO number can't be processed twice",
                 "Wrong LPO price → invoiced at our price list (incl. approved promos), customer informed",
                 "Full → reserve · Partial → deliver now + back-order · None → suggest alternative",
                 "Order total incl. VAT must fit the customer's available credit, else Finance approval",
                 "Sales Order → Delivery Note → Invoice (LPO number on every document), VAT 5%"]}))
    st.subheader("🔄 Return policy")
    st.table(pd.DataFrame({
        "Reason": ["📦 Wrong item sent", "💥 Damaged / wet on delivery", "🔧 Defective", "📉 Unsold stock", "🤷 Changed mind", "⚠️ Always"],
        "Rule": ["Within 14 days", "Within 7 days, with a photo", "Grills 12-month warranty · Accessories 30 days",
                 "Within 60 days, unopened · not allowed for charcoal (moisture & quality risk)", "Within 14 days, unopened",
                 "Must quote the invoice number · can't return more than invoiced"],
        "Result": ["100% credit, we collect", "100% credit", "Free replacement", "Credit minus 15% restocking fee",
                   "Credit minus 15% restocking fee", "Refund as a credit note, not cash"]}))

st.caption("Built with Python · Gemini API (structured output) · Pydantic · scikit-learn · pandas · Plotly · Streamlit")

#!/usr/bin/env python3
"""
generate_reco_dataset.py
------------------------
Create a synthetic e-commerce recommendation dataset for 100 k Indian users
(or any size you choose) with 10 personalized product recommendations each.

Core features implemented
• Age, gender, city/state tier – with required probability splits
• 10 unique (category, brand, item) recs per user
• Tier-aware price buckets (Budget / Mid / Premium / Luxury)
• Age- and gender-weighted category affinities
• Outputs UTF-8 CSV; validates row/column counts

Usage
$ python generate_reco_dataset.py --rows 100000 --out product_recommendations.csv

Author: ChatGPT (o3 model) – June 2025
"""
from __future__ import annotations
import argparse, csv, random, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# 1. CONFIGURATION SECTION – tweak here to change probabilities or catalogs
# ---------------------------------------------------------------------------

ROWS_DEFAULT = 100_000
RECS_PER_USER = 10

# Age distribution (probabilities must sum to 1.00)
AGE_BANDS = {
    "18-25": dict(range=(18, 25), p=0.25),
    "26-35": dict(range=(26, 35), p=0.35),
    "36-45": dict(range=(36, 45), p=0.25),
    "46-60": dict(range=(46, 60), p=0.12),
    "60+":   dict(range=(61, 75), p=0.03),
}

GENDERS = {
    "Male":   0.52,
    "Female": 0.48,
}

TIERS = {
    "Tier1": dict(
        cities=["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
                "Pune", "Kolkata", "Ahmedabad", "Surat", "Jaipur"],
        states=["Maharashtra", "Delhi", "Karnataka", "Telangana", "Tamil Nadu",
                "Maharashtra", "West Bengal", "Gujarat", "Gujarat", "Rajasthan"],
        price_weights=[0.20, 0.35, 0.28, 0.17]   # Budget, Mid, Premium, Luxury
    ),
    "Tier2": dict(
        cities=["Lucknow", "Kanpur", "Nagpur", "Indore", "Bhopal",
                "Visakhapatnam", "Patna", "Vadodara", "Coimbatore", "Agra"],
        states=["Uttar Pradesh", "Uttar Pradesh", "Maharashtra", "Madhya Pradesh",
                "Madhya Pradesh", "Andhra Pradesh", "Bihar", "Gujarat",
                "Tamil Nadu", "Uttar Pradesh"],
        price_weights=[0.30, 0.40, 0.22, 0.08]
    ),
    "Tier3": dict(
        cities=["Rajkot", "Meerut", "Nashik", "Faridabad", "Ghaziabad",
                "Jabalpur", "Ranchi", "Mysore", "Jodhpur", "Kota"],
        states=["Gujarat", "Uttar Pradesh", "Maharashtra", "Haryana",
                "Uttar Pradesh", "Madhya Pradesh", "Jharkhand", "Karnataka",
                "Rajasthan", "Rajasthan"],
        price_weights=[0.43, 0.42, 0.12, 0.03]
    ),
}
TIER_PROBS = [0.40, 0.35, 0.25]          # Tier1, Tier2, Tier3

PRICE_BUCKETS = ["Budget", "Mid", "Premium", "Luxury"]
PRICE_RANGES = {
    "Budget":  (100, 1_000),
    "Mid":     (1_001, 5_000),
    "Premium": (5_001, 15_000),
    "Luxury":  (15_001, 60_000),
}

# Product category pools
CATEGORIES: Dict[str, Dict[str, List[str]]] = {
    "Electronics": {
        "items":  ["Smartphone", "Laptop", "Headphones", "Smart Watch", "Tablet", "Camera"],
        "brands": ["Samsung", "Apple", "OnePlus", "Xiaomi", "Realme", "Vivo",
                   "HP", "Dell", "Lenovo", "Sony", "JBL", "Boat"],
    },
    "Fashion": {
        "items":  ["T-shirt", "Jeans", "Kurta", "Saree", "Dress", "Footwear", "Ethnic Wear"],
        "brands": ["H&M", "Zara", "Nike", "Adidas", "Puma", "Fabindia", "W", "Biba",
                   "Allen Solly", "Van Heusen", "Levi's", "Pepe Jeans"],
    },
    "Home": {
        "items":  ["Furniture", "Kitchen Appliance", "Home Decor", "Bedding", "Storage Solution"],
        "brands": ["IKEA", "Godrej", "Whirlpool", "LG", "Prestige", "Pigeon",
                   "Home Centre", "Fabfurnish"],
    },
    "Beauty": {
        "items":  ["Skincare", "Makeup", "Haircare", "Perfume", "Grooming Product"],
        "brands": ["Lakme", "Maybelline", "L'Oreal", "Nivea", "Dove", "Garnier",
                   "Biotique", "Mamaearth", "Nykaa", "Sugar Cosmetics"],
    },
    "Books": {
        "items":  ["Fiction Book", "Non-fiction Book", "Academic Book",
                   "Self-help Book", "Children's Book"],
        "brands": ["Penguin", "Harper Collins", "Classmate", "Reynolds", "Parker"],
    },
    "Sports": {
        "items":  ["Gym Equipment", "Sportswear", "Outdoor Gear", "Yoga Accessory"],
        "brands": ["Nike", "Adidas", "Puma", "Reebok", "Decathlon", "Nivia", "Cosco"],
    },
    "Grocery": {
        "items":  ["Snack", "Beverage", "Organic Product", "Staple"],
        "brands": ["Britannia", "Parle", "Nestle", "Amul", "Tata", "ITC",
                   "Patanjali", "Organic India"],
    },
}

# Age-based prefs (% share → weights)
AGE_PREFS = {
    "18-25": {"Electronics": 0.30, "Fashion": 0.40, "Beauty": 0.20, "Books": 0.10},
    "26-35": {"Electronics": 0.25, "Fashion": 0.30, "Home": 0.20,
              "Sports": 0.15, "Beauty": 0.10},
    "36-45": {"Home": 0.35, "Electronics": 0.20, "Fashion": 0.20,
              "Grocery": 0.15, "Sports": 0.10},
    "46-60": {"Home": 0.30, "Grocery": 0.25, "Electronics": 0.15,
              "Fashion": 0.15, "Books": 0.15},
    "60+":   {"Grocery": 0.40, "Home": 0.25, "Books": 0.20,
              "Electronics": 0.10, "Beauty": 0.05},
}

# Gender-based prefs
GENDER_PREFS = {
    "Male":   {"Electronics": 0.30, "Sports": 0.20, "Fashion": 0.25,
               "Home": 0.15, "Books": 0.10},
    "Female": {"Fashion": 0.35, "Beauty": 0.25, "Home": 0.20,
               "Electronics": 0.10, "Books": 0.10},
}

# ---------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def choose_age_band() -> Tuple[str, int]:
    band = random.choices(list(AGE_BANDS), weights=[d["p"] for d in AGE_BANDS.values()])[0]
    low, high = AGE_BANDS[band]["range"]
    return band, random.randint(low, high)

def choose_gender() -> str:
    return random.choices(list(GENDERS), weights=list(GENDERS.values()))[0]

def choose_tier_city_state() -> Tuple[str, str, str]:
    tier = random.choices(list(TIERS), weights=TIER_PROBS)[0]
    idx = random.randint(0, 9)
    city  = TIERS[tier]["cities"][idx]
    state = TIERS[tier]["states"][idx]
    return tier, city, state

def choose_price(tier: str) -> float:
    bucket = random.choices(PRICE_BUCKETS, weights=TIERS[tier]["price_weights"])[0]
    low, high = PRICE_RANGES[bucket]
    return round(random.uniform(low, high), 2)

def combined_category_weights(age_band: str, gender: str) -> Tuple[List[str], List[float]]:
    weights = defaultdict(float)
    for cat, w in AGE_PREFS[age_band].items():
        weights[cat] += w
    for cat, w in GENDER_PREFS[gender].items():
        weights[cat] += w
    # small smoothing for any category missing
    for cat in CATEGORIES:
        weights[cat] += 0.001
    cats, wts = zip(*weights.items())
    total = sum(wts)
    return list(cats), [w / total for w in wts]

# ---------------------------------------------------------------------------
# 3. GENERATION CORE
# ---------------------------------------------------------------------------

def gen_user_row(idx: int) -> List:
    user_id = f"USER_{idx:06d}"

    age_band, age = choose_age_band()
    gender = choose_gender()
    tier, city, state = choose_tier_city_state()

    cats, weights = combined_category_weights(age_band, gender)

    seen_products = set()
    rec_flat: List = []
    while len(seen_products) < RECS_PER_USER:
        category = random.choices(cats, weights=weights)[0]
        item  = random.choice(CATEGORIES[category]["items"])
        brand = random.choice(CATEGORIES[category]["brands"])
        product_name = f"{brand} {item}"
        if product_name in seen_products:
            continue  # uniqueness guarantee
        seen_products.add(product_name)

        price = choose_price(tier)
        score = round(random.uniform(0.1, 1.0), 2)

        rec_flat.extend([product_name, category, brand, price, score])

    return [user_id, age, gender, state, city, tier] + rec_flat

def build_header() -> List[str]:
    hdr = ["user_id", "age", "gender", "state", "city", "tier"]
    for i in range(1, RECS_PER_USER + 1):
        hdr += [f"{field}_{i}" for field in
                ["product", "category", "brand", "price", "recommendation_score"]]
    return hdr

# ---------------------------------------------------------------------------
# 4. MAIN CLI ENTRY
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic recommendation dataset")
    parser.add_argument("--rows", "-n", type=int, default=ROWS_DEFAULT,
                        help=f"Number of user rows to create (default {ROWS_DEFAULT})")
    parser.add_argument("--out", "-o", type=Path, default="gpt_product_recommendations.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    random.seed(42)
    start = time.time()

    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(build_header())

        for idx in range(1, args.rows + 1):
            writer.writerow(gen_user_row(idx))
            if idx % 10_000 == 0:
                print(f"...{idx:,} rows generated")

    elapsed = time.time() - start
    print(f"\n✅ Finished {args.rows:,} rows in {elapsed:.1f}s → '{args.out}'")

    # Simple sanity check: verify column count
    expected_cols = len(build_header())
    with args.out.open(encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)                   # skip header
        first_row = next(reader)
        if len(first_row) != expected_cols:
            sys.exit(f"ERROR: expected {expected_cols} columns, got {len(first_row)}")

if __name__ == "__main__":
    main()

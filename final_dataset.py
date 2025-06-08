import pandas as pd
import random
from pathlib import Path
# import ace_tools as tools

# ---------- CONFIGURATION ----------
dataset_sizes = [1000, 10000, 100000]  # Row counts to generate
output_dir = Path("./data")
output_dir.mkdir(exist_ok=True)

# Cities (tier-1, tier-2 mix)
cities = [
    "Ahmedabad", "Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai",
    "Pune", "Jaipur", "Chandigarh", "Lucknow", "Bhopal", "Indore", "Coimbatore",
    "Nagpur", "Visakhapatnam", "Vadodara", "Surat", "Patna", "Thiruvananthapuram"
]

# Product catalog (10 distinct product categories)
product_catalog = [
    {
        "product_category": "Smartphone",
        "category": "Electronics",
        "brands": ["Apple", "Samsung", "Xiaomi", "OnePlus", "Realme"],
        "cost_range": (15000, 100000),
    },
    {
        "product_category": "Laptop",
        "category": "Electronics",
        "brands": ["Apple", "Dell", "HP", "Lenovo", "Asus"],
        "cost_range": (30000, 150000),
    },
    {
        "product_category": "Smartwatch",
        "category": "Electronics",
        "brands": ["Apple", "Samsung", "Fitbit", "Noise"],
        "cost_range": (2000, 50000),
    },
    {
        "product_category": "Earbuds",
        "category": "Electronics",
        "brands": ["boAt", "JBL", "Sony", "Apple"],
        "cost_range": (1000, 20000),
    },
    {
        "product_category": "T-shirt",
        "category": "Fashion",
        "brands": ["Nike", "Adidas", "H&M", "Uniqlo", "Puma"],
        "cost_range": (300, 2000),
    },
    {
        "product_category": "Jeans",
        "category": "Fashion",
        "brands": ["Levi's", "Wrangler", "Lee", "Spykar"],
        "cost_range": (800, 5000),
    },
    {
        "product_category": "Sneakers",
        "category": "Fashion",
        "brands": ["Nike", "Adidas", "Puma", "Reebok"],
        "cost_range": (2000, 12000),
    },
    {
        "product_category": "Mixer Grinder",
        "category": "Home & Kitchen",
        "brands": ["Philips", "Prestige", "Bajaj", "Butterfly"],
        "cost_range": (1500, 10000),
    },
    {
        "product_category": "Yoga Mat",
        "category": "Sports & Outdoors",
        "brands": ["Nivia", "Decathlon", "Domyos", "Boldfit"],
        "cost_range": (300, 3000),
    },
    {
        "product_category": "Face Wash",
        "category": "Beauty & Personal Care",
        "brands": ["Himalaya", "Mamaearth", "Ponds", "Cetaphil"],
        "cost_range": (100, 800),
    },
]

def generate_record(user_id: str, product_id: int) -> dict:
    """Generate a single row adhering to the schema."""
    # Demographics
    age = random.randint(18, 60)
    gender = random.choice(["M", "F"])
    location = random.choice(cities)
    
    # Product details
    item = random.choice(product_catalog)
    brand = random.choice(item["brands"])
    prod_cat = item["product_category"]
    category = item["category"]
    cost = random.randint(*item["cost_range"])
    
    product_name = f"{brand} {prod_cat}"
    
    return {
        "user_id": user_id,
        "Age": age,
        "Gender": gender,
        "Location": location,
        "product_id": product_id,
        "Product": product_name,
        "Brand": brand,
        "Gender_category": gender,
        "Brand_category": brand,
        "product_category": prod_cat,
        "category": category,
        "cost": f"{cost} INR",
    }

def build_dataset(n_rows: int, seed: int = None) -> pd.DataFrame:
    """Create a DataFrame with `n_rows` synthetic recommendation-system rows."""
    if seed is not None:
        random.seed(seed)
    
    # Let each user have up to 10 products on average
    # Define unique user count
    approx_products_per_user = 10
    n_users = max(1, n_rows // approx_products_per_user)
    
    # Pre-generate user IDs
    user_ids = [f"{uid:06d}" for uid in range(1, n_users + 1)]
    
    records = []
    for pid in range(1, n_rows + 1):
        user_id = random.choice(user_ids)  # repeat users across products
        record = generate_record(user_id, product_id=pid + 500)  # +500 to mimic realistic IDs
        records.append(record)
    
    df = pd.DataFrame(records, columns=[
        "user_id", "Age", "Gender", "Location", "product_id",
        "Product", "Brand", "Gender_category", "Brand_category",
        "product_category", "category", "cost"
    ])
    return df

# Generate and save datasets
file_links = {}
for size in dataset_sizes:
    df = build_dataset(size, seed=42)  # deterministic seed for reproducibility
    filename = f"synthetic_reco_dataset_{size}.csv"
    path = output_dir / filename
    df.to_csv(path, index=False)
    file_links[size] = path
    
    # # Display a small preview for each size (first 5 rows)
    # tools.display_dataframe_to_user(
    #     name=f"{size}-row Dataset Preview",
    #     dataframe=df.head(5)
    # )
    
# Return the paths so they can be used as download links
file_links
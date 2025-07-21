import os
import re
from io import BytesIO
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from PIL import Image

# -----------------------------------------------------------------------------
# --- CONFIG & CONSTANTS -------------------------------------------------------
# -----------------------------------------------------------------------------
API_BASE_URL = "https://starbucks-coffee-db2.p.rapidapi.com/api/recipes"
APP_NAME = "Starbucks® Recipe Explorer"

# Retrieve the RapidAPI key from environment variable or Streamlit secrets
RAPIDAPI_KEY = (
    os.getenv("RapidAPI") if os.getenv("RapidAPI") else st.secrets["rapidapi"]["key"]
)

HEADERS = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": "starbucks-coffee-db2.p.rapidapi.com",
}

# Regex helpers ---------------------------------------------------------------
# NOTE: inline flags like (?i) cannot appear mid‑pattern in Python ≥3.13, so we
# convert the string to lowercase before applying these regexes instead of
# embedding the flag.
UNITS_PATTERN = r"(?:oz|ounce(?:s)?|cup(?:s)?|tbsp|tablespoon(?:s)?|tsp|teaspoon(?:s)?|shot(?:s)?|ml|milliliter(?:s)?|l|liter(?:s)?|g|gram(?:s)?|kg|pound(?:s)?|lb|dash(?:es)?|pinch(?:es)?|slice(?:s)?|part(?:s)?)"
AMOUNT_PATTERN = rf"^\s*([\d\.\/]+\s*)?{UNITS_PATTERN}?\s*(of\s+)?"

# -----------------------------------------------------------------------------
# --- DATA ACCESS LAYER --------------------------------------------------------
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Fetching recipes…", ttl=60 * 60)
def fetch_recipes() -> pd.DataFrame:
    try:
        response = requests.get(API_BASE_URL, headers=HEADERS, timeout=30)
        response.raise_for_status()
        raw_data = response.json()
    except requests.exceptions.RequestException:
        st.error("Error fetching data. Check your API key or try again later.")
        st.stop()
    except ValueError:
        st.error("Invalid response format. Please contact support.")
        st.stop()

    df = pd.json_normalize(raw_data)

    required_cols = [
        "name",
        "category",
        "description",
        "datePublished",
        "image",
        "recipeIngredient",
        "recipeInstructions",
        "prepTime",
        "totalTime",
        "recipeYield",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    df["datePublished"] = pd.to_datetime(df["datePublished"], errors="coerce")
    return df


def load_image(url: str) -> Optional[Image.Image]:
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))
    except Exception:
        return None


# -----------------------------------------------------------------------------
# --- UTILS -------------------------------------------------------------------
# -----------------------------------------------------------------------------

def clean_ingredient_list(li: List[str] | None) -> List[str]:
    if not isinstance(li, list):
        return []
    return [item.strip() for item in li if isinstance(item, str) and item.strip()]


def canonicalize_ingredient(raw: str) -> str:
    """Return a simplified token for dedup/search (e.g., ‘1 cup milk’ → ‘milk’)."""
    token = re.sub(AMOUNT_PATTERN, "", raw.lower()).strip()
    token = re.sub(r"[®©™]", "", token)  # remove trademark symbols
    token = re.sub(r"\s+", " ", token)
    return token


def clean_instruction_list(li) -> List[str]:
    if not isinstance(li, list):
        return []
    steps = []
    for step in li:
        if isinstance(step, str):
            steps.append(step.strip())
        elif isinstance(step, dict):
            text = step.get("text") or ""
            steps.append(str(text).strip())
    return [s for s in steps if s]


# -----------------------------------------------------------------------------
# --- UI LAYER ----------------------------------------------------------------
# -----------------------------------------------------------------------------
st.set_page_config(page_title=APP_NAME, page_icon="☕", layout="wide")

st.title(APP_NAME)

df_recipes = fetch_recipes()

# Sidebar – Filters -----------------------------------------------------------
st.sidebar.header("Filters")
search_query = st.sidebar.text_input("Search by name")

all_categories = sorted([c for c in df_recipes["category"].dropna().unique() if c])

default_cat = [c for c in all_categories if c.upper() == "ICED BEVERAGES"] or all_categories
selected_categories = st.sidebar.multiselect(
    "Category (default: Iced Beverages)",
    options=all_categories,
    default=default_cat,
)

# Build canonical ingredient universe ----------------------------------------
canon_to_display = {}
for li in df_recipes["recipeIngredient"]:
    for raw in clean_ingredient_list(li):
        canon = canonicalize_ingredient(raw)
        canon_to_display[canon] = canon.capitalize()

ingredient_options = sorted(set(canon_to_display.values()))

selected_ingredients_display = st.sidebar.multiselect(
    "Ingredients you have (optional)",
    options=ingredient_options,
    help="Pick one or more ingredients to show recipes that use ALL of them.",
)

selected_ingredients = [s.lower() for s in selected_ingredients_display]

# Apply filters ---------------------------------------------------------------
filtered = df_recipes.copy()

if search_query:
    filtered = filtered[filtered["name"].str.contains(search_query, case=False, na=False)]
if selected_categories:
    filtered = filtered[filtered["category"].isin(selected_categories)]

if selected_ingredients:
    def has_all_ings(row) -> bool:
        recipe_canons = [canonicalize_ingredient(i) for i in clean_ingredient_list(row["recipeIngredient"])]
        return all(any(sel in can for can in recipe_canons) for sel in selected_ingredients)

    filtered = filtered[filtered.apply(has_all_ings, axis=1)]

st.markdown(f"**{len(filtered)}** recipes found.")

# Display recipes in grid -----------------------------------------------------
cols = st.columns(3)
for idx, row in filtered.iterrows():
    col = cols[idx % 3]
    with col:
        st.subheader(row["name"] or "Unknown")
        img = load_image(row.get("image"))
        if img is not None:
            st.image(img, use_container_width=True)
        st.caption(row.get("description") or "No description available.")
        with st.expander("Details"):
            st.write("**Category**:", row.get("category"))
            st.write("**Prep Time**:", row.get("prepTime"))
            st.write("**Total Time**:", row.get("totalTime"))
            st.write("**Yield**:", row.get("recipeYield"))

            ingredients = clean_ingredient_list(row.get("recipeIngredient"))
            if ingredients:
                st.markdown("### Ingredients")
                st.markdown("\n".join(f"- {ing}" for ing in ingredients))

            steps = clean_instruction_list(row.get("recipeInstructions"))
            if steps:
                st.markdown("### Instructions")
                st.markdown("\n".join(f"{i+1}. {step}" for i, step in enumerate(steps)))

# -----------------------------------------------------------------------------
# --- VISUAL: Ingredient Count Bar Chart --------------------------------------
# -----------------------------------------------------------------------------
if not filtered.empty:
    filtered["ingredient_count"] = filtered["recipeIngredient"].apply(
        lambda x: len(clean_ingredient_list(x))
    )
    top_n = filtered.sort_values("ingredient_count", ascending=False).head(10)

    st.markdown("## Ingredient Count for Top Recipes (Current Filter)")
    fig, ax = plt.subplots()
    ax.barh(top_n["name"], top_n["ingredient_count"])
    ax.set_xlabel("Number of Ingredients")
    ax.set_ylabel("Recipe")
    ax.invert_yaxis()
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# --- DATA DOWNLOAD -----------------------------------------------------------
# -----------------------------------------------------------------------------
with st.expander("📥 Download filtered data"):
    csv = filtered.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="starbucks_recipes.csv",
        mime="text/csv",
    )
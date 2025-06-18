import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import joblib
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import math
from folium import IFrame
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data
def load_data(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)
    if 'is_actual' in df.columns:
        df = df[df['is_actual'] == 1]
    return df

@st.cache_resource
def load_model_components(path: str):
    model = joblib.load(os.path.join(path, "model.pkl"))
    scaler = joblib.load(os.path.join(path, "scaler.pkl"))
    encoders = joblib.load(os.path.join(path, "encoders.pkl"))
    return model, scaler, encoders

def preprocess_input(X, scaler, encoders):
    for col, le in encoders.items():
        X[col] = le.transform(X[col].astype(str))
    num_cols = X.select_dtypes(include='number').columns
    X[num_cols] = scaler.transform(X[num_cols])
    return X

@st.cache_data
def preprocess_features(X: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([full_df.drop(columns=['price_per_sqm'], errors='ignore'), X], ignore_index=True)
    combined = pd.get_dummies(combined,
                              columns=[col for col in ['object_type', 'listing_type', 'city'] if col in combined.columns],
                              drop_first=True)
    X_enc = combined.tail(1)
    return X_enc.select_dtypes(include=[np.number])

def main():
    st.set_page_config(page_title="CRE Price Explorer", layout="wide")
    st.title("Commercial Real Estate Price Explorer & Predictor")

    tab_map, general_charts, rent_charts, sales_charts = st.tabs(["📍 Map & Prediction", "📊 Relevant general visualisation",
                                                                 "📊 Rental listings", "📊 Sales listings"])

    df = load_data('sample_2025.csv')
    if df.empty:
        st.error("⚠️ После фильтрации нет ни одной записи. Проверь CSV и колонку is_actual.")
        return

    month_num = {1: "January",
                 2: "February",
                 3: "March",
                 4: "April",
                 5: "May",
                 6: "June",
                 7: "July",
                 8: "August",
                 9: "September",
                 10: "October",
                 11: "November",
                 12: "December"}

    with tab_map:

        default_tx = ['Сдам']
        default_prop = ['Торговое помещение']
        default_metro = 'All'
        default_area_from = int(df['total_area'].min())
        default_area_to = int(df['total_area'].max())
        default_price_from = int(df['price'].min())
        default_price_to = int(df['price'].max())
        default_year = 2025
        default_month = 5

        with st.sidebar.form(key="map_form"):
            st.header("📊 Map & Data Filters")

            ru_to_en = {"Сдам": "Rent", "Продам": "Sale"}
            transaction_types = df['listing_type'].unique().tolist()
            st.markdown("**Transaction type:**")
            selected_tx = []
            for t in transaction_types:
                if st.checkbox(ru_to_en[t], value=True):
                    selected_tx.append(t)

            st.markdown("**Property type:**")
            property_types = df["object_type"].unique().tolist()
            selected_prop = []
            if st.checkbox('Office space', value=True):
                selected_prop.append('Офисное помещение')
            if st.checkbox('Retail space', value=True):
                selected_prop.append('Торговое помещение')
            if st.checkbox('Mixed-use premises', value=True):
                selected_prop.append('Помещение свободного назначения')
            if st.checkbox('Warehouse', value=True):
                selected_prop.append('Складское помещение')
            if st.checkbox('Industrial facility', value=True):
                selected_prop.append('Производственное помещение')

            st.markdown("**Metro Access:**")
            selected_metro = []
            if st.checkbox("All", value=True):
                selected_metro.append('All')
            if st.checkbox("Yes", value=True):
                selected_metro.append('Yes')
            if st.checkbox("No", value=True):
                selected_metro.append('No')

            min_area, max_area = int(df['total_area'].min()), int(df['total_area'].max())
            area_from = st.number_input(
                "Area from (m²):", min_value=min_area, max_value=max_area,
                value=min_area, step=1
            )
            area_to = st.number_input(
                "Area to (m²):", min_value=min_area, max_value=max_area,
                value=max_area, step=1
            )

            min_price, max_price = int(df['price'].min()), int(df['price'].max())
            price_from = st.number_input(
                "Price from (₽):", min_value=min_price, max_value=max_price,
                value=min_price, step=10000, format="%d"
            )
            price_to = st.number_input(
                "Price to (₽):", min_value=min_price, max_value=max_price,
                value=max_price, step=10000, format="%d"
            )

            st.markdown("**Year:**")
            years = sorted(df['year'].dropna().unique())
            selected_year = []
            for y in years:
                if st.checkbox(str(y), value=True):
                    selected_year.append(y)

            months = sorted(df['month'].dropna().unique())
            st.markdown("**Month:**")
            selected_month = []

            for m in months:
                if st.checkbox(month_num[m], value=True):
                    selected_month.append(m)

            submit_button = st.form_submit_button(label="Search")

        if submit_button:
            tx, prop, metro, af, at, pf, pt, year, month = (
                selected_tx, selected_prop,
                selected_metro, area_from, area_to,
                price_from, price_from,
                selected_year, selected_month
            )
        else:
            tx, prop, metro, af, at, pf, pt, year, month = (
                default_tx, default_prop,
                default_metro, default_area_from, default_area_to,
                default_price_from, default_price_to,
                default_year, default_month
            )

        filt = df[df['listing_type'].isin(tx) &
                  df['object_type'].isin(prop)]
        if metro != 'All':
            filt = filt[filt['metro_only'].notna()] if metro == 'Yes' else filt[filt['metro_only'].isna()]
        filt = filt[(filt['total_area'] >= area_from) & (filt['total_area'] <= area_to)]
        filt = filt[(filt['price'] >= price_from) & (filt['price'] <= price_to)]
        filt = filt[filt['year'].isin(selected_year)]
        filt = filt[filt['month'].isin(selected_month)]
        tx_encoder = {"Сдам": "rent", "Продам": "sale"}
        if filt.empty:
            st.warning("🚫 По текущим фильтрам ничего не найдено.")
        else:
            m = folium.Map(location=[58.67694, 67.58789], zoom_start=4)
            heat_data = filt[['coords_lat', 'coords_lng', 'price_per_sqm']].dropna().values.tolist()
            HeatMap(heat_data, radius=15).add_to(m)
            markers = folium.FeatureGroup(name="Listings")

            for _, r in filt.iterrows():
                lat, lng = r.coords_lat, r.coords_lng
                popup_html = ""
                if pd.notna(lat) and pd.notna(lng):
                    popup_html += f"<b>{r.title}</b><br>"
                    popup_html += f"<b>Price:</b> {r.price:,.0f} ₽<br>"
                    popup_html += f"<b>Area:</b> {r.total_area:.1f} m²<br>"
                    popup_html += f"<b>Transaction type:</b> {tx_encoder[r.listing_type].capitalize()}<br>"

                    if pd.notna(r.url) and r.url:
                        popup_html += f"<a href='{r.url}' target='_blank'>View listing</a><br>"

                    html_wrapped = f"""
                            <div style="width: 250px; height: 180px;">
                                {popup_html}
                            </div>
                            """

                    iframe = IFrame(html=html_wrapped, width=260, height=190)
                    popup = folium.Popup(iframe, max_width=300)

                    folium.Marker(
                        location=[lat, lng],
                        popup=popup,
                        icon=folium.Icon(color='blue', icon='info-sign')
                    ).add_to(markers)

            m.add_child(markers)
            folium.LayerControl().add_to(m)

            map_data = st_folium(m, width=1200, height=900)


        prop_opts = df['object_type'].unique().tolist()
        tx_opts = df['listing_type'].unique().tolist()
        city_opts = df['city1'].dropna().unique().tolist()

        months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        month_names = [month_num[m] for m in months]

        years = list(map(int, sorted(df['year'].dropna().unique())))

        default_pred_prop = ['Office space']
        default_pred_build = ["Mall"]
        default_pred_area = 100
        default_pred_metro = 'Yes'
        default_pred_tx = "Сдам"
        default_pred_floor = 1
        default_pred_total_floors = 1
        default_pred_month = 1
        default_pred_year = 2025
        default_pred_model = "KNN"

        st.subheader("🤖 Price Prediction")

        st.info("🔍 Click on the map to select a location for prediction.")

        if map_data and map_data.get("last_clicked"):
            lat = map_data["last_clicked"]["lat"]
            lng = map_data["last_clicked"]["lng"]
            st.markdown(
                f"<div style='font-size:18px; font-weight:bold; color:#33cc33;'>📍 Forecast location: {lat:.5f}, {lng:.5f}</div>",
                unsafe_allow_html=True)

        with st.form(key="predict_form"):
            col1, col2 = st.columns(2)
            with col1:

                st.markdown("**Property type:**")
                prop_options_en = [
                    "Office space",
                    "Retail space",
                    "Mixed-use premises",
                    "Warehouse",
                    "Industrial facility",
                ]
                en_to_ru = {
                    "Office space": "Офисное помещение",
                    "Retail space": "Торговое помещение",
                    "Mixed-use premises": "Помещение свободного назначения",
                    "Warehouse": "Складское помещение",
                    "Industrial facility": "Производственное помещение",
                }

                pred_prop_choice = st.radio(
                    "Select property type:",
                    options=prop_options_en
                )

                selected_pred_prop = en_to_ru[pred_prop_choice]
                options = ["Rent", "Sale"]

                en_to_ru = {"Rent": "Сдам", "Sale": "Продам"}
                default_ru = en_to_ru.get(default_pred_tx, options[0])

                pred_tx_choice = st.selectbox(
                    "Select transaction type:",
                    options=options,
                    index=options.index(default_ru)
                )

                selected_pred_tx = pred_tx_choice


                total_area_input = st.number_input(
                    "**Area (m²):**", min_value=1, value=default_pred_area
                )

                floor_input = st.number_input(
                    "**Particular floor:**", min_value=0, value=default_pred_floor
                )

                total_floors_input = st.number_input(
                    "**Total floors:**", min_value=0, value=default_pred_total_floors
                )

                month_input = st.selectbox(
                    "**Month:**", month_names, index=months.index(default_pred_month)
                )

                year_input = st.selectbox(
                    "**Year:**", years, index=years.index(default_pred_year)
                )

                model_choice = st.selectbox("**Select model:**", [
                    "KNN", "XGBoost"
                ])

            with col2:
                st.markdown("**Building type:**")
                build_options_en = [
                    "Residential building",
                    "Business centre",
                    "Administrative building",
                    "Free purpose premises",
                    "Trade and office complex",
                    "Multifunctional complex",
                    "Office building",
                    "Mansion"
                ]

                en_to_ru_build = {
                    "Residential building": "Жилой дом",
                    "Business centre": "Бизнес-центр",
                    "Administrative building": "Административное здание",
                    "Free purpose premises": "Помещение свободного назначения",
                    "Trade and office complex": "Торгово-офисный комплекс",
                    "Multifunctional complex": "Многофункциональный комплекс",
                    "Office building": "Офисное здание",
                    "Mansion": "Особняк"
                }


                pred_build_choice = st.radio(
                    "Select building type:",
                    options=build_options_en
                )


                selected_pred_build = en_to_ru_build[pred_build_choice]

            predict_btn = st.form_submit_button("Predict")

        month_name_to_num = {
            "January": 1,
            "February": 2,
            "March": 3,
            "April": 4,
            "May": 5,
            "June": 6,
            "July": 7,
            "August": 8,
            "September": 9,
            "October": 10,
            "November": 11,
            "December": 12
        }

        if predict_btn:
            pred_prop = selected_pred_prop
            pred_tx = selected_pred_tx
            pred_area = total_area_input
            pred_floor = floor_input
            pred_total_floors = total_floors_input
            pred_month = month_name_to_num[month_input]
            pred_build = selected_pred_build
            pred_model = model_choice
        else:
            pred_prop = default_pred_prop
            pred_tx = default_pred_tx
            pred_area = default_pred_area
            pred_floor = default_pred_floor
            pred_total_floors = default_pred_total_floors
            pred_month = default_pred_month
            pred_build = default_pred_build
            pred_model = default_pred_model

        if predict_btn:
            if not map_data or not map_data.get("last_clicked"):
                st.warning("🚨 Please click on the map to select a location before predicting.")
                st.stop()
            lat = map_data["last_clicked"]["lat"]
            lng = map_data["last_clicked"]["lng"]
            st.markdown(f"**Location:** {lat:.5f}, {lng:.5f}")

            geolocator = Nominatim(user_agent="cre_app")

            def assign_season(month):
                if month in [12, 1, 2]:
                    return "winter"
                if month in [3, 4, 5]:
                    return "spring"
                if month in [6, 7, 8]:
                    return "summer"
                return "autumn"

            def assign_activity(month):
                return "high" if month in [2, 3, 4, 5, 9, 10, 11] else "low"

            def get_city_by_coordinates(lat, lng):
                try:
                    location = geolocator.reverse((lat, lng), exactly_one=True, language='ru')
                    if location and 'address' in location.raw:
                        address = location.raw['address']
                        return address.get('city') or address.get('town') or address.get('village') or address.get(
                            'municipality') or "Unknown"
                    else:
                        return "Unknown"
                except GeocoderTimedOut:
                    return "Unknown"

            def get_center_coordinates(place_name):
                try:
                    location = geolocator.geocode(place_name)
                    if location:
                        return location.latitude, location.longitude
                    else:
                        return None, None
                except Exception as e:
                    print(f"Error: {e}")
                    return None, None

            def haversine(lat1, lon1, lat2, lon2):
                R = 6371
                l1, l2 = math.radians(lat1), math.radians(lat2)
                diff_lat = math.radians(lat2 - lat1)
                diff_lon = math.radians(lon2 - lon1)
                a = math.sin(diff_lat / 2) ** 2 + math.cos(l1) * math.cos(l2) * math.sin(diff_lon / 2) ** 2
                return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            def metro_access(city):
                if city in ["Москва", "Новосибирск", "Казань", "Екатеринбург", "Самара"]:
                    return "Yes"
                return "No"

            curr_season = assign_season(pred_month)

            curr_activity = assign_activity(pred_month)

            curr_city = get_city_by_coordinates(lat, lng)

            city_lat, city_lng = get_center_coordinates(curr_city)

            distance = haversine(lat, lng, city_lat, city_lng)

            pred_metro = metro_access(curr_city)

            if pred_metro == "Yes":

                X_new = pd.DataFrame([{
                    'total_area': pred_area,
                    'floor': pred_floor,
                    'km_do_metro': df['km_do_metro'].mean(),
                    'dist_km': distance,
                    'object_type': pred_prop,
                    'building_type': pred_build,
                    'city1': curr_city,
                    'coords_lat': lat,
                    'coords_lng': lng,
                    'total_floors': pred_total_floors,
                    'season': curr_season,
                    'activity_level': curr_activity,
                    'month': pred_month
                }])

            else:

                X_new = pd.DataFrame([{
                    'total_area': pred_area,
                    'floor': pred_floor,
                    'dist_km': distance,
                    'object_type': pred_prop,
                    'building_type': pred_build,
                    'city1': curr_city,
                    'coords_lat': lat,
                    'coords_lng': lng,
                    'total_floors': pred_total_floors,
                    'season': curr_season,
                    'activity_level': curr_activity,
                    'month': pred_month
                }])

            metro_encoder = {"Yes": "with_metro", "No": "without_metro"}


            MODEL_PATHS = {
                "XGBoost": {
                    "rent": {
                        "with_metro": "models/xgb/rent_with_metro",
                        "without_metro": "models/xgb/rent_without_metro"
                    },
                    "sale": {
                        "with_metro": "models/xgb/sale_with_metro",
                        "without_metro": "models/xgb/sale_without_metro"
                    }
                },
                "KNN": {
                    "rent": {
                        "with_metro": "models/knn/rent_with_metro",
                        "without_metro": "models/knn/rent_without_metro"
                    },
                    "sale": {
                        "with_metro": "models/knn/sale_with_metro",
                        "without_metro": "models/knn/sale_without_metro"
                    }
                }
            }
            metro_model = metro_encoder[pred_metro]

            path = MODEL_PATHS[pred_model][pred_tx.lower()][metro_model]
            model, scaler, encoders = load_model_components(path)

            expected = list(scaler.feature_names_in_)
            X_new = X_new.reindex(columns=expected)

            X_new['month'] = X_new['month'].astype(int)

            X_proc = preprocess_input(X_new, scaler, encoders)


            price_sqm = model.predict(X_proc)[0]
            total_price = price_sqm * pred_area
            ci_low, ci_high = price_sqm * 0.9, price_sqm * 1.1


            st.metric("Price per m²", f"{price_sqm:,.2f}", delta=None)
            st.metric("Total price", f"{total_price:,.2f}", delta=None)
            st.write(f"Confidence interval: [{ci_low:,.2f} – {ci_high:,.2f}]")


            region_med = df[df['city1'] == curr_city]['price_per_sqm'].median()
            if price_sqm < region_med * 0.9:
                st.success("🏷️ This location appears undervalued vs region median.")
            elif price_sqm > region_med * 1.1:
                st.warning("💡 This location appears overpriced vs region median.")
            else:
                st.info("✅ This location is around the average price for the region.")

            comp_df = pd.DataFrame({
                'Label': ['Selected point', 'Region median'],
                'Price per m²': [price_sqm, region_med]
            })

            fig_comp = px.bar(
                comp_df,
                x='Price per m²',
                y='Label',
                orientation='h',
                color='Label',
                color_discrete_sequence=["#1f77b4", "#aec7e8"],
                title="Comparison: Selected vs Region Median Price per m²"
            )

            st.plotly_chart(fig_comp, use_container_width=True, key="price_comparison")
        else:
            if predict_btn:
                st.warning("🔍 Fill in the parameters and click Predict.")

    with general_charts:

        st.subheader("Transaction Type Distribution")
        fig1 = px.pie(
            df,
            names='listing_type',
            hole=0.4
        )
        st.plotly_chart(fig1, use_container_width=True, key="tx_dist")

        st.subheader("Owner Type Distribution")
        fig2 = px.pie(
            df,
            names='person_type',
            hole=0.4
        )
        st.plotly_chart(fig2, use_container_width=True, key="owner_dist")

        st.subheader("Object Type Distribution")
        fig3 = px.pie(
            df,
            names='object_type',
            hole=0.4
        )
        st.plotly_chart(fig3, use_container_width=True, key = "object_dist")


        st.subheader("Top 10 Regions Distribution")
        top10 = df['region'].value_counts().nlargest(10).reset_index()
        top10.columns = ['region', 'count']
        fig4 = px.pie(
            top10,
            names='region',
            values='count',
            hole=0.4
        )
        st.plotly_chart(fig4, use_container_width=True, key="region_dist")

        city_opts = df['region'].dropna().unique().tolist()
        prop_opts = df['object_type'].dropna().unique().tolist()

        st.subheader("📈 Custom Dynamics")

        deal_type = st.radio(
            "Select listing type:",
            options=list(ru_to_en.keys()),
            format_func=lambda x: ru_to_en[x],
            horizontal=True
        )

        mode = st.radio(
            "Analyze dynamics by:",
            options=["City", "Property Type"]
        )

        if mode == "City":
            city = st.selectbox("Select city:", options=city_opts)
            sub = df[
                (df['listing_type'] == deal_type) &
                (df['region'] == city)
                ].dropna(subset=['time', 'price_per_sqm', 'object_type'])
            q5_cm = sub['price_per_sqm'].quantile(0.05)
            q95_cm = sub['price_per_sqm'].quantile(0.95)
            sub = sub[(sub['price_per_sqm'] >= q5_cm) & (sub['price_per_sqm'] <= q95_cm)]
            data = (
                sub
                .groupby(['time', 'object_type'])['price_per_sqm']
                .median()
                .reset_index()
            )
            fig5 = px.line(
                data,
                x="time",
                y="price_per_sqm",
                color="object_type",
                title=f"Median Price/m² over Time in {city} by Property Type",
                labels={"price_per_sqm": "Price per m²", "time": "Time", "object_type": "Property Type"}
            )

            median_value = data['price_per_sqm'].median()

            fig5.add_trace(
                go.Scatter(
                    x=data['time'],
                    y=[median_value] * len(data),
                    mode='lines',
                    name='Overall Median',
                    line=dict(color='pink', width=3, dash='dot'),
                    showlegend=True
                )
            )
            st.plotly_chart(fig5, use_container_width=True, key="city_prop")

        else:

            prop = st.selectbox("Select property type:", options=prop_opts)
            sub = df[
                (df['listing_type'] == deal_type) &
                (df['object_type'] == prop)
                ].dropna(subset=['time', 'price_per_sqm', 'region'])
            q5_pm = sub['price_per_sqm'].quantile(0.05)
            q95_pm = sub['price_per_sqm'].quantile(0.95)
            sub = sub[(sub['price_per_sqm'] >= q5_pm) & (sub['price_per_sqm'] <= q95_pm)]
            data = (
                sub
                .groupby(['time', 'region'])['price_per_sqm']
                .median()
                .reset_index()
            )
            fig6 = px.line(
                data,
                x="time",
                y="price_per_sqm",
                color="region",
                title=f"Median Price/m² over Time for {prop} by City",
                labels={"price_per_sqm": "Price per m²", "time": "Time", "region": "Region"}
            )
            median_value = data['price_per_sqm'].median()

            fig6.add_trace(
                go.Scatter(
                    x=data['time'],
                    y=[median_value] * len(data),
                    mode='lines',
                    name='Overall Median',
                    line=dict(color='pink', width=3, dash='dot'),
                    showlegend=True
                )
            )
            st.plotly_chart(fig6, use_container_width=True, key="prop_city")

    monthly = (
        df
        .groupby(['listing_type', 'year', 'month'])['price_per_sqm']
        .median()
        .reset_index()
    )
    monthly['time'] = pd.to_datetime(monthly[['year', 'month']].assign(day=1))


    with rent_charts:
        st.subheader("Median price per m² over time — Rent")

        sub = df[df['listing_type'] == "Сдам"].dropna(subset=['time', 'price_per_sqm'])
        q5 = sub['price_per_sqm'].quantile(0.05)
        q95 = sub['price_per_sqm'].quantile(0.95)
        sub = sub[(sub['price_per_sqm'] >= q5) & (sub['price_per_sqm'] <= q95)]

        data = (
            sub
            .groupby('time')['price_per_sqm']
            .median()
            .reset_index()
        )

        fig_rent = px.line(
            data,
            x="time",
            y="price_per_sqm",
            labels={"price_per_sqm": "Price per m²", "time": "Time"}
        )

        median_value = data["price_per_sqm"].median()
        fig_rent.add_hline(
            y=median_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Median: {median_value:,.0f}",
            annotation_position="top left"
        )
        st.plotly_chart(fig_rent, use_container_width=True, key="rent_trend")

        listing = "Сдам"
        title_suffix = "Rent"

        sub = df[df['listing_type'] == listing].dropna(
            subset=['price_per_sqm', 'dist_km', 'km_do_metro', 'time', 'region'])
        sub['time'] = pd.to_datetime(sub['time'])

        q10_price = sub['price_per_sqm'].quantile(0.10)
        q90_price = sub['price_per_sqm'].quantile(0.90)
        sub = sub[(sub['price_per_sqm'] >= q10_price) & (sub['price_per_sqm'] <= q90_price)]

        st.subheader(f"Distance to Center vs Price/m² — {title_suffix}")
        q10_d = sub['dist_km'].quantile(0.01)
        q90_d = sub['dist_km'].quantile(0.95)
        sub_c = sub[(sub['dist_km'] >= q10_d) & (sub['dist_km'] <= q90_d)]
        fig7 = px.scatter(
            sub_c, x='dist_km', y='price_per_sqm',
            labels={'dist_km': 'Dist to Center (km)', 'price_per_sqm': 'Price/m²'},
            opacity=0.6
        )
        st.plotly_chart(fig7, use_container_width=True, key="dist_price_rent")

        st.subheader(f"Distance to Metro vs Price/m² — {title_suffix}")
        q10_m = sub['km_do_metro'].quantile(0.01)
        q90_m = sub['km_do_metro'].quantile(0.90)
        sub_m = sub[(sub['km_do_metro'] >= q10_m) & (sub['km_do_metro'] <= q90_m)]
        fig8 = px.scatter(
            sub_m, x='km_do_metro', y='price_per_sqm',
            labels={'km_do_metro': 'Dist to Metro (km)', 'price_per_sqm': 'Price/m²'},
            opacity=0.6
        )
        st.plotly_chart(fig8, use_container_width=True, key="metro_price_rent")

        rent = df[df['listing_type'] == "Сдам"].dropna(subset=['region', 'price_per_sqm'])

        q5 = rent['price_per_sqm'].quantile(0.05)
        q95 = rent['price_per_sqm'].quantile(0.95)
        rent = rent[(rent['price_per_sqm'] >= q5) & (rent['price_per_sqm'] <= q95)]

        rent_avg = rent.groupby('region')['price_per_sqm'].mean().reset_index()
        rent_median = rent_avg['price_per_sqm'].median()

        fig_rent = px.bar(
            rent_avg.sort_values('price_per_sqm', ascending=False),
            x='region',
            y='price_per_sqm',
            labels={'price_per_sqm': 'Avg Price per m²', 'region': 'Region'}
        )
        fig_rent.add_hline(
            y=rent_median,
            line_dash="dash",
            line_color="red",
            annotation_text="Median",
            annotation_position="bottom right"
        )

        st.subheader("Average Rent Price per Region")
        st.plotly_chart(fig_rent, use_container_width=True, key="bar_rent_avg_filtered")

        st.subheader(f"Distribution of {title_suffix} Prices")
        fig_h = px.histogram(
            sub, x='price_per_sqm', nbins=50,
            labels={'price_per_sqm': 'Price per m²'}
        )
        st.plotly_chart(fig_h, use_container_width=True, key="price_rent")


    with sales_charts:
        st.subheader("Median price per m² over time — Sale")

        sub = df[df['listing_type'] == "Продам"].dropna(subset=['time', 'price_per_sqm'])
        q5 = sub['price_per_sqm'].quantile(0.05)
        q95 = sub['price_per_sqm'].quantile(0.95)
        sub = sub[(sub['price_per_sqm'] >= q5) & (sub['price_per_sqm'] <= q95)]

        data = (
            sub
            .groupby('time')['price_per_sqm']
            .median()
            .reset_index()
        )

        fig_sale = px.line(
            data,
            x="time",
            y="price_per_sqm",
            labels={"price_per_sqm": "Price per m²", "time": "Time"}
        )
        median_value = data["price_per_sqm"].median()
        fig_sale.add_hline(
            y=median_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Median: {median_value:,.0f}",
            annotation_position="top left"
        )
        st.plotly_chart(fig_sale, use_container_width=True, key="sale_trend")

        listing = "Продам"
        title_suffix = "Sale"

        st.subheader(f"Distance to Center vs Price/m² — {title_suffix}")
        q10_d = sub['dist_km'].quantile(0.01)
        q90_d = sub['dist_km'].quantile(0.85)
        sub_c = sub[(sub['dist_km'] >= q10_d) & (sub['dist_km'] <= q90_d)]
        fig9 = px.scatter(
            sub_c, x='dist_km', y='price_per_sqm',
            labels={'dist_km': 'Dist to Center (km)', 'price_per_sqm': 'Price/m²'},
            opacity=0.6
        )
        st.plotly_chart(fig9, use_container_width=True, key="dist_price_sale")

        st.subheader(f"Distance to Metro vs Price/m² — {title_suffix}")
        q10_m = sub['km_do_metro'].quantile(0.01)
        q90_m = sub['km_do_metro'].quantile(0.90)
        sub_m = sub[(sub['km_do_metro'] >= q10_m) & (sub['km_do_metro'] <= q90_m)]
        fig_m = px.scatter(
            sub_m, x='km_do_metro', y='price_per_sqm',
            labels={'km_do_metro': 'Dist to Metro (km)', 'price_per_sqm': 'Price/m²'},
            opacity=0.6
        )
        st.plotly_chart(fig_m, use_container_width=True, key="metro_price_sale")

        sale = df[df['listing_type'] == "Продам"].dropna(subset=['region', 'price_per_sqm'])

        q5_s = sale['price_per_sqm'].quantile(0.05)
        q95_s = sale['price_per_sqm'].quantile(0.95)
        sale = sale[(sale['price_per_sqm'] >= q5_s) & (sale['price_per_sqm'] <= q95_s)]

        sale_avg = sale.groupby('region')['price_per_sqm'].mean().reset_index()
        sale_median = sale_avg['price_per_sqm'].median()

        fig_sale = px.bar(
            sale_avg.sort_values('price_per_sqm', ascending=False),
            x='region',
            y='price_per_sqm',
            labels={'price_per_sqm': 'Avg Price per m²', 'region': 'Region'}
        )
        fig_sale.add_hline(
            y=sale_median,
            line_dash="dash",
            line_color="red",
            annotation_text="Median",
            annotation_position="bottom right"
        )

        st.subheader("Average Sale Price per Region")
        st.plotly_chart(fig_sale, use_container_width=True, key="bar_sale_avg_filtered")

        st.subheader(f"Distribution of {title_suffix} Prices")
        fig10 = px.histogram(
            sub, x='price_per_sqm', nbins=50,
            labels={'price_per_sqm': 'Price per m²'}
        )

        st.plotly_chart(fig10, use_container_width=True, key="price_sale")


if __name__ == "__main__":
    main()
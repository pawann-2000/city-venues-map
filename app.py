from flask import Flask, render_template, request, url_for
from sklearn import preprocessing, cluster
from matplotlib.figure import Figure
from pandas import json_normalize
from dotenv import load_dotenv
from kneed import KneeLocator
from geopy import Nominatim
import os
import io
import scipy
import base64
import folium
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

load_dotenv()

app = Flask(__name__)

CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
API_VERSION = '20200316'
VENUE_LIMIT = 10000

cities = sorted([
    'Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad', 'Ahmedabad', 'Chennai', 'Kolkata', 'Surat', 'Pune', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 'Indore', 'Thane', 'Bhopal', 'Visakhapatnam', 'Patna', 'Vadodara', 'Ghaziabad', 'Ludhiana', 'Agra', 'Nashik', 'Faridabad', 'Meerut', 'Rajkot', 'Varanasi', 'Srinagar', 'Aurangabad', 'Navi Mumbai', 'Allahabad', 'Ranchi', 'Coimbatore', 'Vijayawada', 'Jodhpur', 'Madurai', 'Raipur', 'Chandigarh', 'Mysore', 'Tiruchirappalli', 'Bhubaneswar', 'Thiruvananthapuram', 'Guntur', 'Jamshedpur', 'Cuttack', 'Kochi', 'Dehradun', 'Gwalior', 'Amritsar', 'Udaipur'
])

@app.route('/')
def index():
    return render_template('index.html', cities=cities)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        msg = Message('Contact Form Submission', recipients=['your-email@example.com'])
        msg.body = f"Name: {name}\nEmail: {email}\nMessage: {message}"
        
        try:
            mail.send(msg)
            return redirect(url_for('contact_success'))
        except:
            return "There was an error sending your message. Please try again later."

    return render_template('contact.html')

@app.route('/contact_success')
def contact_success():
    return render_template('contact_success.html')

@app.route('/result', methods=['POST'])
def result():
    selected_city = request.form['city']
    
    geolocator = Nominatim(user_agent="CityVenueMapper")
    location = geolocator.geocode(selected_city)
    
    if not location:
        error_message = "Could not find location for the selected city."
        return render_template('index.html', error=error_message, cities=cities)
    
    latitude, longitude = location.latitude, location.longitude

    venues = get_venues(latitude, longitude)
    if not venues:
        error_message = "No venues found for the selected city."
        return render_template('index.html', error=error_message, cities=cities)
    
    venue_data = process_venues(venues)
    
    if venue_data.shape[0] < 3:
        error_message = "Not enough data points to perform clustering."
        return render_template('index.html', error=error_message, cities=cities)

    clustered_data = perform_clustering(venue_data)
    
    if clustered_data is None:
        error_message = "Clustering failed. Please try again."
        return render_template('index.html', error=error_message, cities=cities)
    
    map_html = create_map(latitude, longitude, clustered_data)
    analytics_html = create_analytics(clustered_data)
    
    return render_template('result.html', map_url=map_html, analytics=analytics_html)


def get_venues(lat, lng):
    url = f'https://api.foursquare.com/v2/venues/explore?&client_id={CLIENT_ID}&client_secret={CLIENT_SECRET}&v={API_VERSION}&ll={lat},{lng}&radius=30000&limit={VENUE_LIMIT}'
    response = requests.get(url).json()
    
    try:
        return response['response']['groups'][0]['items']
    except (KeyError, IndexError):
        return None

def process_venues(venues):
    venue_df = json_normalize(venues)
    
    restaurant_counts = []
    other_counts = []
    for lat, lng in zip(venue_df['venue.location.lat'], venue_df['venue.location.lng']):
        nearby_venues = get_nearby_venues(lat, lng)
        if nearby_venues is not None:
            venue_count, other_count = count_venue_types(nearby_venues)
            restaurant_counts.append(venue_count)
            other_counts.append(other_count)
        else:
            restaurant_counts.append(0)
            other_counts.append(0)
    
    venue_df['venue_count'] = restaurant_counts
    venue_df['other_count'] = other_counts

    columns_to_drop = [
        'referralId', 'reasons.count', 'reasons.items', 'venue.id', 'venue.name',
        'venue.location.labeledLatLngs', 'venue.location.distance', 'venue.location.cc',
        'venue.categories', 'venue.photos.count', 'venue.photos.groups',
        'venue.location.crossStreet', 'venue.location.address', 'venue.location.city',
        'venue.location.state', 'venue.location.crossStreet', 'venue.location.neighborhood',
        'venue.venuePage.id', 'venue.location.postalCode', 'venue.location.country'
    ]
    columns_to_drop = [col for col in columns_to_drop if col in venue_df.columns]
    
    processed_df = venue_df.drop(columns_to_drop, axis=1).dropna()
    processed_df = processed_df.rename(columns={'venue.location.lat': 'latitude', 'venue.location.lng': 'longitude'})
    
    spec_chars = ["[", "]","'"]
    for char in spec_chars:
        processed_df['venue.location.formattedAddress'] = processed_df['venue.location.formattedAddress'].astype(str).str.replace(char, '')
    
    return processed_df

def get_nearby_venues(lat, lng):
    url = f'https://api.foursquare.com/v2/venues/explore?&client_id={CLIENT_ID}&client_secret={CLIENT_SECRET}&v={API_VERSION}&ll={lat},{lng}&radius=1000&limit=100'
    response = requests.get(url).json()
    try:
        return response['response']['groups'][0]['items']
    except (KeyError, IndexError):
        return None

def count_venue_types(venues):
    nearby_venues = json_normalize(venues)
    categories = nearby_venues.get('venue.categories', [])
    
    venue_count = sum(1 for category in categories if isinstance(category, list) and len(category) > 0 and 'food' in category[0]['icon']['prefix'])
    other_count = len(categories) - venue_count
    
    return venue_count, other_count

def determine_optimal_k(X, wcs):
    max_clusters = min(10, len(X))  # Adjust the range based on the number of samples
    wcss = []

    for i in range(1, max_clusters + 1):
        try:
            kmeans = cluster.KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        except Exception as e:
            print(f"Error for k={i}: {e}")
            wcss.append(np.inf)  # Append infinity if an error occurs

    if np.max(wcss) == np.min(wcss):
        wcss_normalized = wcss  # Avoid division by zero if all WCSS are the same
    else:
        wcss_normalized = (wcss - np.min(wcss)) / (np.max(wcss) - np.min(wcss))
    
    if(wcs):
        return wcss_normalized

    kn = KneeLocator(range(1, max_clusters + 1), wcss_normalized, curve='convex', direction='decreasing')
    optimal_k = kn.knee
    if optimal_k is None or np.isnan(optimal_k):
        optimal_k = min(3, len(X))  # Fallback to 3 clusters or the number of samples if less
    return optimal_k

def create_analytics(data):
    sns.set(style="whitegrid")

    # Pairplot
    pairplot = sns.pairplot(data[['latitude', 'longitude', 'venue_count', 'other_count', 'cluster']], diag_kind="kde")
    pairplot_fig = pairplot.fig
    pairplot_buf = io.BytesIO()
    pairplot_fig.savefig(pairplot_buf, format='png')
    pairplot_buf.seek(0)
    pairplot_base64 = base64.b64encode(pairplot_buf.getvalue()).decode('utf-8')
    
    # Boxplot
    boxplot_fig = Figure()
    boxplot_ax = boxplot_fig.subplots()
    sns.boxplot(x='cluster', y='venue_count', data=data, ax=boxplot_ax)
    boxplot_buf = io.BytesIO()
    boxplot_fig.savefig(boxplot_buf, format='png')
    boxplot_buf.seek(0)
    boxplot_base64 = base64.b64encode(boxplot_buf.getvalue()).decode('utf-8')
    
    # Elbow Method Graphs
    elbow_fig = Figure()
    elbow_ax = elbow_fig.subplots()
    X = data[['latitude', 'longitude']]
    wcsss = determine_optimal_k(X, True)
    elbow_ax.plot(range(1, 11), wcsss)
    elbow_ax.set_title('Elbow Method')
    elbow_ax.set_xlabel('Number of clusters')
    elbow_ax.set_ylabel('WCSS')
    elbow_buf = io.BytesIO()
    elbow_fig.savefig(elbow_buf, format='png')
    elbow_buf.seek(0)
    elbow_base64 = base64.b64encode(elbow_buf.getvalue()).decode('utf-8')
    
    # Clustered Locations Map (Already created by `create_map` function)
    
    # Scatter Plots
    scatter_fig1 = Figure()
    scatter_ax1 = scatter_fig1.subplots()
    sns.scatterplot(x='latitude', y='longitude', hue='cluster', data=data, ax=scatter_ax1)
    scatter_buf1 = io.BytesIO()
    scatter_fig1.savefig(scatter_buf1, format='png')
    scatter_buf1.seek(0)
    scatter_base64_1 = base64.b64encode(scatter_buf1.getvalue()).decode('utf-8')
    
    scatter_fig2 = Figure()
    scatter_ax2 = scatter_fig2.subplots()
    sns.scatterplot(x='venue_count', y='other_count', hue='cluster', data=data, ax=scatter_ax2)
    scatter_buf2 = io.BytesIO()
    scatter_fig2.savefig(scatter_buf2, format='png')
    scatter_buf2.seek(0)
    scatter_base64_2 = base64.b64encode(scatter_buf2.getvalue()).decode('utf-8')

    return {
        'pairplot': pairplot_base64,
        'boxplot': boxplot_base64,
        'elbow': elbow_base64,
        'scatter1': scatter_base64_1,
        'scatter2': scatter_base64_2
    }

def perform_clustering(data):
    coordinates = data[["latitude", "longitude"]]
    optimal_k = determine_optimal_k(coordinates, False)
    
    if coordinates.shape[0] < optimal_k:
        optimal_k = coordinates.shape[0]

    model = cluster.KMeans(n_clusters=optimal_k, init='k-means++')
    
    try:
        data["cluster"] = model.fit_predict(coordinates)
    except ValueError as e:
        print(f"Clustering error: {e}")
        return None

    try:
        centroids, _ = scipy.cluster.vq.vq(model.cluster_centers_, coordinates.values)
        data["is_centroid"] = 0
        centroids = [i for i in centroids if i < len(data)]
        data.loc[centroids, "is_centroid"] = 1
    except Exception as e:
        print(f"Centroid error: {e}")
        data["is_centroid"] = 0
    
    return data

def create_map(lat, lng, data):
    map_object = folium.Map(location=[lat, lng], tiles="cartodbpositron", zoom_start=11)
    
    cluster_colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for _ in range(data['cluster'].nunique())]
    data["color"] = data["cluster"].apply(lambda x: cluster_colors[x])
    
    scaler = preprocessing.MinMaxScaler(feature_range=(3, 15))
    data["size"] = scaler.fit_transform(data["venue_count"].values.reshape(-1, 1)).reshape(-1)
    
    for _, row in data.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            color=row["color"],
            fill=True,
            popup=row["venue.location.formattedAddress"],
            radius=row["size"]
        ).add_to(map_object)
    
    legend_html = create_legend(data['cluster'].unique(), cluster_colors)
    map_object.get_root().html.add_child(folium.Element(legend_html))
    
    data[data["is_centroid"]==1].apply(
        lambda row: folium.Marker(
            location=[row["latitude"], row["longitude"]],
            draggable=False,
            popup=row["venue.location.formattedAddress"],
            icon=folium.Icon(color="red")
        ).add_to(map_object),
        axis=1
    )

    if not os.path.exists('static'):
        os.makedirs('static')
    
    map_file = "static/map.html"
    map_object.save(map_file)
    return url_for('static', filename='map.html')

def create_legend(elements, colors):
    legend_html = """
    <div style="position:fixed; bottom:10px; left:10px; border:2px solid black; z-index:9999; font-size:14px;">
    &nbsp;<b>Cluster:</b><br>
    """
    for element, color in zip(elements, colors):
        legend_html += f'&nbsp;<i class="fa fa-circle fa-1x" style="color:{color}"></i>&nbsp;{element}<br>'
    legend_html += "</div>"
    return legend_html

if __name__ == '__main__':
    app.run(debug=True)
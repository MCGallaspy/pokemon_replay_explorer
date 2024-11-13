import streamlit as st
import pandas as pd
import numpy as np

def get_raw_data():
    df = pd.read_hdf("regh_slim.h5", "table")
    df = df.sample(100, random_state=42)
    return df

df = get_raw_data()

st.header("Filters")
start, end = df.uploadtime.min(), df.uploadtime.max()
date_filters = st.date_input(
    "Upload date range",
    value=(start, end),
    min_value=start,
    max_value=end,
    format="YYYY/MM/DD"
)

if len(date_filters) == 0:
    filter_start, filter_end = start, end
if len(date_filters) == 1:
    filter_start, filter_end = date_filters[0], end
if len(date_filters) == 2:
    filter_start, filter_end = date_filters

rating_filter_start, rating_filter_end = st.slider(
    "Rating range",
    value=(df.rating.min(), df.rating.max()),
    min_value=df.rating.min(),
    max_value=df.rating.max()
)

mask = df.uploadtime <= pd.to_datetime(filter_end)
mask &= pd.to_datetime(filter_start) <= df.uploadtime
mask &= rating_filter_start <= df.rating
mask &= df.rating <= rating_filter_end
sample_df = df[mask]

col_config = {
    "replay_link": st.column_config.LinkColumn(
        display_text="Go to replay"),
}
if not st.toggle("Show hidden data", value=False):
    col_config.update({
        "appearances": None,
        "wins": None,
    })

st.header("Data used")
st.dataframe(sample_df, hide_index=False, column_config=col_config)

st.header("Single Pokemon Win-%")

@st.cache_data
def estimate_win_probability(sample_df):
    appearances = {}
    wins = {}
    for row in sample_df.itertuples():
        for pokemon, num_appearances in row.appearances.items():
            try:
                appearances[pokemon] += num_appearances
            except KeyError:
                appearances[pokemon] = num_appearances
        for pokemon, num_wins in row.wins.items():
            try:
                wins[pokemon] += num_wins
            except KeyError:
                wins[pokemon] = num_wins
    result = [{
        "pokemon": pokemon,
        "appearances": num_appearances,
        "wins": wins[pokemon],
    } for (pokemon, num_appearances) in appearances.items()]
    result = pd.DataFrame(data=result)
    result['win_pct'] = result.wins / result.appearances
    result['max_error_999pct_confidence'] = \
        3.29053 * np.sqrt(result.win_pct * (1 - result.win_pct) / result.appearances)
    result = result.sort_values(by="win_pct", ascending=False)
    return result

if st.button("Crunch the numbers"):
    single_pokemon_results_df = estimate_win_probability(
        sample_df
        )
    st.dataframe(single_pokemon_results_df)

st.header("Multiple Pokemon Win-%")
    
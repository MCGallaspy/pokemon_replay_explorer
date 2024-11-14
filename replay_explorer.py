import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def get_raw_data():
    df = pd.read_hdf("regh_slim.h5", "table")
    return df

df = get_raw_data()

st.header("Filters")
filters_columns = st.columns(3)

with filters_columns[0]:
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

with filters_columns[1]:
    rating_filter_start, rating_filter_end = st.slider(
        "Rating range",
        value=(df.rating.min(), df.rating.max()),
        min_value=df.rating.min(),
        max_value=df.rating.max()
    )

col_config = {
    "replay_link": st.column_config.LinkColumn(
        display_text="Go to replay"),
}

with filters_columns[2]:
    if not st.toggle("Show hidden data", value=False):
        col_config.update({
            "appearances": None,
            "wins": None,
        })

st.header("Data used")
data_selector_container = st.container()

mask = df.uploadtime <= pd.to_datetime(filter_end)
mask &= pd.to_datetime(filter_start) <= df.uploadtime
mask &= rating_filter_start <= df.rating
mask &= df.rating <= rating_filter_end
sample_df = df[mask]

bottom_menu = st.columns((3, 1, 1))
with bottom_menu[2]:
    page_size = st.number_input("Page Size", value=50)
with bottom_menu[1]:
    num_pages = (sample_df.shape[0] + page_size - 1) // page_size
    current_page = st.number_input(
        "Page", min_value=1, max_value=num_pages, step=1
    )
with bottom_menu[0]:
    st.markdown(f"Page **{current_page}** of **{num_pages}** ")



data_selector_container.dataframe(
    sample_df.iloc[(current_page - 1) * page_size:current_page * page_size + 1],
    hide_index=False,
    column_config=col_config,
)


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
    result['win_pct'] *= 100
    result['max_error_999pct_confidence'] *= 100
    result['confidence_lower_bound'] = result.win_pct - result.max_error_999pct_confidence
    result['confidence_upper_bound'] = result.win_pct + result.max_error_999pct_confidence
    result = result.sort_values(by="win_pct", ascending=False)
    return result

single_pokemon_results_df = estimate_win_probability(sample_df)
appearances_start, appearances_end = st.slider(
    "Num appearances filter",
    value=(single_pokemon_results_df.appearances.min(), single_pokemon_results_df.appearances.max()),
    min_value=single_pokemon_results_df.appearances.min(),
    max_value=single_pokemon_results_df.appearances.max()
)
mask = appearances_start <= single_pokemon_results_df.appearances
mask &= single_pokemon_results_df.appearances <= appearances_end
single_pokemon_results_df = single_pokemon_results_df[mask]
st.dataframe(single_pokemon_results_df,
column_config={
    "win_pct": st.column_config.NumberColumn("% Win", format="%.2f %%"),
    "max_error_999pct_confidence": st.column_config.NumberColumn("99.9% confidence interval size", format="%.2f %%"),
    "confidence_lower_bound": st.column_config.NumberColumn("% Win lower bound", format="%.2f %%"),
    "confidence_upper_bound": st.column_config.NumberColumn("% Win upper bound", format="%.2f %%"),
})

st.header("Multiple Pokemon Win-%")
    
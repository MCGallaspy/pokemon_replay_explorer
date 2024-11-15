import streamlit as st
import pandas as pd
import numpy as np

@st.cache_resource
def get_raw_data():
    df = pd.read_hdf("regh_slim.h5", "table")
    bo3_df = pd.read_hdf("regh_slim_bo3.h5", "table")
    df = pd.concat([df, bo3_df], axis=0)
    mons = df.appearances.apply(lambda x: set(x.keys()))
    df['replay_link'] = df.apply(lambda row: f"https://replay.pokemonshowdown.com/{row.name}", axis=1)
    all_mons = set()
    for monset in mons:
        all_mons |= monset
    return df, sorted(all_mons)

df, all_mons = get_raw_data()

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
        value=(1200.0, df.rating.max()),
        min_value=df.rating.min(),
        max_value=df.rating.max()
    )

col_config = {
    "replay_link": st.column_config.LinkColumn(
        display_text="Go to replay"),
    "pokemon": None,
}

with filters_columns[2]:
    include_unrated = st.toggle("Include unrated games", value=False)
    if not st.toggle("Show hidden data", value=False):
        col_config.update({
            "appearances": None,
            "wins": None,
        })

meta_formats = list(df.format.value_counts().index)
meta_format_filter = st.multiselect(
    "Filter by meta format",
    meta_formats,
    meta_formats,
)

seen_pokemon_filter = st.multiselect(
    "Filter by pokémon seen (any of these may appear)",
    all_mons,
)

won_pokemon_filter = st.multiselect(
    "Filter by pokémon that won",
    all_mons,
)

st.header("Data used")
data_selector_container = st.container()

mask = df.uploadtime <= pd.to_datetime(filter_end)
mask &= pd.to_datetime(filter_start) <= df.uploadtime
mask &= rating_filter_start <= df.rating
mask &= df.rating <= rating_filter_end
mask &= df.format.isin(meta_format_filter)

if include_unrated:
    mask |= df.rating.isna()

if len(seen_pokemon_filter) > 0:
    seen_mask = df.appearances.apply(
        lambda pokeset: any(p in seen_pokemon_filter for p in pokeset))
    mask &= seen_mask

if len(won_pokemon_filter) > 0:
    won_mask = df.wins.apply(
        lambda pokeset: any(pokeset.get(mon, 0) for mon in won_pokemon_filter))
    mask &= won_mask

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
    st.markdown(f"**{sample_df.shape[0]}** total replays")


columns = list(sample_df)
columns.remove('replay_link')
columns = ['replay_link'] + columns
data_selector_container.dataframe(
    sample_df.iloc[(current_page - 1) * page_size:current_page * page_size + 1],
    hide_index=True,
    column_config=col_config,
    column_order=columns,
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
    value=(10, single_pokemon_results_df.appearances.max()),
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

st.markdown("""A note on interpreting this:
* If only two pokémon appear, it means someone won with that lead pair,
  without switching in pokémon from the back.
* If three pokémon appear, it means one pokémon got switched in.
* If four pokémon appear, it means that the whole team was seen.
* If five pokémon appear, it's a Zoroark situation.
""")

@st.cache_data
def estimate_win_probability_teams(wins_df):
    appearances = {}
    wins = {}
    for row in wins_df.itertuples():
        winner = row.winner
        winning_pokemon = tuple(sorted(set(poke['name'] for poke in row.pokemon if poke['player'] == winner)))
        losing_pokemon = tuple(sorted(set(poke['name'] for poke in row.pokemon if poke['player'] != winner)))
        try:
            wins[winning_pokemon] += 1
        except KeyError:
            wins[winning_pokemon] = 1
        try:
            appearances[winning_pokemon] += 1
        except KeyError:
            appearances[winning_pokemon] = 1
        try:
            appearances[losing_pokemon] += 1
        except KeyError:
            appearances[losing_pokemon] = 1
    result = [{
        "team": pokemon,
        "appearances": num_appearances,
        "wins": wins.get(pokemon, 0),
    } for (pokemon, num_appearances) in appearances.items()]
    result = pd.DataFrame(data=result)
    result['win_pct'] = result.wins / result.appearances
    result['max_error_999pct_confidence'] = \
        3.29053 * np.sqrt(result.win_pct * (1- result.win_pct) / result.appearances)
    result['win_pct'] *= 100
    result['max_error_999pct_confidence'] *= 100
    result['confidence_lower_bound'] = result.win_pct - result.max_error_999pct_confidence
    result['confidence_upper_bound'] = result.win_pct + result.max_error_999pct_confidence
    result = result.sort_values(by="win_pct", ascending=False)
    return result

multi_pokemon_results_df = estimate_win_probability_teams(sample_df)

appearances_start, appearances_end = st.slider(
    "Num appearances filter",
    value=(10, multi_pokemon_results_df.appearances.max()),
    min_value=multi_pokemon_results_df.appearances.min(),
    max_value=multi_pokemon_results_df.appearances.max()
)

seen_pokemon_filter = st.multiselect(
    "Filter by pokémon seen (ALL of them must appear)",
    all_mons,
    key="multi_pokemon_results_seen_filter",
)

sort_keys = st.multiselect(
    "Sort pages by",
    list(multi_pokemon_results_df),
    "win_pct",
)
st.markdown("""The built-in table controls only sort the current page.
To sort results across all pages, use the selector above.""")

mask = appearances_start <= multi_pokemon_results_df.appearances
mask &= multi_pokemon_results_df.appearances <= appearances_end

if len(seen_pokemon_filter) > 0:
    seen_mask = multi_pokemon_results_df.team.apply(
        lambda pokeset: all(p in pokeset for p in seen_pokemon_filter)
    )
    mask &= seen_mask

multi_pokemon_results_df = multi_pokemon_results_df[mask]

if len(sort_keys) > 0:
    multi_pokemon_results_df = multi_pokemon_results_df.sort_values(by=sort_keys, ascending=False)

multi_results_container = st.container()

multi_results_bottom = st.columns((3, 1, 1))

with multi_results_bottom[2]:
    page_size = st.number_input("Page Size", value=50, key="multi_results_page_size")
with multi_results_bottom[1]:
    num_pages = (multi_pokemon_results_df.shape[0] + page_size - 1) // page_size
    current_page = st.number_input(
        "Page", min_value=1, max_value=num_pages, step=1,
        key="multi_pokemon_results_page",
    )
with multi_results_bottom[0]:
    st.markdown(f"Page **{current_page}** of **{num_pages}** ")
    st.markdown(f"**{multi_pokemon_results_df.shape[0]}** unique teams seen.")

multi_results_container.dataframe(
    multi_pokemon_results_df.iloc[(current_page - 1) * page_size:current_page * page_size + 1],
    column_config={
        "win_pct": st.column_config.NumberColumn("% Win", format="%.2f %%"),
        "max_error_999pct_confidence": st.column_config.NumberColumn("99.9% confidence interval size", format="%.2f %%"),
        "confidence_lower_bound": st.column_config.NumberColumn("% Win lower bound", format="%.2f %%"),
        "confidence_upper_bound": st.column_config.NumberColumn("% Win upper bound", format="%.2f %%"),
})
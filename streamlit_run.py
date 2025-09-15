# ---
# lambda-test: false  # auxiliary-file
# ---
# ## Demo Streamlit application.
#
# This application is the example from https://docs.streamlit.io/library/get-started/create-an-app.
#
# Streamlit is designed to run its apps as Python scripts, not functions, so we separate the Streamlit
# code into this module, away from the Modal application code.

def main():
    import numpy as np
    import pandas as pd
    import streamlit as st
    import plotly.express as px
    import os

    st.title("Uber pickups in NYC!")

    DATE_COLUMN = "date/time"
    DATA_URL = (
        "https://s3-us-west-2.amazonaws.com/"
        "streamlit-demo-data/uber-raw-data-sep14.csv.gz"
    )

    @st.cache_data
    def load_data(nrows):
        data = pd.read_csv(DATA_URL, nrows=nrows)

        def lowercase(x):
            return str(x).lower()

        data.rename(lowercase, axis="columns", inplace=True)
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
        return data

    data_load_state = st.text("Loading data...")
    data = load_data(10000)
    data_load_state.text("Done! (using st.cache_data)")

    if st.checkbox("Show raw data"):
        st.subheader("Raw data")
        st.write(data)

    st.subheader("Number of pickups by hour")
    hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
    st.bar_chart(hist_values)

    # Some number in the range 0-23
    hour_to_filter = st.slider("hour", 0, 23, 17)
    filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

    st.subheader("Map of all pickups at %s:00" % hour_to_filter)
    st.map(filtered_data)

    st.markdown("### Daily pickups over time")
    daily_counts = (
        data.groupby(pd.Grouper(key=DATE_COLUMN, freq="D"))
        .size()
        .reset_index(name="pickups")
        )
    daily_counts.rename(columns={DATE_COLUMN: "date"}, inplace=True)
    fig_daily = px.line(
        daily_counts,
        x="date",
        y="pickups",
        markers=True,
        title="Pickups per day"
        )
    st.plotly_chart(fig_daily, use_container_width=True)
    
    st.markdown("### Pickup intensity by hour and weekday")

    # make sure hour and weekday exist
    data["hour"] = data[DATE_COLUMN].dt.hour
    data["weekday"] = data[DATE_COLUMN].dt.day_name()
    data["weekday_num"] = data[DATE_COLUMN].dt.weekday  # Monday=0

    # aggregate counts
    heat = (
        data.groupby(["weekday_num", "weekday", "hour"])
            .size()
            .reset_index(name="pickups")
            )

    # pivot to wide format for heatmap
    heat_pivot = (
        heat.pivot_table(
            index=["weekday_num", "weekday"],
            columns="hour",
            values="pickups",
            fill_value=0
            )
            .sort_index(level=0)
            )

    # use weekday names (not numbers) for y-axis labels
    heat_display = heat_pivot.copy()
    heat_display.index = [idx[1] for idx in heat_display.index]

    fig_heat = px.imshow(
        heat_display.values,
        labels=dict(x="Hour", y="Weekday", color="Pickups"),
        x=list(heat_display.columns),
        y=list(heat_display.index),
        aspect="auto",
        title="Heatmap: pickups by hour × weekday"
        )

    st.plotly_chart(fig_heat, use_container_width=True)

    try:
        from supabase import create_client 
    except Exception:
        create_client = None

    @st.cache_data(show_spinner=True)
    def load_soccerplays():
        if not create_client:
            st.warning("supabase-py not installed. `pip install supabase`")
            return pd.DataFrame()
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not (url and key):
            st.info("Set SUPABASE_URL and SUPABASE_KEY to load 'soccerplays'.")
            return pd.DataFrame()
        client = create_client(url, key)
        res = client.table("soccerplays").select("*").execute()
        return pd.DataFrame(res.data)

    st.header("Soccer Plays (Supabase)")

    sb_df = load_soccerplays()
    if sb_df.empty:
        st.stop()

    # Preview table
    st.subheader("Data sample")
    st.dataframe(sb_df.head(25), use_container_width=True)

    # ---- Column mapping (case-insensitive) ----
    cols = {c.lower(): c for c in sb_df.columns}
    def pick(*cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    col_match_date  = pick("match_date", "date")
    col_team        = pick("team_name", "team")
    col_player      = pick("player_name", "player")
    col_play_type   = pick("play_type", "event_type", "action")
    col_minute      = pick("minute_played", "minute", "time_minute")

    # Make sure minute is numeric
    if col_minute and sb_df[col_minute].dtype.kind not in "iu":
        sb_df[col_minute] = pd.to_numeric(sb_df[col_minute], errors="coerce")

    # ---- Event timeline (by minute) ----
    if col_minute and col_team and col_player:
        st.subheader("Event Timeline (by Minute)")
        # Optional filter by match_date if present
        df_t = sb_df.copy()
        if col_match_date and df_t[col_match_date].notna().any():
            # ensure string for selectbox compatibility
            date_opts = sorted(pd.Series(df_t[col_match_date].dropna()).astype(str).unique().tolist())
            chosen = st.selectbox("Filter by match_date (optional)", ["All"] + date_opts, index=0)
            if chosen != "All":
                df_t = df_t[df_t[col_match_date].astype(str) == chosen]

        fig_tl = px.strip(
            df_t,
            x=col_minute,
            y=col_team,
            color=col_play_type if col_play_type else col_team,
            hover_data=[c for c in [col_player, col_play_type, col_match_date] if c],
            title="Events over match minute (each dot = an event)",
        )
        fig_tl.update_traces(jitter=0.25, marker=dict(size=9), selector=dict(mode="markers"))
        fig_tl.update_layout(
            xaxis_title="Minute",
            yaxis_title="Team",
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_tl, use_container_width=True)
    else:
        st.info("Need columns for minute, team, and player to draw the timeline. Check your 'soccerplays' schema.")

if __name__ == "__main__":
    main()
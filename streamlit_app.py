import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import altair as alt
import numpy as np
import os
from datetime import datetime as dt
import base64
from pathlib import Path


@st.cache(persist=True)
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
    df = df.set_index(['Country', 'Country_id', 'Region', 'Region_id', 'Date'])
    df[['Cases', 'Deaths',
        'Partial Vaccinations', 'Full Vaccinations']] = df.groupby(level=[0, 1, 2, 3]).apply(lambda x: x.diff())

    return df.reset_index()


@st.cache(persist=True)
def load_geodata(url, feature):
    source = alt.topo_feature(url, feature)

    return source


@st.cache
def load_metadata(filepath):
    df = pd.read_csv(filepath)

    return df


@st.cache
def load_keys(filepath):
    df = pd.read_csv(filepath)

    return df


def filter_data(df, countries, regions, start_dt, end_dt, level, interval):
    lookup = countries if level == 'Country' else regions

    grouped = df.groupby([level, level + '_id', 'Date']).sum()

    if interval != 'D':
        level_values = grouped.index.get_level_values
        grouped = (grouped.groupby([level_values(i) for i in [0, 1]] +
                                   [pd.Grouper(freq=interval, level=-1)]).agg({'Cumulative Cases': 'last',
                                                                               'Cumulative Deaths': 'last',
                                                                               'Cumulative Partial Vaccinations': 'last',
                                                                               'Cumulative Full Vaccinations': 'last',
                                                                               'Cases': 'sum',
                                                                               'Deaths': 'sum',
                                                                               'Partial Vaccinations': 'sum',
                                                                               'Full Vaccinations': 'sum'}))

    grouped = grouped.reset_index()

    filtered = grouped[(grouped[level].isin(lookup)) & (grouped['Date'] >= start_dt) & (grouped['Date'] <= end_dt)]

    return filtered


def determine_metric(metric_type, cumulative, rolling, per100k, interval):
    if metric_type == 'Cases':
        if cumulative:
            if rolling and interval == 'Daily':
                if per100k:
                    metric = 'Cumulative Cases (rolling average, per 100k)'
                else:
                    metric = 'Cumulative Cases (rolling average)'
            else:
                if per100k:
                    metric = 'Cumulative Cases (per 100k)'
                else:
                    metric = 'Cumulative Cases'
        else:
            if rolling and interval == 'Daily':
                if per100k:
                    metric = 'Cases (rolling average, per 100k)'
                else:
                    metric = 'Cases (rolling average)'
            else:
                if per100k:
                    metric = 'Cases (per 100k)'
                else:
                    metric = 'Cases'

    elif metric_type == 'Deaths':
        if cumulative:
            if rolling and interval == 'Daily':
                if per100k:
                    metric = 'Cumulative Deaths (rolling average, per 100k)'
                else:
                    metric = 'Cumulative Deaths (rolling average)'
            else:
                if per100k:
                    metric = 'Cumulative Deaths (per 100k)'
                else:
                    metric = 'Cumulative Deaths'
        else:
            if rolling and interval == 'Daily':
                if per100k:
                    metric = 'Deaths (rolling average, per 100k)'
                else:
                    metric = 'Deaths (rolling average)'
            else:
                if per100k:
                    metric = 'Deaths (per 100k)'
                else:
                    metric = 'Deaths'
    elif metric_type == 'Partial Vaccinations':
        if cumulative:
            if rolling and interval == 'Daily':
                if per100k:
                    metric = 'Cumulative Partial Vaccinations (rolling average, per 100k)'
                else:
                    metric = 'Cumulative Partial Vaccinations (rolling average)'
            else:
                if per100k:
                    metric = 'Cumulative Partial Vaccinations (per 100k)'
                else:
                    metric = 'Cumulative Partial Vaccinations'
        else:
            if rolling and interval == 'Daily':
                if per100k:
                    metric = 'Partial Vaccinations (rolling average, per 100k)'
                else:
                    metric = 'Partial Vaccinations (rolling average)'
            else:
                if per100k:
                    metric = 'Partial Vaccinations (per 100k)'
                else:
                    metric = 'Partial Vaccinations'
    elif metric_type == 'Full Vaccinations':
        if cumulative:
            if rolling and interval == 'Daily':
                if per100k:
                    metric = 'Cumulative Full Vaccinations (rolling average, per 100k)'
                else:
                    metric = 'Cumulative Full Vaccinations (rolling average)'
            else:
                if per100k:
                    metric = 'Cumulative Full Vaccinations (per 100k)'
                else:
                    metric = 'Cumulative Full Vaccinations'
        else:
            if rolling and interval == 'Daily':
                if per100k:
                    metric = 'Full Vaccinations (rolling average, per 100k)'
                else:
                    metric = 'Full Vaccinations (rolling average)'
            else:
                if per100k:
                    metric = 'Full Vaccinations (per 100k)'
                else:
                    metric = 'Full Vaccinations'

    return metric


def calculate_metrics(df, level, how, interval, metadata, rolling_period=14, scale=100000):
    df = df.merge(metadata[['id', 'population', 'latitude', 'longitude']], left_on=level + '_id', right_on='id',
                  how='inner').drop(columns=['id'])

    if how == 'geo':
        group = df.groupby([level, level + '_id', 'population', 'latitude', 'longitude']).agg(
            {'Cumulative Cases': 'last',
             'Cumulative Deaths': 'last',
             'Cumulative Partial Vaccinations': 'last',
             'Cumulative Full Vaccinations': 'last',
             'Cases': 'sum',
             'Deaths': 'sum',
             'Partial Vaccinations': 'sum',
             'Full Vaccinations': 'sum'})
        group = group.reset_index(level=2)
    elif how == 'temp':
        group = df.groupby('Date').sum()
        if interval == 'Daily':
            group['Cumulative Cases (rolling average)'] = round(
                group['Cumulative Cases'].rolling(rolling_period, min_periods=1).mean(), 0)
            group['Cumulative Deaths (rolling average)'] = round(
                group['Cumulative Deaths'].rolling(rolling_period, min_periods=1).mean(), 0)
            group['Cumulative Partial Vaccinations (rolling average)'] = round(
                group['Cumulative Partial Vaccinations'].rolling(rolling_period, min_periods=1).mean(), 0)
            group['Cumulative Full Vaccinations (rolling average)'] = round(
                group['Cumulative Full Vaccinations'].rolling(rolling_period, min_periods=1).mean(), 0)

            group['Cases (rolling average)'] = round(group['Cases'].rolling(rolling_period, min_periods=1).mean(), 0)
            group['Deaths (rolling average)'] = round(group['Deaths'].rolling(rolling_period, min_periods=1).mean(), 0)
            group['Partial Vaccinations (rolling average)'] = round(group['Partial Vaccinations'].rolling(rolling_period, min_periods=1).mean(), 0)
            group['Full Vaccinations (rolling average)'] = round(group['Full Vaccinations'].rolling(rolling_period, min_periods=1).mean(), 0)
    elif how == 'geo_temp':
        group = df.set_index([level, level + '_id', 'Date'])
        if interval == 'Daily':
            group['Cumulative Cases (rolling average)'] = round(group.groupby(level=[0, 1])['Cumulative Cases'].apply(
                lambda x: x.rolling(rolling_period, min_periods=1).mean()), 0)
            group['Cumulative Deaths (rolling average)'] = round(group.groupby(level=[0, 1])['Cumulative Deaths'].apply(
                lambda x: x.rolling(rolling_period, min_periods=1).mean()), 0)
            group['Cumulative Partial Vaccinations (rolling average)'] = round(group.groupby(level=[0, 1])['Cumulative Partial Vaccinations'].apply(
                lambda x: x.rolling(rolling_period, min_periods=1).mean()), 0)
            group['Cumulative Full Vaccinations (rolling average)'] = round(group.groupby(level=[0, 1])['Cumulative Full Vaccinations'].apply(
                lambda x: x.rolling(rolling_period, min_periods=1).mean()), 0)

            group['Cases (rolling average)'] = round(group.groupby(level=[0, 1])['Cases'].apply(
                lambda x: x.rolling(rolling_period, min_periods=1).mean()), 0)
            group['Deaths (rolling average)'] = round(group.groupby(level=[0, 1])['Deaths'].apply(
                lambda x: x.rolling(rolling_period, min_periods=1).mean()), 0)
            group['Partial Vaccinations (rolling average)'] = round(group.groupby(level=[0, 1])['Partial Vaccinations'].apply(
                lambda x: x.rolling(rolling_period, min_periods=1).mean()), 0)
            group['Full Vaccinations (rolling average)'] = round(group.groupby(level=[0, 1])['Full Vaccinations'].apply(
                lambda x: x.rolling(rolling_period, min_periods=1).mean()), 0)

    group['Cumulative Cases (per 100k)'] = round(group['Cumulative Cases'] / group['population'] * scale, 0)
    group['Cumulative Deaths (per 100k)'] = round(group['Cumulative Deaths'] / group['population'] * scale, 0)
    group['Cumulative Partial Vaccinations (per 100k)'] = round(group['Cumulative Partial Vaccinations'] / group['population'] * scale, 0)
    group['Cumulative Full Vaccinations (per 100k)'] = round(group['Cumulative Full Vaccinations'] / group['population'] * scale, 0)
    group['Cases (per 100k)'] = round(group['Cases'] / group['population'] * scale, 0)
    group['Deaths (per 100k)'] = round(group['Deaths'] / group['population'] * scale, 0)
    group['Partial Vaccinations (per 100k)'] = round(group['Partial Vaccinations'] / group['population'] * scale, 0)
    group['Full Vaccinations (per 100k)'] = round(group['Full Vaccinations'] / group['population'] * scale, 0)

    if interval == 'Daily' and how != 'geo':
        group['Cumulative Cases (rolling average, per 100k)'] = round(
            group['Cumulative Cases (rolling average)'] / group['population'] * scale, 0)
        group['Cumulative Deaths (rolling average, per 100k)'] = round(
            group['Cumulative Deaths (rolling average)'] / group['population'] * scale, 0)
        group['Cumulative Partial Vaccinations (rolling average, per 100k)'] = round(
            group['Cumulative Partial Vaccinations (rolling average)'] / group['population'] * scale, 0)
        group['Cumulative Full Vaccinations (rolling average, per 100k)'] = round(
            group['Cumulative Full Vaccinations (rolling average)'] / group['population'] * scale, 0)

        group['Cases (rolling average, per 100k)'] = round(
            group['Cases (rolling average)'] / group['population'] * scale, 0)
        group['Deaths (rolling average, per 100k)'] = round(
            group['Deaths (rolling average)'] / group['population'] * scale, 0)
        group['Partial Vaccinations (rolling average, per 100k)'] = round(
            group['Partial Vaccinations (rolling average)'] / group['population'] * scale, 0)
        group['Full Vaccinations (rolling average, per 100k)'] = round(
            group['Full Vaccinations (rolling average)'] / group['population'] * scale, 0)

    return group.reset_index()


def plot_choropleth(geo, df, level, per100k):
    source = geo

    click = alt.selection_multi(fields=[level])
    if metric_type == 'Cases':
        focus = 'Cases (per 100k)' if per100k else 'Cases'
        altfocus = 'Cases' if per100k else 'Cases (per 100k)'
    elif metric_type == 'Deaths':
        focus = 'Deaths (per 100k)' if per100k else 'Deaths'
        altfocus = 'Deaths' if per100k else 'Deaths (per 100k)'
    elif metric_type == 'Partial Vaccinations':
        focus = 'Partial Vaccinations (per 100k)' if per100k else 'Partial Vaccinations'
        altfocus = 'Partial Vaccinations' if per100k else 'Partial Vaccinations (per 100k)'
    elif metric_type == 'Full Vaccinations':
        focus = 'Full Vaccinations (per 100k)' if per100k else 'Full Vaccinations'
        altfocus = 'Full Vaccinations' if per100k else 'Full Vaccinations (per 100k)'

    df = df[df[focus] > 0]
    # nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=['NAME'], empty='none')
    # brush = alt.selection(type='interval', encodings=['x'], fields=['NAME'])

    choropleth = alt.Chart(source).mark_geoshape().encode(
        color=alt.Color(focus + ':Q', legend=None),  # , scale=alt.Scale(scheme='redyellowblue', reverse=True)
        tooltip=[level + ':N',
                 alt.Tooltip('population:Q', format=',.0f', title='Population'),
                 alt.Tooltip(focus + ':Q', format=',.0f'), alt.Tooltip(altfocus + ':Q', format=',.0f')],
        opacity=alt.condition(click, alt.value(1), alt.value(0.2))
    ).transform_lookup(
        lookup='properties.UID',
        from_=alt.LookupData(df, level + '_id', ['Country', 'Region',
                                                 'Cases', 'Cases (per 100k)',
                                                 'Deaths', 'Deaths (per 100k)',
                                                 'Partial Vaccinations', 'Partial Vaccinations (per 100k)',
                                                 'Full Vaccinations', 'Full Vaccinations (per 100k)',
                                                 'population', 'latitude', 'longitude'])
    ).add_selection(
        click
    )#.properties(
    #     title=alt.TitleParams(text=' ', subtitle=format_date(start_dt) + ' - ' + format_date(end_dt), align='left',
    #                           anchor='start')
    # )

    points = alt.Chart(df).mark_circle().encode(
        longitude=alt.Longitude('longitude'),
        latitude=alt.Latitude('latitude'),
        color=alt.value('#F63366'),
        opacity=alt.condition(click, alt.value(1), alt.value(0.2))
    )

    text = points.mark_text(align='left', dx=5, dy=0).encode(
        text=alt.Text(level),
        opacity=alt.condition(click, alt.value(1), alt.value(0.2))

    )

    if label:
        return choropleth + points + text
    else:
        return choropleth


def plot_bubblechart(df, level, per100k):
    if metric_type == 'Cases':
        focus = 'Cases (per 100k)' if per100k else 'Cases'
        altfocus = 'Cases' if per100k else 'Cases (per 100k)'
    elif metric_type == 'Deaths':
        focus = 'Deaths (per 100k)' if per100k else 'Deaths'
        altfocus = 'Deaths' if per100k else 'Deaths (per 100k)'
    elif metric_type == 'Partial Vaccinations':
        focus = 'Partial Vaccinations (per 100k)' if per100k else 'Partial Vaccinations'
        altfocus = 'Partial Vaccinations' if per100k else 'Partial Vaccinations (per 100k)'
    elif metric_type == 'Full Vaccinations':
        focus = 'Full Vaccinations (per 100k)' if per100k else 'Full Vaccinations'
        altfocus = 'Full Vaccinations' if per100k else 'Full Vaccinations (per 100k)'

    df['JITTER'] = np.random.randint(0, 100, len(df))

    click = alt.selection_multi(fields=[level])

    points = alt.Chart(df).mark_circle().encode(
        x=alt.X(focus, axis=alt.Axis(grid=False, title=focus)),
        y=alt.Y('JITTER:Q', axis=None),  # alt.Axis(title='', grid=False, labels=False, ticks=False))
        color=alt.Color(focus, legend=None, scale=alt.Scale(scheme='yellowgreenblue')),
        opacity=alt.condition(click, alt.value(1), alt.value(0.2)),
        tooltip=[level, alt.Tooltip(focus, format=',.0f'), alt.Tooltip(altfocus, format=',.0f')],
        size=alt.Size(altfocus, legend=None)
    ).properties(
        height=100,
    ).add_selection(click)

    return points


def plot_choropleth_compound(geo, df, level, per100k):
    choropleth = plot_choropleth(geo, df, level, per100k)
    points = plot_bubblechart(df, level, per100k)

    combined_click = alt.selection_multi(fields=[level])
    choropleth.encode(
        opacity=alt.condition(combined_click, alt.value(1), alt.value(0.2))
    ).add_selection(
        combined_click
    )
    points.encode(
        opacity=alt.condition(combined_click, alt.value(1), alt.value(0.2))
    ).add_selection(
        combined_click
    )

    chart = alt.vconcat(choropleth, points).configure_view(strokeWidth=0)

    #chart.encode(opacity=alt.condition(click, alt.value(1), alt.value(0.2))).add_selection(combined_click)

    return chart


def plot_timeseries(df, level, metric):
    df = df[df[metric] > 0]

    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=['Date'], empty='none')
    #highlight = alt.selection(type='single', nearest=True, on='mouseover', fields=[level], empty='none')

    # The basic line
    line = alt.Chart(df).mark_line(interpolate='monotone').encode(
        x=alt.X('Date:T', title='', axis=alt.Axis(grid=False)),
        y=alt.Y(metric, axis=alt.Axis(grid=False)),
        color=alt.Color(level, title=level),
        #opacity=alt.condition(highlight, alt.value(1), alt.value(0.2))
    )

    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(df).mark_point().encode(
        x='Date:T',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        tooltip=[level],
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, alt.Text(metric + ':Q', format=',.0f'), alt.value(' '))
    )

    # Draw a rule at the location of the selection
    rules = alt.Chart(df).mark_rule(color='gray').encode(
        x='Date:T',
    ).transform_filter(
        nearest
    )

    # Put the five layers into a chart and bind the data
    chart = alt.layer(
        line, selectors, points, rules, text
    )
    chart.display()
    return chart.configure_view(strokeWidth=0)


def plot_trend(df, metric, color, upisbad):
    col_cum = determine_metric(metric, True, rolling, False, interval)
    col_inc = determine_metric(metric, False, rolling, False, interval)

    df = df[df[col_inc] >= 0]

    last_value_cum = int(df[col_cum].iloc[-1])
    first_value_cum = int(df[col_cum].iloc[0])
    last_value = int(df[col_inc].iloc[-1])
    total = last_value_cum - first_value_cum
    change = df[col_inc].pct_change().mul(100).round(1).iloc[-1]

    if change > 0:
        indicator = '🔺'
        if upisbad:
            txt_color = 'red'
        else:
            txt_color = 'green'
    elif change < 0:
        indicator = '🔻'
        if upisbad:
            txt_color = 'green'
        else:
            txt_color = 'red'
    else:
        indicator = '🔹'
        txt_color = 'green'
    change_txt = indicator + str(abs(change)) + '%'

    if interval == 'Daily':
        chart_type = alt.Chart(df).mark_area(line=True, interpolate='monotone')
    else:
        chart_type = alt.Chart(df).mark_bar()

    chart = chart_type.encode(
        x=alt.X('Date:O', axis=alt.Axis(grid=False, labels=False, ticks=False,
                                        title=f'{last_value:,.0f}' + ' ' + change_txt,
                                        titleAlign='right',
                                        titleAnchor='end', titleColor=txt_color), ),
        y=alt.Y(col_inc, axis=None),
        color=alt.value(color),
        tooltip=['Date', alt.Tooltip(col_inc, format=',.0f')]
    ).properties(
        title=alt.TitleParams(text=col_inc,
                              subtitle=f'Total: {total:,.0f}',
                              anchor='start', align='left')
    ).configure_view(strokeWidth=0, height=100)

    return chart


def plot_trend_stacked(df, title, metric1, metric2, upisbad):
    col_cum1 = determine_metric(metric1, True, rolling, False, interval)
    col_inc1 = determine_metric(metric1, False, rolling, False, interval)
    col_cum2 = determine_metric(metric2, True, rolling, False, interval)
    col_inc2 = determine_metric(metric2, False, rolling, False, interval)

    last_value_cum1 = int(df[col_cum1].iloc[-1])
    first_value_cum1 = int(df[col_cum1].iloc[0])
    last_value_cum2 = int(df[col_cum2].iloc[-1])
    first_value_cum2 = int(df[col_cum2].iloc[0])

    last_value1 = int(df[col_inc1].iloc[-1])
    last_value2 = int(df[col_inc2].iloc[-1])

    total1 = last_value_cum1 - first_value_cum1
    total2 = last_value_cum2 - first_value_cum2

    change = df[col_inc2].pct_change().mul(100).round(1).iloc[-1]

    if change > 0:
        indicator = '🔺'
        if upisbad:
            txt_color = 'red'
        else:
            txt_color = 'green'
    elif change < 0:
        indicator = '🔻'
        if upisbad:
            txt_color = 'green'
        else:
            txt_color = 'red'
    else:
        indicator = '🔹'
        txt_color = 'green'
    change_txt = indicator + str(abs(change)) + '%'

    source = df[['Date', col_inc1, col_inc2]].melt(id_vars='Date', var_name='Metric', value_name='Value')

    if interval == 'Daily':
        chart_type = alt.Chart(source).mark_area(line=True, interpolate='monotone')
    else:
        chart_type = alt.Chart(source).mark_bar()

    chart = chart_type.encode(
        x=alt.X('Date:O', axis=alt.Axis(grid=False, labels=False, ticks=False,
                                        title=f'{last_value2:,.0f}' + ' ' + change_txt,
                                        titleAlign='right',
                                        titleAnchor='end', titleColor=txt_color), ),
        y=alt.Y('sum(Value)', axis=None),
        color=alt.Color('Metric', legend=None, scale=alt.Scale(scheme='greens', reverse=True)),
        tooltip=['Date', 'Metric', alt.Tooltip('Value', format=',.0f')]
    ).properties(
        title=alt.TitleParams(text=title,
                              subtitle=f'Total: {total2:,.0f} (Full)',
                              anchor='start', align='left')
    ).configure_view(strokeWidth=0, height=100)

    return chart


def plot_progressbar(df):
    case_ratio = round(df[determine_metric('Cases', True, False, True, interval)].iloc[-1] / 100000, 2)
    vac_ratio = round(df[determine_metric('Full Vaccinations', True, False, True, interval)].iloc[-1] / 100000, 2)

    if layout == 'Desktop':
        title = alt.TitleParams(text='Population Infection & Vaccination Rates', anchor='start', align='left')
    else:
        title = ''

    source = pd.DataFrame({'Total': ['Total', 'Total'],
                           'Metric': ['Infection Rate', 'Vaccination Rate'],
                           'Value': [case_ratio, vac_ratio]})

    points = alt.Chart(source).mark_point(size=400).encode(
        alt.X(
            'Value:Q',
            title="",
            scale=alt.Scale(zero=True, domain=(0, 1)),
            axis=alt.Axis(grid=False, format='%'),

        ),
        alt.Y(
            'Total:N',
            title="",
            #sort='-x',
            axis=None
        ),
        color=alt.Color('Metric:N', legend=None),
        # row=alt.Row(
        #     'site:N',
        #     title="",
        #     sort=alt.EncodingSortField(field='yield', op='sum', order='descending'),
        # ),
        opacity=alt.value(0),
        tooltip=['Metric:N', alt.Tooltip('Value:Q', format='.0%')]
    ).properties(
        #height=alt.Step(20),
        title=title
    )

    text = points.mark_text(align='center', fontSize=40).encode(
        text=alt.Text('emoji:N'),
        opacity=alt.value(1)
    ).transform_calculate(
        emoji="{'Infection Rate': '🦠', 'Vaccination Rate': '💉'}[datum.Metric]"
    )

    return (points + text).configure_view(stroke="transparent", height=100)


def format_date(x):
    if interval_cd == 'D':
        return dt.strptime(str(x), '%Y-%m-%d %H:%M:%S').strftime('%-d-%b %y')
    elif interval_cd == 'W':
        return dt.strptime(str(x), '%Y-%m-%d %H:%M:%S').strftime('w/e %-d-%b %y')
    elif interval_cd == 'M':
        return dt.strptime(str(x), '%Y-%m-%d %H:%M:%S').strftime('%b %y')


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def mobile_view():
    with st.beta_expander(format_date(start_dt) + ' - ' + format_date(end_dt), expanded=False):
        st.text(', '.join(sorted(countries)))

    with st.beta_expander('🧭 Trends - ' + interval, expanded=True):
        t1, t2, t3 = st.beta_columns(3)
        with t1:
            st.altair_chart(plot_trend(timeseries_summary, 'Cases', '#faca2b', True),
                            use_container_width=True)
        with t2:
            st.altair_chart(plot_trend(timeseries_summary, 'Deaths', '#ff2b2b', True),
                            use_container_width=True)
        with t3:
            st.altair_chart(plot_trend_stacked(timeseries_summary, 'Vaccinations',
                                               'Partial Vaccinations', 'Full Vaccinations', False),
                            use_container_width=True)

    with st.beta_expander('🩺️ Cumulative Infection & Vaccination Rates', expanded=True):
        st.altair_chart(plot_progressbar(timeseries_summary), use_container_width=True)

    with st.beta_expander('🌎️ Heatmap - ' + metric, expanded=True):
        st.altair_chart(plot_choropleth(geodata, summary, level, per100k).configure_view(strokeWidth=0),
                        use_container_width=True)
        st.altair_chart(plot_bubblechart(summary, level, per100k).configure_view(strokeWidth=0),
                        use_container_width=True)

    with st.beta_expander('〽️ Time Series - ' + metric, expanded=False):
        st.altair_chart(plot_timeseries(timeseries, level, metric), use_container_width=True)

    with st.beta_expander('💾 Data', expanded=False):
        st.dataframe(summary.set_index(level).iloc[:, 3:-1])
        #st.dataframe(timeseries_summary)


def desktop_view():
    with st.beta_expander(format_date(start_dt) + ' - ' + format_date(end_dt), expanded=False):
        st.write(', '.join(sorted(countries)))

    with st.beta_expander('🧭 Trends - ' + interval, expanded=True):
        t1, t2, t3, t4 = st.beta_columns([1, 1, 1, 2])
        with t1:
            st.altair_chart(plot_trend(timeseries_summary, 'Cases', '#faca2b', True),
                            use_container_width=True)
        with t2:
            st.altair_chart(plot_trend(timeseries_summary, 'Deaths', '#ff2b2b', True),
                            use_container_width=True)
        with t3:
            st.altair_chart(plot_trend_stacked(timeseries_summary, 'Vaccinations',
                                               'Partial Vaccinations', 'Full Vaccinations', False),
                            use_container_width=True)
        with t4:
            st.altair_chart(plot_progressbar(timeseries_summary), use_container_width=True)

    col_l, col_r = st.beta_columns(2)

    with col_l:
        with st.beta_expander('🌎️ Heatmap', expanded=True):
            st.altair_chart(plot_choropleth(geodata, summary, level, per100k).configure_view(strokeWidth=0, height=700),
                            use_container_width=True)
            st.altair_chart(plot_bubblechart(summary, level, per100k).configure_view(strokeWidth=0),
                            use_container_width=True)

    with col_r:
        with st.beta_expander('〽️ Time Series', expanded=True):
            st.altair_chart(plot_timeseries(timeseries, level, metric).configure_view(strokeWidth=0, height=400),
                            use_container_width=True)
        with st.beta_expander('💾 Data', expanded=True):
            #st.dataframe(summary.set_index(level).iloc[:, 3:-1], height=343)
            st.dataframe(timeseries_summary)


def banner(title, background='#F63366', img='data/covid_icon.png'):
    html = '<style>body { margin: 0; font-family: "IBM Plex Sans", Arial, Helvetica, sans-serif;} .header{padding: 10px 16px; ' + \
        'background: ' + background + ';color: #f1f1f1; top:0;}</style><div class="header" id="myHeader">' + title + \
        '<img src="data:image/png;base64,{}" '.format(img_to_bytes(img)) + \
           'height="30" align="right" class="img-fluid"></div>'

    return st.markdown(html, unsafe_allow_html=True)


def main(layout):
    if layout == 'Mobile':
        return mobile_view()
    elif layout == 'Desktop':
        return desktop_view()


st.set_page_config('Covid Dash', page_icon='🦠️', layout='wide')
# force altair tooltips in full-screen mode
st.markdown('<style>#vg-tooltip-element{z-index: 1000051}</style>', unsafe_allow_html=True)

# initialise variables
raw_data = '/Users/nick/Documents/GitHub/COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/'
geo_url = 'https://gist.githubusercontent.com/nodlind/6018712d70c3262c1f9f0f860988bc10/raw/3d714ed6a688755daa164003ae894eb9f7b5fd50/countries_regions_10m.json'
geo_feature = 'countries_regions_10m'
df = load_data('data/covid_data.csv')
metadata = load_metadata('data/metadata.csv')
geodata = load_geodata(geo_url, geo_feature)
disclaimer = '''About \n
Global COVID-19 situation dashboard powered by Streamlit. Analyse and explore COVID-19 epidemiological and vaccine data. \n
Epidemiological data taken from the COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) 
at Johns Hopkins University https://github.com/CSSEGISandData/COVID-19 \n
Vaccination data taken from the JHU GovEx repository https://github.com/govex/COVID-19'''

# ••••••••••••••••••••••••••••••••••••••••••••••• SIDEBAR ••••••••••••••••••••••••••••••••••••••••••••••••••••••••

st.sidebar.subheader('Settings')

layout = st.sidebar.radio('Layout', options=['Mobile', 'Desktop'], help='Mobile layout sets out dashboard elements in one column, desktop in two')

metric_type = st.sidebar.radio('Metric', options=['Cases', 'Deaths', 'Partial Vaccinations', 'Full Vaccinations'])
cumulative = st.sidebar.checkbox('Cumulative', value=False, help='Show time series data as cumulative')
rolling = st.sidebar.checkbox('Rolling Average', value=True, help='Show daily trend and time series data as a 14 day rolling average')
per100k = st.sidebar.checkbox('Per 100k', value=True, help='Show trend, heatmap and time series data relative to population size')

level = st.sidebar.radio('Level of Detail', options=['Country', 'Region'], help='Define level of detail in heatmap and timeseries charts')
label = st.sidebar.checkbox('Labels', help='Add labels to heatmap')
interval = st.sidebar.radio('Interval', options=['Daily', 'Weekly', 'Monthly'], index=1, help='Define time interval for trends and timeseries charts')
interval_cd = interval[0]

st.sidebar.markdown('---')

st.sidebar.subheader('Filters')

date_rng = pd.date_range(start=df.Date.min(), end=df.Date.max(), freq=interval_cd)
start_dt, end_dt = st.sidebar.select_slider('Date Range', options=date_rng,
                                            value=[date_rng[date_rng.year == 2021][0], date_rng.max()],
                                            format_func=lambda x: format_date(x))

country_container = st.sidebar.beta_container()
country_container.empty()

button_col1, button_col2, _ = st.sidebar.beta_columns([0.8, 1, 2])
europe_button = button_col1.button('EUR')
latam_button = button_col2.button('LATAM')

if europe_button:
    default_countries = ['Albania', 'Andorra', 'Austria', 'Belgium', 'Bosnia and Herzegovina', 'Belarus', 'Bulgaria',
                         'Czechia', 'Croatia', 'Denmark', 'Estonia', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland',
                         'Ireland', 'Faroe Islands', 'Finland', 'Italy', 'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania',
                         'Luxembourg', 'Moldova', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 'Poland',
                         'Portugal', 'Romania', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland',
                         'Ukraine', 'United Kingdom']
elif latam_button:
    default_countries = ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Paraguay', 'Peru',
                         'Uruguay', 'Venezuela']
else:
    default_countries = []

with country_container:
    select_countries = st.multiselect('Countries', options=sorted(df.Country.unique()), default=default_countries)

if select_countries == []:
    countries = sorted(df.Country.unique())
else:
    countries = select_countries

region_container = st.sidebar.beta_container()
region_container.empty()

default_regions = sorted(df[df['Country'].isin(countries)].Region.unique())
if level == 'Region':
    select_regions = st.sidebar.multiselect('Regions', options=default_regions, default=default_regions)
    regions = select_regions if select_regions != [] else default_regions
else:
    regions = default_regions

# refresh_data = st.sidebar.button('Refresh Data')
# if refresh_data:
#     collate_data(raw_data, 'data/keys.csv', 'data/covid_data.csv')


# ••••••••••••••••••••••••••••••••••••••••••••••• MAIN ••••••••••••••••••••••••••••••••••••••••••••••••••••••••

metric = determine_metric(metric_type, cumulative, rolling, per100k, interval)
subset = filter_data(df, countries, regions, start_dt, end_dt, level, interval_cd)
timeseries = calculate_metrics(subset, level, 'geo_temp', interval, metadata)
summary = calculate_metrics(subset, level, 'geo', interval, metadata)
timeseries_summary = calculate_metrics(subset, level, 'temp', interval, metadata)

banner('COVID-19 Situation Dashboard')
main(layout)
st.markdown('---')
st.caption(disclaimer)
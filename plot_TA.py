'''
changelog:
Version  | Description
0.1      | Changed RSI/Stoch to candle pattern
'''
from tqdm import tqdm
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import investpy
import yfinance as yf
import talib
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
from matplotlib.pylab import date2num
 
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models.widgets import Select
from bokeh.io import output_file, show
from bokeh.layouts import row, column, gridplot, widgetbox

from bokeh.models import BooleanFilter, CDSView, Select, Range1d, HoverTool, DatetimeTickFormatter, ColorBar, LinearColorMapper, CustomJS, CrosshairTool, Span
from bokeh.models import Label, Title, HoverTool, Patch, Legend, LegendItem, Band, LinearAxis, BoxAnnotation
from bokeh.models import DataTable, DateFormatter, TableColumn, NumberFormatter
from bokeh.palettes import Category20, Viridis256
from bokeh.models.formatters import NumeralTickFormatter

def addSpans(plots):
    spanH = Span(location=0, dimension='height', line_dash='dashed', line_width=0.5, line_color='dimgray', visible=False)
    spanW = [Span(location=0, dimension='width', line_dash='dashed', line_width=0.5, line_color='dimgray', visible=False) for p in plots]
    labelW = [Label(x=0, y=70, visible=False, 
                  text='', text_font_size="11px", text_color='whitesmoke', text_baseline='middle',
                  border_line_color='darkgray', border_line_alpha=1.0,
                  background_fill_color='darkgray', background_fill_alpha=1.0) for p in plots]
    
    js_move = """
        spanH.visible = true
        spanW.visible = true
        labelW.visible = true
        spanH.location=cb_obj.x
        spanW.location=cb_obj.y
        labelW.x = p.x_range.start
        labelW.y = cb_obj.y
        labelW.text = '\t' + cb_obj.y.toFixed(2) +'\t'
    """
    js_leave = """
        spanH.visible = false
        spanW.visible = false
        labelW.visible = false
    """
    for i, p in enumerate(plots):
        p.add_layout(spanH)
        p.add_layout(spanW[i])
        p.add_layout(labelW[i])

        args = {'spanH': spanH, 'spanW': spanW[i], 'labelW':labelW[i], 'p':p}
        p.js_on_event('mousemove', CustomJS(args=args, code=js_move))
        p.js_on_event('mouseleave', CustomJS(args=args, code=js_leave))

def cleanDate(p, df):
    p.xaxis.major_label_overrides = {
        i: date.strftime('%b %d') for i, date in enumerate(pd.to_datetime(df["Date"]))
    }

def plot_chart_bokeh(processed, dataCollection, keys, outputfilename):
    stock = processed['data']

    # Define constants
    W_PLOT = 1000
    H_PLOT = 360
    TOOLS = 'pan,wheel_zoom,reset'

    VBAR_WIDTH = 1*12*60*60*1000 # one day in ms
    RED = Category20[7][6]
    GREEN = Category20[5][4]

    BLUE = Category20[3][0]
    BLUE_LIGHT = Category20[3][1]

    ORANGE = Category20[3][2]
    PURPLE = Category20[9][8]
    BROWN = Category20[11][10]

    # ==========================================================================
    # ===================       PLOT CANDLE STICK GRAPH     ====================
    # ==========================================================================
    p1 = figure(plot_width=W_PLOT, plot_height=H_PLOT, tools=TOOLS, toolbar_location='right')

    inc = stock.data['Close'] >= stock.data['Open']
    dec = stock.data['Open'] > stock.data['Close']
    # limit = stock.data['ZC_d2/dt2'] > 10
    # limit = stock.data['ZC_d/dt'] > 0

    # view_inc = CDSView(source=stock, filters=[BooleanFilter(inc), BooleanFilter(limit)])
    # view_dec = CDSView(source=stock, filters=[BooleanFilter(dec), BooleanFilter(limit)])
    view_inc = CDSView(source=stock, filters=[BooleanFilter(inc)])
    view_dec = CDSView(source=stock, filters=[BooleanFilter(dec)])

    # # map dataframe indices to date strings and use as label overrides
    p1.y_range.start = 0.9 * min(stock.data['Low'])
    p1.y_range.end = 1.1 * max(stock.data['High'])
    p1.segment(x0='Date', x1='Date', y0='Low', y1='High', color=GREEN, source=stock, view=view_inc)
    p1.segment(x0='Date', x1='Date', y0='Low', y1='High', color=RED, source=stock, view=view_dec)

    vb1 = p1.vbar(x='Date', width=VBAR_WIDTH, top='Open', bottom='Close', fill_color='forestgreen', fill_alpha=1, line_color='forestgreen',
           source=stock,view=view_inc, name="price")
    vb2 = p1.vbar(x='Date', width=VBAR_WIDTH, top='Open', bottom='Close', fill_color='orangered', fill_alpha=1, line_color='orangered',
           source=stock,view=view_dec, name="price")
    
    # Bollinger band plot
    patch1 = p1.varea(x='Date', y1='lowerband', y2='upperband', source=stock, fill_alpha=0.05, fill_color='dodgerblue')
    patch_line1 = p1.line(x='Date', y='lowerband', source=stock, line_color='blue', line_alpha=0.4)
    patch_line2 = p1.line(x='Date', y='middleband', source=stock, line_color='grey', line_alpha=0.8, line_dash='dotdash')
    patch_line3 = p1.line(x='Date', y='upperband', source=stock, line_color='blue', line_alpha=0.4)

    # ZC Line plot
    zc_7 = p1.line(x='Date', y='ma7', source=stock, line_color='crimson', line_alpha=0.4)
    zc_26 = p1.line(x='Date', y='ma26', source=stock, line_color='darkslateblue', line_alpha=0.4)

    # # Resistance plots
    # r1 = p1.line(x='Date', y='r1', source=stock, line_color='forestgreen', line_dash='dotdash', line_alpha=0.6)
    # r2 = p1.line(x='Date', y='r2', source=stock, line_color='forestgreen', line_dash='dotdash', line_alpha=0.8)
    # r3 = p1.line(x='Date', y='r3', source=stock, line_color='forestgreen', line_dash='dotdash', line_alpha=1.0)

    # # Support plots
    # s1 = p1.line(x='Date', y='s1', source=stock, line_color='crimson', line_dash='dotdash', line_alpha=0.6)
    # s2 = p1.line(x='Date', y='s2', source=stock, line_color='crimson', line_dash='dotdash', line_alpha=0.8)
    # s3 = p1.line(x='Date', y='s3', source=stock, line_color='crimson', line_dash='dotdash', line_alpha=1.0)

    # Extrema plots
    # minima = p1.inverted_triangle(x='Date', y='minima', source=stock, size=5, color="goldenrod", alpha=0.5)
    # maxima = p1.triangle(x='Date', y='maxima', source=stock, size=5, color="teal", alpha=0.5)
    # minima = p1.circle(
    #     x='Date', y='minima', source=stock, size=10,
    #     fill_color="grey", hover_fill_color="firebrick",
    #     fill_alpha=0.2, hover_alpha=0.8, hover_line_color="white")
    # maxima = p1.triangle(
    #     x='Date', y='maxima', source=stock,
    #     size=10, fill_color="grey", fill_alpha=0.2,
    #     hover_fill_color="firebrick", hover_alpha=0.8, hover_line_color="white")

    # Volume plot
    # Setting the second y axis range name and range
    p1.extra_y_ranges = {"vol_axis": Range1d(start=0, end=max(stock.data['Volume']) * 4)}
    # Adding the second axis to the plot.  
    p1.add_layout(LinearAxis(y_range_name="vol_axis", visible=False), 'right')
    vol_inc = p1.vbar(x="Date", top="Volume", bottom=0, width=int(VBAR_WIDTH * 2), fill_color=GREEN, fill_alpha=0.1, line_color=GREEN, line_alpha=0.2,
            source=stock, view=view_inc, y_range_name="vol_axis")
    vol_dec = p1.vbar(x="Date", top="Volume", bottom=0, width=int(VBAR_WIDTH * 2), fill_color=RED, fill_alpha=0.1, line_color=RED, line_alpha=0.2,
            source=stock, view=view_dec, y_range_name="vol_axis")

    legend = Legend(items=[
        LegendItem(label="All", renderers=[
                   patch1, patch_line1, patch_line2, patch_line3, vol_inc, vol_dec, zc_7, zc_26,
                #    s1, s2, s3,r1, r2, r3, 
                #    minima, maxima
                   ], index=0),
        LegendItem(label="BB", renderers=[
                   patch1, patch_line1, patch_line2, patch_line3], index=1),
        LegendItem(label="Volume", renderers=[vol_inc, vol_dec], index=2),
        LegendItem(label="ZC", renderers=[zc_7, zc_26], index=3),
        LegendItem(label="MA7", renderers=[zc_7], index=4),
        LegendItem(label="MA26", renderers=[zc_26], index=5),
        # LegendItem(label="Support", renderers=[s1, s2, s3], index=6),
        # LegendItem(label="Resistance", renderers=[r1, r2, r3], index=7),
        # LegendItem(label="Extrema", renderers=[minima, maxima], index=8)
    ])
    p1.add_layout(legend)
    p1.legend.location = "top_left"
    p1.legend.border_line_alpha = 0
    p1.legend.background_fill_alpha = 0
    p1.legend.click_policy = "hide"
    p1.legend.orientation = "horizontal"
    # p1.add_layout(Title(text="Stock price", align="left"), "left")

    p1.yaxis.axis_label = 'Stock price'
    p1.yaxis.formatter = NumeralTickFormatter(format='0.00')
    p1.x_range.range_padding = 0.05
    p1.xaxis.ticker.desired_num_ticks = 40
    p1.xaxis.major_label_orientation = 3.14/4
    p1.xaxis.visible = False
    p1.xgrid.grid_line_color = None
    p1.ygrid.grid_line_color = None

    # Select specific tool for the plot
    p1.add_tools(HoverTool(
        tooltips=[
            ("Datetime", "@Date{%Y-%m-%d}"),
            ("Open", "@Open{0,0.00}"),
            ("Close", "@Close{0,0.00}"),
            ("Volume", "@Volume{(0.00 a)}")
        ],

        formatters={"@Date": 'datetime'},

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='vline',

        renderers=[vb1, vb2]
    ))

    # ==========================================================================
    # ===================      PLOT STOCH / RSI GRAPH          =================
    # ==========================================================================
    p2 = figure(plot_width=W_PLOT, plot_height=int(H_PLOT/4), tools=TOOLS, toolbar_location='above', x_range=p1.x_range, 
                x_axis_type='datetime') # , y_range=(-20, 120)

    stoch_k = p2.line(x='Date', y='slowk', source=stock, line_color='royalblue', alpha=0.8, muted_alpha=0.2)
    stoch_d = p2.line(x='Date', y='slowd', source=stock, line_color='tomato', alpha=0.8, muted_alpha=0.2)
    rsi = p2.line(x='Date', y='rsi', source=stock, line_color='gray', alpha=0.8, muted_alpha=0.2)
    mid_box = BoxAnnotation(bottom=20, top=80, fill_alpha=0.2, fill_color='palegreen', line_color='lightcoral', line_alpha=0.4, line_dash='dashed')
    # candle = p2.line(x='Date', y='candle', source=stock, line_color='royalblue', alpha=0.8, muted_alpha=0.2)
    # mid_box = BoxAnnotation(bottom=-300, top=300, fill_alpha=0.2, fill_color='palegreen', line_color='lightcoral', line_alpha=0.4, line_dash='dashed')
    legend = Legend(items=[
        LegendItem(label="Stoch", renderers=[stoch_k, stoch_d], index=0),
        LegendItem(label="RSI", renderers=[rsi], index=1)
        # LegendItem(label="Candle", renderers=[candle], index=1)
    ])
    p2.add_layout(legend)
    p2.add_layout(mid_box)
    # p2.add_layout(lower)
    zero = Span(location=0, dimension='width', line_color='seagreen', line_dash='solid', line_width=0.8)
    p2.add_layout(zero)
    p2.yaxis.axis_label = 'Stochastic / RSI'
    p2.x_range.range_padding = 0.05
    # p2.toolbar.autohide = True
    p2.xaxis.visible = False
    p2.legend.location = "top_left"
    p2.legend.border_line_alpha = 0
    p2.legend.background_fill_alpha = 0
    p2.legend.click_policy = "mute"
    p2.xgrid.grid_line_color = None
    p2.ygrid.grid_line_color = None

    # ==========================================================================
    # ===================                 Plot MACD         ====================
    # ==========================================================================
    y_limit = abs(max(stock.data['macd_hist'], key=abs))
    y2_limit = abs(max(stock.data['macd_d/dt'], key=abs))
    p3 = figure(plot_width=W_PLOT, plot_height=int(H_PLOT/2.5), tools=TOOLS, toolbar_location='above',
                x_range=p1.x_range, x_axis_type='datetime', y_range=(-y_limit, y_limit))
    mapper = LinearColorMapper(palette=Viridis256)
    macd_line = p3.line(x='Date', y='macd_hist', source=stock, line_color='darkgreen', alpha=0.8, muted_alpha=0.2)
    # macd_hist = p3.vbar_stack(['macd'], x='Date', source=stock, width=int(VBAR_WIDTH * 2), fill_color={'field':'macd', 'transform': mapper})

    mid_box = BoxAnnotation(bottom=-0.5, top=0.5, fill_alpha=0.2, fill_color='blanchedalmond', line_color='grey', line_alpha=0.4, line_dash='dashed')
    zero = Span(location=0, dimension='width', line_color='seagreen', line_dash='solid', line_width=0.8)
    p3.add_layout(zero)
    p3.add_layout(mid_box)

    # Setting the second y axis range name and range
    p3.extra_y_ranges = {"extra_y_axis": Range1d(start=-y2_limit, end=y2_limit)}
    # Adding the second axis to the plot.  
    p3.add_layout(LinearAxis(y_range_name="extra_y_axis"), 'right')
    macd_v = p3.line(x='Date', y='macd_d/dt', source=stock, line_color='dodgerblue', line_dash='solid', alpha=0.8, muted_alpha=0.2, y_range_name="extra_y_axis")
    macd_acc = p3.line(x='Date', y='macd_d2/dt2', source=stock, line_color='tomato', line_dash='dotdash', alpha=0.8, muted_alpha=0.2, y_range_name="extra_y_axis")

    legend = Legend(items=[
        LegendItem(label="MACD", renderers=[macd_line], index=0),
        LegendItem(label="MACD-v", renderers=[macd_v], index=1),
        LegendItem(label="MACD-a", renderers=[macd_acc], index=2)
    ])
    p3.add_layout(legend)
    p3.legend.location = "top_left"
    p3.legend.border_line_alpha = 0
    p3.legend.background_fill_alpha = 0
    p3.legend.click_policy = "mute"
    p3.legend.orientation = "horizontal"
    # p3.add_layout(Title(text="MACD", align="center"), "left")
    p3.yaxis.axis_label = 'MACD'
    p3.x_range.range_padding = 0.05
    p3.xaxis.visible = False
    p3.xaxis.ticker.desired_num_ticks = 40
    p3.xaxis.major_label_orientation = 3.14/4
    p3.toolbar.autohide = True
    p3.xgrid.grid_line_color = None
    p3.ygrid.grid_line_color = None

    # ==========================================================================
    # ===================         Plot ZC        ====================
    # ==========================================================================
    y_limit = abs(max(stock.data['ZC'], key=abs))
    y2_limit = abs(max(stock.data['ZC_d/dt'], key=abs))
    # y_limit = abs(max(stock.data['slowk'], key=abs))
    # y2_limit = abs(max(stock.data['slowk_d/dt'], key=abs))
    p4 = figure(plot_width=W_PLOT, plot_height=int(H_PLOT/3), tools=TOOLS, toolbar_location='above',
                x_range=p1.x_range, x_axis_type='datetime', y_range=(-y_limit, y_limit))

    p4.xaxis.formatter=DatetimeTickFormatter(
        hours=["%d.%m.%y"],
        days=["%d.%m.%y"],
        months=["%d.%m.%y"],
        years=["%d.%m.%y"],
    )
    # macd_v = p4.line(x='Date', y='macd_d/dt', source=stock, line_color='royalblue', alpha=0.8, muted_alpha=0.2)
    # macd_acc = p4.line(x='Date', y='macd_d2/dt2', source=stock, line_color='tomato', alpha=0.8, muted_alpha=0.2)
    # ad = p4.line(x='Date', y='ck_AD', source=stock, line_color='royalblue', alpha=0.8, muted_alpha=0.2)
    # adosc = p4.line(x='Date', y='ck_ADOSC', source=stock, line_color='tomato', alpha=0.8, muted_alpha=0.2)
    # obv = p4.line(x='Date', y='OBV', source=stock, line_color='darkgreen', alpha=0.8, muted_alpha=0.2)

    # Setting the second y axis range name and range
    p4.extra_y_ranges = {"extra_y_axis": Range1d(start=-y2_limit, end=y2_limit)}
    # Adding the second axis to the plot.  
    p4.add_layout(LinearAxis(y_range_name="extra_y_axis"), 'right')
    
    zc = p4.line(x='Date', y='ZC', source=stock, line_color='darkgreen', alpha=0.8, muted_alpha=0.2)
    zc_v = p4.line(x='Date', y='ZC_d/dt', source=stock, line_color='dodgerblue', line_dash='dotdash', alpha=0.8, muted_alpha=0.2, y_range_name="extra_y_axis")
    zc_a = p4.line(x='Date', y='ZC_d2/dt2', source=stock, line_color='tomato', line_dash='dotdash', alpha=0.8, muted_alpha=0.2, y_range_name="extra_y_axis")
    # slowk = p4.line(x='Date', y='slowk', source=stock, line_color='darkgreen', alpha=0.8, muted_alpha=0.2)
    # slowk_v = p4.line(x='Date', y='slowk_d/dt', source=stock, line_color='dodgerblue', line_dash='dotdash', alpha=0.8, muted_alpha=0.2, y_range_name="extra_y_axis")
    # slowk_a = p4.line(x='Date', y='slowk_d2/dt2', source=stock, line_color='tomato', line_dash='dotdash', alpha=0.8, muted_alpha=0.2, y_range_name="extra_y_axis")
    
    mid_box = BoxAnnotation(bottom=-0.5, top=0.5, fill_alpha=0.2, fill_color='blanchedalmond', line_color='grey', line_alpha=0.4, line_dash='dashed')
    zero = Span(location=0, dimension='width', line_color='seagreen', line_dash='solid', line_width=0.8)
    p4.add_layout(zero)
    p4.add_layout(mid_box)
    # p4.yaxis.axis_label = 'MACD v/acc'
    legend = Legend(items=[
        LegendItem(label="ZC", renderers=[zc], index=0),
        LegendItem(label="ZC-v", renderers=[zc_v], index=1),
        LegendItem(label="ZC-a", renderers=[zc_a], index=2),
        # LegendItem(label="slowk", renderers=[slowk], index=0),
        # LegendItem(label="slowk-v", renderers=[slowk_v], index=1),
        # LegendItem(label="slowk-a", renderers=[slowk_a], index=2)
    ])
    p4.add_layout(legend)
    p4.legend.location = "top_left"
    p4.legend.border_line_alpha = 0
    p4.legend.background_fill_alpha = 0
    p4.legend.click_policy = "mute"
    p4.legend.orientation = "horizontal"
    p4.x_range.range_padding = 0.05
    p4.xaxis.ticker.desired_num_ticks = 40
    p4.xaxis.major_label_orientation = 3.14/4
    p4.toolbar.autohide = True
    p4.xgrid.grid_line_color = None
    p4.ygrid.grid_line_color = None

    addSpans([p1, p2, p3, p4])

    columns = [
        TableColumn(field="Date", title="Date", formatter=DateFormatter(format='%d.%b')),
        # TableColumn(field="Open", title="Open", formatter=NumberFormatter(format='0.00')),
        # TableColumn(field="Close", title="Close", formatter=NumberFormatter(format='0.00')),
        TableColumn(field="ZC", title="ZC", formatter=NumberFormatter(format='0.000', text_align='right')),
        TableColumn(field="ZC_d/dt", title="ZC-v", formatter=NumberFormatter(format='0.000', text_align='right')),
        TableColumn(field="macd_hist", title="MACD", formatter=NumberFormatter(format='0.000', text_align='right')),
        TableColumn(field="macd_d/dt", title="MACD-v", formatter=NumberFormatter(format='0.000', text_align='right')),
        # TableColumn(field="macd_d2/dt2", title="MACD-a", formatter=NumberFormatter(format='0.000')),
        TableColumn(field="stoch", title="STOCH", formatter=NumberFormatter(format='0.0', text_align='right')),
        TableColumn(field="stoch-v", title="STOCH-v", formatter=NumberFormatter(format='0.0', text_align='right')),
        # TableColumn(field="slowk_d/dt", title="slowk-v", formatter=NumberFormatter(format='0.000')),
        # TableColumn(field="slowk_d2/dt2", title="slowk-a", formatter=NumberFormatter(format='0.000')),
    ]
    data_table = DataTable(source=stock, columns=columns, width=int(W_PLOT/3), height=int(H_PLOT * 2.2), index_position=None, width_policy='min')

    # ==========================================================================
    # ===================            SELECT WIDGET          ====================
    # ==========================================================================

    callback_select_main = """
        var d0 = s0.data;
        var symbol = cb_obj.value.split(" ")[0]
        var data_all = dataCollection[symbol]
        var data = data_all.data.data;

        /// Iterate over keys in new data and reassign old data with new data
        for (const key of Object.keys(data)) {
            d0[key] = []
            d0[key] = data[key]
        }
        s0.change.emit()

        /// Update y-axes range
        plot.y_range.have_updated_interactively = true
        plot.y_range.start = 0.9 * Math.min(...data['Low'])
        plot.y_range.end = 1.1 * Math.max(...data['High'])
        plot.extra_y_ranges['vol_axis'].have_updated_interactively = true
        plot.extra_y_ranges['vol_axis'].start  = 0
        plot.extra_y_ranges['vol_axis'].end  = Math.max(...data['Volume']) * 4
        """

    callback_select_va = """
        var symbol = cb_obj.value.split(" ")[0]
        var data_all = dataCollection[symbol]
        var data = data_all.data.data;
        var y_limit = Math.max.apply(null, data[param_main].map(Math.abs));
        var y_extra_limit = Math.max.apply(null, data[param_extra].map(Math.abs));
        /// Update y-axes range
        plot.y_range.have_updated_interactively = true
        plot.y_range.start = -y_limit
        plot.y_range.end = y_limit
        plot.extra_y_ranges['extra_y_axis'].have_updated_interactively = true
        plot.extra_y_ranges['extra_y_axis'].start  = -y_extra_limit
        plot.extra_y_ranges['extra_y_axis'].end  = y_extra_limit
        """
    selecthandler_main = CustomJS(args={'s0':stock, 'dataCollection':dataCollection, 'plot':p1}, code = callback_select_main)
    selecthandler_p3 = CustomJS(args={'dataCollection':dataCollection, 'plot':p3, 'param_main': 'macd_hist', 'param_extra': 'macd_d/dt'}, code = callback_select_va)
    selecthandler_p4 = CustomJS(args={'dataCollection':dataCollection, 'plot':p4, 'param_main': 'ZC', 'param_extra': 'ZC_d/dt'}, code = callback_select_va)
    # selecthandler_p4 = CustomJS(args={'dataCollection':dataCollection, 'plot':p4, 'param_main': 'slowk', 'param_extra': 'slowk_d/dt'}, code = callback_select_va)
    select = Select(title="Select:", value=keys[0], options=keys)
    select.js_on_change('value', selecthandler_main)
    select.js_on_change('value', selecthandler_p3)
    select.js_on_change('value', selecthandler_p4)

    # [cleanDate(x, stock.data) for x in [p1, p2, p3, p4]]
    # show the results
    gp1 = gridplot([select, data_table], ncols=1, plot_width=150, toolbar_options=dict(autohide=True))
    gp2 = gridplot([p1, p2, p3, p4], ncols=1, sizing_mode='scale_width', toolbar_location='right')

    output_file(outputfilename+'.html')
    show(row(gp1, gp2)) 
    # show(gp2) 
    return True

today = datetime.today().strftime("%d%m%Y")

# Data options
# outputfilename = 'Norway_' + today
# outputfilename = 'Malaysia_' + today
outputfilename = 'Holding_' + today
# outputfilename = 'Test_24082020'
# outputfilename = 'Norway_25092020'

# Read all sheets and return as dataframe
df_fromExcel = pd.read_excel(outputfilename + '.xlsx', sheet_name=None)
# Segregate stock list and data
df_list = df_fromExcel['list']
# Convert dataframe into bokeh CDS
dataCollection = dict()
keys = list()
for i, ticker in enumerate(df_fromExcel):
    if ticker != 'list':
        data = df_fromExcel[ticker]
        key = '{0} : {1}'.format(
            ticker,
            df_list[df_list['symbol']==ticker]['full_name'].values[0]
        )
        keys.append(key)
        if ticker == df_list['symbol'][0]:
            # Declare initial data. Needs to be separately declared to avoid JS callback changing initial data
            # Referring s1 from data collection will change the data at every callback
            s0 = {'data': ColumnDataSource(data)}
        dataCollection.update({ticker:{'data': ColumnDataSource(data)}})

plot_chart_bokeh(s0, dataCollection, keys, outputfilename)



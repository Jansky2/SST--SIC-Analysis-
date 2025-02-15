# SST-SIC Trend Analysis-
by using linear regression method trend has been calculated 


# for SST TREND 


import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import linregress
import matplotlib.ticker as mticker

file_path = "C:/Users/Janki/Downloads/MOMSIS/MOMSIS_SST_2011_2023_merged.nc"  
ds_sst = xr.open_dataset(file_path, decode_times=True)

lat = ds_sst["yt_ocean"]
lon = ds_sst["xt_ocean"]
time_dates = pd.to_datetime(ds_sst["TIME"].values)
sst = ds_sst["SST_REGRID"]

antarctic_mask = (lat >= -82) & (lat <= -55)

sst_antarctic = sst.where(antarctic_mask, drop=True)
lat_filtered = lat.where(antarctic_mask, drop=True)

start_date = pd.Timestamp("2015-01-01")
end_date = pd.Timestamp("2023-12-31")

time_mask = (time_dates >= start_date) & (time_dates <= end_date)
filtered_time_dates = time_dates[time_mask]

sst_antarctic = sst_antarctic.isel(TIME=time_mask)

sst_avg = sst_antarctic.mean(dim=("xt_ocean", "yt_ocean"))  # Area weighting happens automatically

df_sst = pd.DataFrame({"Date": filtered_time_dates, "SST (°C)": sst_avg.values})
df_sst = df_sst.set_index('Date')
df_sst_yearly_avg = df_sst.resample('Y').mean()
df_sst_yearly_avg = df_sst_yearly_avg.dropna()
if len(df_sst_yearly_avg) < 2:
    print("Not enough data points for trend analysis after NaN removal.")
    exit()

# Perform least squares fit
x = np.arange(len(df_sst_yearly_avg))
y = df_sst_yearly_avg["SST (°C)"].values

slope, intercept = np.polyfit(x, y, 1)
slope_sst_per_year = slope

# Statistical Significance
result = linregress(x, y)
p_value = result.pvalue

print(f"SST Trend: {slope_sst_per_year:.3f} °C per year")
print(f"P-value: {p_value:.3f}")  # Print p-value

# Plot 
plt.figure(figsize=(10, 5))
plt.plot(df_sst_yearly_avg.index.year, df_sst_yearly_avg["SST (°C)"], marker="o", linestyle="-", color="b", label="Yearly Avg SST")
plt.plot(df_sst_yearly_avg.index.year, slope * x + intercept, color="r", linestyle="--", label=f"Trend: {slope_sst_per_year:.3f}x + {intercept:.3f}")

plt.text(df_sst_yearly_avg.index.year[-1], slope * x[-1] + intercept, f"Slope = {slope_sst_per_year:.3f} °C per year\nP-value = {p_value:.3f}",
         color="r", fontsize=10, ha='left', va='center')

plt.xticks(df_sst_yearly_avg.index.year)
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

plt.grid(True)
plt.xlabel("Year")
plt.ylabel("SST (°C)")
plt.title("Yearly Average Antarctic SST Trend (55°S–82°S, 2015-2023)")
plt.legend()

def calculate_trend(data, x):
    return linregress(x, data)[0]  # Return only the slope

sst_yearly = sst_antarctic.resample(TIME="YS").mean()

# Compute trend at each grid point
trend = xr.apply_ufunc(
    calculate_trend,
    sst_yearly,
    input_core_dims=[["TIME"]],
    output_core_dims=[[]],
    vectorize=True,
    kwargs={"x": np.arange(len(sst_yearly["TIME"]))},
)

# Plot the trend map
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.SouthPolarStereo()})

levels = np.linspace(-0.3, 0.3, 21)

trend_plot = ax.contourf(lon, lat_filtered, trend, levels=levels, cmap=plt.cm.RdBu_r, extend='both', transform=ccrs.PlateCarree())

land_feature = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white')
ax.add_feature(land_feature, color='lightgray')
ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
ax.set_extent([-180, 180, -82, -55], crs=ccrs.PlateCarree())


for lon_line in [20, 90, 160, 230, 300]:
    ax.plot([lon_line, lon_line], [-90, -55], color='black', linestyle='-', linewidth=1.0, transform=ccrs.PlateCarree())

ax.text(20, -55, '(20)', fontsize=10, color='black', transform=ccrs.PlateCarree())
ax.text(90, -55, '(90)', fontsize=10, color='black', transform=ccrs.PlateCarree())
ax.text(160, -55, '(160)', fontsize=10, color='black', transform=ccrs.PlateCarree())
ax.text(230, -55, '(230)', fontsize=10, color='black', transform=ccrs.PlateCarree())
ax.text(300, -55, '(300)', fontsize=10, color='black', transform=ccrs.PlateCarree())

ax.text(40, -80, '(IO)', fontsize=14, color='black', transform=ccrs.PlateCarree())
ax.text(120, -80, '(PO)', fontsize=14, color='black', transform=ccrs.PlateCarree())
ax.text(200, -60, '(RS)', fontsize=14, color='black', transform=ccrs.PlateCarree())
ax.text(260, -55, '(BAS)', fontsize=14, color='black', transform=ccrs.PlateCarree())
ax.text(340, -65, '(WS)', fontsize=14, color='black', transform=ccrs.PlateCarree())

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1.0, color='gray', alpha=0.5, linestyle='--')
gl.ylocator = mticker.FixedLocator([-90, -80, -70, -60])
gl.ylabel_style = {'size': 10, 'color': 'black'}
gl.xlabel_style = {'size': 10, 'color': 'black'}
ax.axis('off')
gl.xlines = False
gl.xlines = False
ax.set_xticks([])
ax.set_yticks([])

cbar = plt.colorbar(trend_plot, ax=ax, orientation='vertical', shrink=0.6, label="SST Trend (°C/year)")

ax.set_title("Antarctic SST Trend (2015-2023) [°C/year]", fontsize=14)

plt.show()  

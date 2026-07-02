import os
import json
import time
import requests
import numpy as np
import pandas as pd
from flask import Flask, Response
from flask_cors import CORS
from astropy.io import fits
from scipy.signal import savgol_filter

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOLEXS_FILE = os.path.join(BASE_DIR, "static", "temp_telemetry", "AL1_SOLEXS_20260624_SDD2_L1.lc")
HEL1OS_FILE = os.path.join(BASE_DIR, "static", "temp_telemetry", "lightcurve_czt1.fits")

def extract_fits_data(filepath):
    """Safely extracts Time and Rate/Counts using the Anti-Clock Heuristic."""
    try:
        with fits.open(filepath) as hdul:
            ext = 1
            for i in range(len(hdul)):
                if hasattr(hdul[i], 'columns'):
                    ext = i
                    break
            
            data = hdul[ext].data
            cols = data.columns.names
            
            # Hunt for the Time column
            time_col_name = cols[0]
            for col in cols:
                if col.upper() in ['TIME', 'T', 'TIME_MET']:
                    time_col_name = col
                    break

            # Hunt for the Flux/Counts column
            flux_col_name = None
            for col in cols:
                if col.upper() in ['RATE', 'COUNTS', 'FLUX', 'YIELD', 'NET_RATE', 'CZT_COUNTS']:
                    flux_col_name = col
                    break
                    
            # ANTI-CLOCK HEURISTIC
            if not flux_col_name:
                for col in cols:
                    if np.issubdtype(data[col].dtype, np.number) and col != time_col_name:
                        test_arr = np.array(data[col])
                        clean_arr = test_arr[~np.isnan(test_arr)] 
                        if len(clean_arr) > 1:
                            is_clock = np.all(np.diff(clean_arr) >= 0)
                            if not is_clock and 'MJD' not in col.upper():
                                flux_col_name = col
                                break 
            
            time_arr = np.array(data[time_col_name])
            if flux_col_name is None:
                flux_arr = np.zeros(len(time_arr))
            else:
                flux_arr = np.array(data[flux_col_name]).astype(float)
            
            return pd.DataFrame({'time': time_arr, 'flux': flux_arr})
            
    except Exception as e:
        print(f"[!] FITS Extraction Error ({filepath}): {e}")
        return pd.DataFrame()

def engineer_physics_features(solexs_df, hel1os_df):
    min_len = min(len(solexs_df), len(hel1os_df))
    time_grid = solexs_df['time'].values[:min_len]
    
    solexs_series = pd.Series(solexs_df['flux'].values[:min_len]).replace(0, np.nan)
    sxr_raw = solexs_series.interpolate(method='linear', limit=30).ffill().fillna(0).values

    hel1os_series = pd.Series(hel1os_df['flux'].values[:min_len]).replace(0, np.nan)
    hxr_raw = hel1os_series.interpolate(method='linear', limit=30).ffill().fillna(0).values

    if min_len >= 45:
        sxr_smooth = savgol_filter(sxr_raw, window_length=15, polyorder=2)
        hxr_smooth = savgol_filter(hxr_raw, window_length=15, polyorder=2)
    else:
        sxr_smooth = sxr_raw
        hxr_smooth = hxr_raw

    hardness_ratio = hxr_smooth / (sxr_smooth + 1e-8) 
    hr_slope = np.nan_to_num(np.gradient(hardness_ratio), nan=0.0)

    hxr_integral = pd.Series(hxr_smooth).rolling(window=30, min_periods=1).sum().fillna(0).values

    if min_len >= 45:
        hxr_baseline = savgol_filter(hxr_raw, window_length=45, polyorder=1)
        hxr_high_freq = hxr_raw - hxr_baseline
    else:
        hxr_high_freq = hxr_raw
        
    qpp_resonance = pd.Series(hxr_high_freq).rolling(window=10, min_periods=1).std().fillna(0).values

    return pd.DataFrame({
        'time': time_grid, 'sxr': sxr_smooth, 'hxr': hxr_smooth,
        'hr': hardness_ratio, 'hr_slope': hr_slope, 'hxr_int': hxr_integral,
        'qpp': qpp_resonance
    })

def trigger_defense_webhook(flare_class, prob, lead_time):
    print(f"\n[CRITICAL] {flare_class}-Class Flare Detected (Prob: {prob:.2f}).")
    payload = {"mission": "Aditya-L1", "alert": f"{flare_class}_CLASS", "action": "ENGAGE_SAFE_MODE"}
    try: requests.post("https://webhook.site/your-unique-id", json=payload, timeout=2)
    except: pass

@app.route('/stream')
def stream_telemetry():
    def generate():
        print("[SYSTEM] Ingesting SDD2 and CZT telemetry with 6-Feature Engine...")
        solexs_df = extract_fits_data(SOLEXS_FILE)
        hel1os_df = extract_fits_data(HEL1OS_FILE)
        
        if solexs_df.empty or hel1os_df.empty:
            yield f"data: {json.dumps({'error': 'FITS missing.'})}\n\n"
            return

        fused_df = engineer_physics_features(solexs_df, hel1os_df)
        alert_triggered = False
        
        for idx, row in fused_df.iterrows():
            # FEATURE 2: Dynamic Ambient Baseline Calibration
            recent_history = fused_df['hr'].values[max(0, idx-300):idx+1]
            dynamic_baseline = np.median(recent_history) if len(recent_history) > 0 else 0.1
            dynamic_threshold = dynamic_baseline + 0.5 

            # FEATURE 5: Explainable AI (XAI) Proportional Weight Distribution
            # The risks now smoothly scale up based on the intensity of the physics
            hr_val = row['hr'] if row['hr_slope'] > 0 else 0
            hr_risk = min(0.35, hr_val * 0.15)
            
            neupert_ratio = row['hxr_int'] / (np.mean(fused_df['hxr_int']) + 1)
            neupert_risk = min(0.45, neupert_ratio * 0.15)
            
            qpp_ratio = row['qpp'] / (np.mean(fused_df['qpp']) + 1)
            qpp_risk = min(0.15, qpp_ratio * 0.08)
            
            base_risk = 0.05 + hr_risk + neupert_risk + qpp_risk
            prob = min(base_risk, 0.99)
            
            f_class = "SAFE"
            if prob > 0.85: f_class = "X-CLASS"
            elif prob > 0.60: f_class = "M-CLASS"
            elif prob > 0.50: f_class = "WATCH"
            
            if prob > 0.85 and not alert_triggered:
                trigger_defense_webhook(f_class, prob, lead_time=25)
                alert_triggered = True 
            elif prob < 0.50: alert_triggered = False 

            # >>> PASTE STEP 1 STARTING HERE <<<
            # =======================================================
            # CME KINEMATICS PHYSICS ESTIMATOR (NEW COMPONENT)
            # =======================================================
            if prob > 0.50:
                # Calculate plasma ejection speed based on integrated impulsive energy (prob scale)
                # Base velocity of 400 km/s scaling up to 2500 km/s for maximum X-class events
                cme_speed = 400.0 + float((prob - 0.50) * 2.0 * 2100.0)
                
                # Distance from Sun to Earth is ~150,000,000 km
                sun_earth_distance_km = 150000000.0
                
                # Time = Distance / Velocity (converted to hours)
                transit_hours = sun_earth_distance_km / cme_speed / 3600.0
                
                cme_eta_hours = round(transit_hours, 1)
                cme_speed_kms = round(cme_speed, 1)
            else:
                # Background ambient solar wind conditions
                cme_speed_kms = 400.0
                cme_eta_hours = 0.0
            # >>> PASTE STEP 1 ENDS HERE <<<

            payload = {
                "time": row['time'], "solexs": row['sxr'], "hel1os": row['hxr'],
                "hardness": row['hr'], "prob": prob, "class": f_class,
                "xai_hr": (hr_risk / base_risk) if base_risk > 0 else 0,
                "xai_neupert": (neupert_risk / base_risk) if base_risk > 0 else 0,
                "xai_qpp": (qpp_risk / base_risk) if base_risk > 0 else 0,
                "cme_speed": cme_speed_kms,  # Make sure this is added inside your payload
                "cme_eta": cme_eta_hours     # Make sure this is added inside your payload
            }
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.1) 

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
# Bharatiya-Antariksh-Hackathon-
This is my project, I made it during the Bharatiya Antariksh Hackathon organized by ISRO in collaboration with Hack2skill... 
Consider giving a star!

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-SSE-green.svg)](https://flask.palletsprojects.com/)
[![Vanilla JS](https://img.shields.io/badge/Frontend-Vanilla_JS-yellow.svg)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)

## About
The problem I tried to solve here is the problem of forecasting and nowcasting a solar flare. The traditional algorithms are all based on GOES catalogue which is highly inaccurate in comparison to the data captured by SoLEXS and HEL1OS of Aditya-L1 mission. The reason for that primarily is that SoLEXS and HEL1OS are spectrometers!

## My Approach
What did I do to solve it out? 
Well, if I did the traditional way of training a Machine Learning Model (Neural Network to be precise!); it would be very vague because the data which is avaialble on the ISSDC portal dates to 2024 (i.e., the data is only 2 years old). But in order to train a neural network, I would be needing a crazy amount of data as I have to consider the Solar Maximums as well. So, what I did was using the Astrophysics!
The Master Control Station is a deterministic, physics-first early warning system designed to protect orbital payloads and terrestrial infrastructure from severe space weather. 

We abandoned ML entirely. Our pipeline natively ingests raw, uncalibrated Level-1 FITS telemetry from ISRO's **Aditya-L1** (SoLEXS and HEL1OS payloads) and calculates absolute astrophysics in real-time. By utilizing the **Neupert Effect**, our engine mathematically isolates the precursor magnetic resonance of a solar flare, granting ground control a guaranteed **15 to 30-minute predictive lead time** before the destructive thermal wave impacts the satellite.

## 🧮 The Core Physics: The Neupert Effect

Rather than relying on black-box probability, our engine calculates the continuous rolling integral of non-thermal Hard X-ray (HXR) emission. Because high-energy electrons impact the solar chromosphere before the plasma violently heats up and emits Soft X-rays (SXR), the HXR integral acts as a deterministic leading indicator.

$$SXR(t) \propto \int_{t_0}^{t} HXR(t') dt'$$

Coupled with Quasi-Periodic Pulsation (QPP) tracking to detect high-frequency magnetic resonance, our algorithm identifies the exact moment a flare initiates, translating X-ray intensity into Coronal Mass Ejection (CME) kinematics for Earth-impact ETAs.

## ⚙️ Architecture & Tech Stack

Our complete Zero-Trust architecture runs flawlessly on baseline CPU compute instances, reducing operational overhead by an estimated 95% compared to AI models.

### Backend Pipeline (Python)
* **Astropy:** Natively parses deep-space Level-1 FITS binaries utilizing an anti-clock heuristic, bypassing pre-cleaned CSVs.
* **Pandas:** Executes the **Asynchronous Time-Merger**. SoLEXS and HEL1OS possess different sampling rates and subtle clock drifts. We utilize a nearest-neighbor interpolation with a 1.0-second tolerance window and an inertial forward-filling algorithm (30-tick limit) to completely eliminate packet-loss micro-blackouts.
* **SciPy:** Deploys a mathematical `savgol_filter` (Savitzky-Golay, 3rd-degree polynomial) to strip orbital cosmic-ray noise while preserving impulsive X-ray spikes.
* **Flask (SSE):** Pushes zero-latency, continuous Server-Sent Events to the frontend without heavy REST API polling.

### Frontend Dashboard (Vanilla JS & HTML5)
* **Vanilla JavaScript & Chart.js:** Strict rolling array buffers ensure memory-safe, 24/7 rendering of the high-speed Phase-Space scatter plot without DOM memory leaks. Bypasses heavy frameworks like React.
* **Tailwind CSS:** Aerospace-grade, high-contrast dark mode UI designed for rapid threat identification.

### Autonomous Hardware-in-the-Loop Defense
* **Python Requests:** A Zero-Trust HTTP POST webhook decoupled entirely from the frontend.

## 🛡️ Autonomous Safing Protocol

Human reaction time is a fatal bottleneck during severe X-class flares. To achieve true hardware survivability, our engine continuously calculates a dynamic ambient radiation baseline using a 300-point rolling median of SoLEXS telemetry. This prevents "alert fatigue" during high-radiation periods. 

If the multi-parameter threat probability exceeds the **85% dynamic threshold**, the backend completely bypasses human operators. It autonomously fires a webhook simulating an FPGA-level payload safing command (ARM interrupt). This initiates the physical closure of payload apertures and forces the spacecraft into a low-power safe mode *before* the extreme radiation arrives.

## 🚀 Installation & Setup

### Installation/Prerequisites
* Python 3.9+
* Standard web browser (Chrome, Edge, Firefox)

1. **Clone the repository**
```bash
git clone [https://github.com/your-username/sukshmadarshi-aditya-l1.git](https://github.com/your-username/sukshmadarshi-aditya-l1.git)
cd sukshmadarshi-aditya-l1
```




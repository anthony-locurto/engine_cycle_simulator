# Engine Cycle Thermodynamic Simulator

An interactive 1D thermodynamic cycle simulator for undergraduate-level education and UAV propulsion design exploration. Built with Streamlit and Plotly.

## Features

- **Three ideal cycles:** Otto, Diesel, and Brayton
- **Live P-V and T-S diagrams** with color-coded phase traces
- **ISA standard atmosphere model** — drag the altitude slider and T₁/P₁ update automatically
- **Sea-level reference traces** on all diagrams when operating at altitude, making the thermodynamic penalty immediately visible
- **Efficiency vs. compression ratio** curves comparing all three cycles simultaneously
- **Energy balance** bar chart (Q_in, W_net, Q_out)

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Community Cloud

1. Fork or push this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as the main file
4. Click **Deploy** — live in ~60 seconds

## Physics Notes

All cycles use ideal gas assumptions with fixed heat addition:
- Otto / Diesel: q_in = 1800 kJ/kg
- Brayton: q_in = 1200 kJ/kg

Atmosphere uses the ISA standard day model (troposphere + lower stratosphere).

## Context

Originally developed to explore small-scale engine design constraints for Group 2 UAV long-endurance missions at altitude, with particular focus on how inlet conditions affect cycle performance and BSFC.

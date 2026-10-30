"""
Interactive Clean & Minimal Prototype Server for Biochar Optimization Demo
Clean black-and-white minimalist enterprise styling.
Zero dependencies needed. Runs locally in any web browser.
"""

import http.server
import json
import os
import socketserver
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
sys.path.append(str(PROJECT_ROOT))

from real_life.evaluate import load_trained_ensemble
from real_life.data.dataloader import load_real_life_data
from real_life.acquisition.optimizer import recommend_next_experiments
from utils.transforms import clr_transform
from utils.ode_integrator import KineticDecompositionODE

# Load configuration and ensemble models
cfg_file = PROJECT_ROOT / "real_life" / "config.yaml"
with open(cfg_file, "r") as f:
    config = yaml.safe_load(f)

if not Path(config["data"]["raw_path"]).is_absolute():
    config["data"]["raw_path"] = str(REPO_ROOT / config["data"]["raw_path"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, _, scalers = load_real_life_data(config)
models = load_trained_ensemble(config, feedstock_dim=4, device=device)
ode_solver = KineticDecompositionODE(learnable=False)

PRESET_FEEDSTOCKS = {
    "Rice Straw": {"hemi": 22.0, "cel": 39.2, "lig": 22.0, "ext": 16.8, "ash": 22.55},
    "Rice Husk": {"hemi": 40.39, "cel": 46.75, "lig": 3.71, "ext": 9.15, "ash": 14.57},
    "Corn Cobs": {"hemi": 31.0, "cel": 42.0, "lig": 18.0, "ext": 9.0, "ash": 4.50},
    "Wheat Straw": {"hemi": 28.0, "cel": 38.0, "lig": 20.0, "ext": 14.0, "ash": 8.20},
    "Pine Wood": {"hemi": 25.0, "cel": 43.0, "lig": 28.0, "ext": 4.0, "ash": 1.50},
}


HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Biochar Yield & Process Optimization</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; }
    body { background-color: #ffffff; color: #111827; padding: 32px 24px; max-width: 1200px; margin: 0 auto; line-height: 1.5; }
    
    .top-bar { border-bottom: 2px solid #111827; padding-bottom: 16px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: flex-end; }
    h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.5px; }
    .subtitle { color: #4b5563; font-size: 14px; margin-top: 4px; }
    .system-status { font-size: 12px; font-weight: 600; color: #111827; border: 1px solid #111827; padding: 4px 8px; border-radius: 4px; }

    .grid { display: grid; grid-template-columns: 340px 1fr; gap: 32px; }
    .panel { border: 1px solid #e5e7eb; border-radius: 6px; padding: 20px; background: #fafafa; }
    .panel-title { font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 16px; border-bottom: 1px solid #e5e7eb; padding-bottom: 8px; }

    .form-group { margin-bottom: 14px; }
    label { display: block; font-size: 13px; font-weight: 600; color: #374151; margin-bottom: 4px; }
    select, input[type="range"] { width: 100%; padding: 8px; background: #ffffff; border: 1px solid #d1d5db; border-radius: 4px; font-size: 13px; }
    .slider-row { display: flex; align-items: center; gap: 10px; }
    .slider-val { min-width: 45px; text-align: right; font-size: 13px; font-weight: 700; color: #111827; }

    .btn {
      width: 100%; padding: 11px; background: #111827; color: #ffffff; border: 1px solid #111827; border-radius: 4px; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.15s; margin-top: 10px;
    }
    .btn:hover { background: #374151; }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }

    .kpi-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }
    .kpi-card { border: 1px solid #e5e7eb; border-radius: 6px; padding: 16px; background: #ffffff; }
    .kpi-label { font-size: 12px; font-weight: 600; text-transform: uppercase; color: #6b7280; letter-spacing: 0.5px; }
    .kpi-value { font-size: 26px; font-weight: 700; color: #111827; margin-top: 6px; }

    .table-container { border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden; margin-bottom: 24px; background: #ffffff; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th { background: #f3f4f6; text-align: left; padding: 10px 12px; font-weight: 600; color: #374151; border-bottom: 1px solid #e5e7eb; }
    td { padding: 12px; border-bottom: 1px solid #f3f4f6; color: #111827; }
    tr.best-row { background: #f9fafb; font-weight: 600; }
    .badge { display: inline-block; padding: 2px 6px; border: 1px solid #111827; border-radius: 3px; font-size: 11px; font-weight: 700; }

    .chart-box { border: 1px solid #e5e7eb; border-radius: 6px; padding: 16px; background: #ffffff; }
    .chart-container { position: relative; height: 240px; }
  </style>
</head>
<body>

  <div class="top-bar">
    <div>
      <h1>Biochar Production Optimizer</h1>
      <div class="subtitle">Physics-Informed Neural Network (PINN) • Closed-Loop Process Setpoint Recommendation</div>
    </div>
    <div class="system-status" id="systemStatus">System: Ready</div>
  </div>

  <div class="grid">
    <!-- INPUT COLUMN -->
    <div class="panel">
      <div class="panel-title">Feedstock Characteristics</div>

      <div class="form-group">
        <label>Select Biomass:</label>
        <select id="presetSelect" onchange="applyPreset()">
          <option value="Rice Straw" selected>Rice Straw (High Ash: 22.5%)</option>
          <option value="Pine Wood">Pine Wood (High Lignin: 28.0%)</option>
          <option value="Corn Cobs">Corn Cobs (High Cellulose: 42.0%)</option>
          <option value="Rice Husk">Rice Husk (High Hemicellulose)</option>
          <option value="Wheat Straw">Wheat Straw</option>
        </select>
      </div>

      <div class="form-group">
        <label>Cellulose (%):</label>
        <div class="slider-row">
          <input type="range" id="cel" min="10" max="60" value="39.2" step="0.1" oninput="updateVal('cel')">
          <span class="slider-val" id="cel-val">39.2%</span>
        </div>
      </div>

      <div class="form-group">
        <label>Hemicellulose (%):</label>
        <div class="slider-row">
          <input type="range" id="hemi" min="5" max="50" value="22.0" step="0.1" oninput="updateVal('hemi')">
          <span class="slider-val" id="hemi-val">22.0%</span>
        </div>
      </div>

      <div class="form-group">
        <label>Lignin (%):</label>
        <div class="slider-row">
          <input type="range" id="lig" min="2" max="50" value="22.0" step="0.1" oninput="updateVal('lig')">
          <span class="slider-val" id="lig-val">22.0%</span>
        </div>
      </div>

      <div class="form-group">
        <label>Extractives (%):</label>
        <div class="slider-row">
          <input type="range" id="ext" min="1" max="30" value="16.8" step="0.1" oninput="updateVal('ext')">
          <span class="slider-val" id="ext-val">16.8%</span>
        </div>
      </div>

      <div class="form-group">
        <label>Mineral Ash Content (%):</label>
        <div class="slider-row">
          <input type="range" id="ash" min="0.5" max="30" value="22.55" step="0.05" oninput="updateVal('ash')">
          <span class="slider-val" id="ash-val">22.55%</span>
        </div>
      </div>

      <button class="btn" id="runBtn" onclick="runOptimization()">Calculate Optimal Recipe</button>
    </div>

    <!-- OUTPUT COLUMN -->
    <div>
      <!-- KPI Row -->
      <div class="kpi-row">
        <div class="kpi-card">
          <div class="kpi-label">Recommended Temp</div>
          <div class="kpi-value" id="statTemp">—</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Residence Time</div>
          <div class="kpi-value" id="statTime">—</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Max Expected Yield</div>
          <div class="kpi-value" id="statYield">—</div>
        </div>
      </div>

      <!-- Recommendation Table -->
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th style="width: 80px;">Rank</th>
              <th>Furnace Temp (T)</th>
              <th>Residence Time (RT)</th>
              <th>Heating Rate (HR)</th>
              <th>Expected Yield</th>
              <th>Process Score</th>
            </tr>
          </thead>
          <tbody id="recTableBody">
            <tr>
              <td colspan="6" style="text-align: center; color: #9ca3af; padding: 24px;">
                Select a feedstock on the left and click <strong>"Calculate Optimal Recipe"</strong> to run optimization.
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Temperature Trajectory Chart -->
      <div class="chart-box">
        <div style="font-size: 13px; font-weight: 700; text-transform: uppercase; margin-bottom: 8px;">
          Thermal Devolatilization Profile (Yield vs Reactor Temperature)
        </div>
        <div class="chart-container">
          <canvas id="yieldChart"></canvas>
        </div>
      </div>
    </div>
  </div>

  <script>
    const presets = """ + json.dumps(PRESET_FEEDSTOCKS) + r""";

    function updateVal(id) {
      document.getElementById(id + '-val').innerText = document.getElementById(id).value + '%';
    }

    function applyPreset() {
      const p = presets[document.getElementById('presetSelect').value];
      if (!p) return;
      document.getElementById('cel').value = p.cel; updateVal('cel');
      document.getElementById('hemi').value = p.hemi; updateVal('hemi');
      document.getElementById('lig').value = p.lig; updateVal('lig');
      document.getElementById('ext').value = p.ext; updateVal('ext');
      document.getElementById('ash').value = p.ash; updateVal('ash');
      // Values updated on screen; operator must click "Calculate Optimal Recipe" to run!
    }

    let chartInstance = null;
    function initChart(temps, pinnYields) {
      const ctx = document.getElementById('yieldChart').getContext('2d');
      if (chartInstance) chartInstance.destroy();
      chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels: temps.map(t => Math.round(t) + '°C'),
          datasets: [
            {
              label: 'Predicted Biochar Yield (wt%)',
              data: pinnYields,
              borderColor: '#111827',
              backgroundColor: 'rgba(17, 24, 39, 0.05)',
              borderWidth: 2,
              pointBackgroundColor: '#111827',
              pointRadius: 4,
              fill: true,
              tension: 0.2
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              title: { display: true, text: 'Biochar Yield (wt%)', color: '#6b7280' },
              grid: { color: '#f3f4f6' },
              ticks: { color: '#374151' }
            },
            x: {
              title: { display: true, text: 'Reactor Temperature (°C)', color: '#6b7280' },
              grid: { color: '#f3f4f6' },
              ticks: { color: '#374151' }
            }
          },
          plugins: {
            legend: { display: false }
          }
        }
      });
    }

    async function runOptimization() {
      const btn = document.getElementById('runBtn');
      const statusBadge = document.getElementById('systemStatus');
      btn.innerText = 'Calculating...';
      btn.disabled = true;
      statusBadge.innerText = 'System: Simulating 5,000 Recipes...';

      const t0 = performance.now();

      const payload = {
        cel: parseFloat(document.getElementById('cel').value),
        hemi: parseFloat(document.getElementById('hemi').value),
        lig: parseFloat(document.getElementById('lig').value),
        ext: parseFloat(document.getElementById('ext').value),
        ash: parseFloat(document.getElementById('ash').value) / 100.0
      };

      try {
        const res = await fetch('/api/optimize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        
        const elapsedSec = ((performance.now() - t0) / 1000).toFixed(2);
        statusBadge.innerText = `Completed in ${elapsedSec}s`;

        // Update KPIs to match Rank #1 Best in the table exactly
        const top = data.recommendations[0];
        document.getElementById('statTemp').innerText = top.Temperature_C.toFixed(1) + ' °C';
        document.getElementById('statTime').innerText = top.Residence_Time_min.toFixed(1) + ' min';
        document.getElementById('statYield').innerText = top.Predicted_Yield_pct.toFixed(1) + ' wt%';

        // Update Table
        let html = '';
        data.recommendations.forEach((r, idx) => {
          const isBest = idx === 0;
          html += `<tr ${isBest ? 'class="best-row"' : ''}>
            <td><span class="badge">${isBest ? '#1 BEST' : '#' + (idx+1)}</span></td>
            <td><strong>${r.Temperature_C.toFixed(1)} °C</strong></td>
            <td>${r.Residence_Time_min.toFixed(1)} min</td>
            <td>${r.Heating_Rate_C_per_min.toFixed(1)} °C/min</td>
            <td><strong>${r.Predicted_Yield_pct.toFixed(1)} wt%</strong></td>
            <td>${r.Acquisition_Score.toFixed(1)} / 100</td>
          </tr>`;
        });
        document.getElementById('recTableBody').innerHTML = html;

        // Update Curve
        initChart(data.curve_temps, data.curve_pinn);

      } catch (err) {
        alert('Calculation error: ' + err);
      } finally {
        btn.innerText = 'Calculate Optimal Recipe';
        btn.disabled = false;
      }
    }

    window.onload = () => {
      // Clean blank state: operator initiates calculation by clicking the button
    };
  </script>
</body>
</html>
"""


class SimpleRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/optimize":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode("utf-8"))

            cel = float(params.get("cel", 39.2))
            hemi = float(params.get("hemi", 22.0))
            lig = float(params.get("lig", 22.0))
            ext = float(params.get("ext", 16.8))
            ash = float(params.get("ash", 0.2255))

            raw_bio = np.array([hemi, cel, lig, ext], dtype=np.float32)
            raw_bio_norm = raw_bio / np.sum(raw_bio)
            f_feat = clr_transform(raw_bio_norm.reshape(1, -1))

            recs = recommend_next_experiments(
                ensemble_models=models,
                feedstock_features=f_feat[0],
                process_scaler=scalers["process"],
                config=config,
                raw_biopolymers=raw_bio_norm,
                ash_weight=ash,
            )

            # Generate response curve across 300 to 600 C at optimal RT & HR
            opt_rt = recs[0]["Residence_Time_min"]
            opt_hr = recs[0]["Heating_Rate_C_per_min"]
            temps = np.linspace(300, 600, 13)

            curve_pinn = []

            for t_val in temps:
                cand_raw = np.array([[t_val, opt_rt, opt_hr]], dtype=np.float32)
                cand_scaled = scalers["process"].transform(cand_raw)

                f_t = torch.tensor(f_feat, dtype=torch.float32, device=device)
                p_s_t = torch.tensor(cand_scaled, dtype=torch.float32, device=device)
                p_raw_t = torch.tensor(cand_raw, dtype=torch.float32, device=device)
                bio_t = torch.tensor(raw_bio_norm.reshape(1, 4), dtype=torch.float32, device=device)
                ash_t = torch.tensor([[ash]], dtype=torch.float32, device=device)

                means = []
                for m in models:
                    m.eval()
                    with torch.no_grad():
                        out = m(
                            feedstock_x=f_t,
                            process_x=p_s_t,
                            raw_process=p_raw_t,
                            raw_biopolymers=bio_t,
                            ash_weight=ash_t,
                        )
                        means.append(out["yield_mean"].cpu().item())
                
                # Hybrid prediction (NN + Kinetics)
                with torch.no_grad():
                    kin_y, _ = ode_solver(
                        biopolymer_weights=bio_t,
                        T_peak_C=p_raw_t[:, :1],
                        residence_time_min=p_raw_t[:, 1:2],
                        heating_rate_C_per_min=p_raw_t[:, 2:3],
                        ash_weight=ash_t,
                    )
                hybrid_y = 0.5 * np.mean(means) + 0.5 * kin_y.cpu().item()
                curve_pinn.append(float(hybrid_y))

            response_data = {
                "recommendations": recs,
                "curve_temps": list(temps),
                "curve_pinn": curve_pinn,
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def start_server(port=None):
    if port is None:
        port = int(os.environ.get("PORT", 8501))
    handler = SimpleRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"\n=======================================================")
        print(f"  Biochar App Live at: http://0.0.0.0:{port}")
        print(f"=======================================================\n")
        httpd.serve_forever()


if __name__ == "__main__":
    start_server()

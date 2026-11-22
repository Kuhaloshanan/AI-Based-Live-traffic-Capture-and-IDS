\# AI Driven Cyber Threat Intelligence System



\*\*AI Driven Cyber Threat Intelligence System with Real-Time Network Intrusion Detection and Threat Scoring\*\*



\## Project Overview



An academic cybersecurity prototype combining machine learning, real-time network monitoring, intrusion detection, threat scoring and security alert management.



\## Main Features



\- Real-time network packet capture using Scapy.

\- Bidirectional network flow processing.

\- Isolation Forest anomaly detection.

\- Live Random Forest packet-window classification.

\- Unified detection decision engine.

\- Threat scoring and alert management.

\- FastAPI prediction backend.

\- Streamlit SOC dashboard.



\## System Architecture



\### Live Detection Pipeline



1\. Scapy captures network packets.

2\. FlowProcessor generates bidirectional network flows.

3\. Isolation Forest detects unusual network flows.

4\. Live Random Forest classifies completed packet windows.

5\. Unified Decision Engine combines available detection evidence.

6\. Threat Scoring assigns severity.

7\. Alert Manager records flagged decisions.

8\. Streamlit displays monitoring results.



\### Supervised Classification Pipeline



1\. CICIoT2023 test samples provide 39 network features.

2\. FastAPI receives feature data.

3\. Improved Random Forest predicts the traffic category.

4\. Prediction Engine calculates the threat score and severity.

5\. Dashboard displays the classification result.



\## Project Structure



\- `dashboard/` – Streamlit dashboard

\- `dataset/` – Processed datasets and runtime baseline

\- `logs/` – Runtime logs and telemetry

\- `models/` – Trained machine learning models

\- `notebook/` – Machine learning development notebooks

\- `scripts/` – Data collection, preparation, training and analysis

\- `src/` – Main application modules

\- `tests/` – Automated tests



\## Technologies



Python 3.11, Scapy, Pandas, NumPy, Scikit-learn, Random Forest, Isolation Forest, FastAPI, Uvicorn, Streamlit, Pytest, Git and Git LFS.



\## Installation



Prerequisites: Python 3.11, Git, Git LFS and Npcap for live packet capture.



Clone the repository:



```powershell

git lfs install

git clone https://github.com/Kuhaloshanan/AI-Based-Live-traffic-Capture-and-IDS.git

cd AI-Based-Live-traffic-Capture-and-IDS

git lfs pull

```



Create a virtual environment and install dependencies:



```powershell

py -3.11 -m venv venv

.\\venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip

pip install -r requirements.txt

```



\## Running the System



Run each component in a separate PowerShell terminal.



\*\*1. FastAPI backend\*\*



```powershell

python -m uvicorn src.api:app --host 127.0.0.1 --port 8000

```



API documentation: http://127.0.0.1:8000/docs



\*\*2. Streamlit dashboard\*\*



```powershell

streamlit run dashboard/app.py

```



\*\*3. Live network monitoring\*\*



```powershell

python -m src.live\_flow\_monitor

```



Live packet capture requires a supported network interface and appropriate Npcap permissions. Only monitor networks you are authorized to monitor.



\## Model Evaluation



Improved Random Forest results on the held-out CICIoT2023 test set:



| Metric | Result |

|---|---:|

| Accuracy | 76.72% |

| Macro Precision | 62.88% |

| Macro Recall | 58.93% |

| Macro F1-score | 59.03% |

| Weighted F1-score | 76.11% |



Evaluation used one cleaned source file, Merged01.csv, with a stratified 80/20 train-test split.



\## Threat Scoring



| Score | Threat Level |

|---|---|

| 0–24 | LOW |

| 25–49 | MEDIUM |

| 50–74 | HIGH |

| 75–100 | CRITICAL |



These are application-defined severity categories, not calibrated attack probabilities.



\## Automated Testing



Run the regression suite:



```powershell

python -m pytest -q

```



The completed regression verification recorded 38 passing tests.



\## Dataset Information



The repository includes processed CICIoT2023 training and testing data and the runtime normal-traffic baseline.



The original raw CICIoT2023 dataset is excluded because of its size. Obtain it separately and place it inside `dataset/MERGED\_CSV/`.



\## Limitations



\- Live anomalies are not independently confirmed attacks.

\- Independently labelled real-world live attack validation remains future work.

\- Threat scoring is heuristic and application-defined.

\- Random Forest confidence values are not calibrated probabilities.

\- The live Random Forest feature mapping is experimental.

\- Live Random Forest predictions require complete packet windows.

\- The live Random Forest window processor currently handles IPv4 traffic.

\- Live packet capture requires access to a supported network interface.



\## Security



Do not publish credentials, private network logs or sensitive network information. Use this system only on networks you are authorized to monitor.



\## Academic Project



Developed as a final-year academic cybersecurity prototype combining machine learning, network intrusion detection, threat scoring and security monitoring.

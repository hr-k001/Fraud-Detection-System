import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Database,
  Pause,
  Play,
  Radar,
  RefreshCcw,
  ShieldAlert,
  ShieldCheck,
  Zap,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API_BASE = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? "http://127.0.0.1:8000" : "");

const MERCHANTS = ["Amazon", "Walmart", "Target", "Apple", "Uber", "Netflix", "Flipkart", "Steam"];
const PRODUCTS = ["W", "C", "H", "S", "R"];
const CARDS = ["visa", "mastercard", "discover", "american express"];
const EMAILS = ["gmail.com", "outlook.com", "yahoo.com", "protonmail.com", "unknown.com"];

function buildTransaction(mode = "mixed") {
  const risky = mode === "risky" || (mode === "mixed" && Math.random() > 0.58);
  const amount = risky ? randomNumber(900, 12500) : randomNumber(12, 450);
  return {
    TransactionAmt: Number(amount.toFixed(2)),
    ProductCD: pick(risky ? ["C", "H", "W"] : PRODUCTS),
    card1: randomInt(risky ? 1800 : 900, risky ? 4200 : 1700),
    card2: randomInt(240, 620),
    card3: 150,
    card4: pick(CARDS),
    card5: randomInt(100, 260),
    card6: pick(risky ? ["credit", "credit", "debit"] : ["debit", "credit"]),
    addr1: risky ? randomInt(400, 999) : randomInt(120, 360),
    addr2: risky ? randomInt(50, 999) : 87,
    dist1: risky ? randomInt(80, 650) : randomInt(1, 35),
    P_emaildomain: pick(risky ? ["unknown.com", "protonmail.com", "fraud.com"] : EMAILS),
    R_emaildomain: pick(risky ? ["unknown.com", "protonmail.com", "fraud.com"] : EMAILS),
    C1: risky ? randomInt(6, 24) : randomInt(0, 3),
    C2: risky ? randomInt(5, 18) : randomInt(0, 3),
    C3: risky ? randomInt(2, 11) : randomInt(0, 1),
    C4: risky ? randomInt(2, 10) : 0,
    C5: risky ? randomInt(1, 7) : 0,
    Merchant: pick(MERCHANTS),
  };
}

const initialManual = {
  TransactionAmt: 250,
  ProductCD: "W",
  card1: 1500,
  card2: 400,
  card3: 150,
  card4: "visa",
  card5: 180,
  card6: "credit",
  addr1: 300,
  addr2: 87,
  dist1: 15,
  P_emaildomain: "gmail.com",
  R_emaildomain: "gmail.com",
  C1: 2,
  C2: 2,
  C3: 0,
  C4: 0,
  C5: 0,
};

export default function App() {
  const [health, setHealth] = useState(null);
  const [storage, setStorage] = useState(null);
  const [transactions, setTransactions] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [latest, setLatest] = useState(null);
  const [manual, setManual] = useState(initialManual);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(2200);
  const [mode, setMode] = useState("mixed");
  const [error, setError] = useState("");
  const timer = useRef(null);

  const stats = useMemo(() => {
    const checked = transactions.length;
    const fraud = transactions.filter((item) => item.prediction === 1).length;
    const highRisk = transactions.filter((item) => ["High", "Critical"].includes(item.risk_level)).length;
    const avgProb = checked
      ? transactions.reduce((sum, item) => sum + Number(item.fraud_probability || 0), 0) / checked
      : 0;
    return { checked, fraud, highRisk, avgProb };
  }, [transactions]);

  const probabilityTrend = useMemo(
    () =>
      transactions
        .slice(0, 16)
        .reverse()
        .map((item, index) => ({
          name: `T${index + 1}`,
          probability: Math.round(Number(item.fraud_probability || 0) * 100),
        })),
    [transactions]
  );

  const riskBars = useMemo(() => {
    const counts = { Low: 0, Medium: 0, High: 0, Critical: 0 };
    transactions.forEach((item) => {
      counts[item.risk_level] = (counts[item.risk_level] || 0) + 1;
    });
    return Object.entries(counts).map(([name, value]) => ({ name, value }));
  }, [transactions]);

  async function request(path, options) {
    const response = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `Request failed: ${response.status}`);
    }
    return response.json();
  }

  async function refresh() {
    try {
      const [healthData, storageData, txData, alertData] = await Promise.all([
        request("/health"),
        request("/storage/status"),
        request("/transactions?limit=100"),
        request("/alerts?limit=50"),
      ]);
      setHealth(healthData);
      setStorage(storageData);
      setTransactions(txData.transactions || []);
      setAlerts(alertData.alerts || []);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  async function checkTransaction(transaction) {
    const result = await request("/predict", {
      method: "POST",
      body: JSON.stringify(transaction),
    });
    setLatest({ request: transaction, response: result, created_at: new Date().toISOString() });
    await refresh();
  }

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    if (!running) {
      clearInterval(timer.current);
      return;
    }
    timer.current = setInterval(() => {
      checkTransaction(buildTransaction(mode)).catch((err) => setError(err.message));
    }, speed);
    return () => clearInterval(timer.current);
  }, [running, speed, mode]);

  function submitManual(event) {
    event.preventDefault();
    checkTransaction(normalizeManual(manual)).catch((err) => setError(err.message));
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Real-time banking transaction checks</p>
          <h1>Fraud Detection Console</h1>
        </div>
        <div className="status-strip">
          <StatusPill
            icon={health?.model_loaded ? ShieldCheck : ShieldAlert}
            label={health?.model_loaded ? "Model online" : "Model offline"}
            tone={health?.model_loaded ? "good" : "bad"}
          />
          <StatusPill
            icon={storage?.connected ? Database : AlertTriangle}
            label={storage?.connected ? "Azure SQL connected" : "Storage fallback"}
            tone={storage?.connected ? "good" : "warn"}
          />
          <button className="icon-button" onClick={refresh} title="Refresh dashboard">
            <RefreshCcw size={18} />
          </button>
        </div>
      </header>

      {error && <div className="banner bad">{error}</div>}
      {storage && !storage.connected && (
        <div className="banner warn">
          Azure SQL is not reachable yet. Predictions are still running and mirrored in memory.
        </div>
      )}

      <section className="control-band">
        <div className="stream-control">
          <button className={`primary ${running ? "danger" : ""}`} onClick={() => setRunning((value) => !value)}>
            {running ? <Pause size={18} /> : <Play size={18} />}
            {running ? "Pause stream" : "Start stream"}
          </button>
          <Segmented value={mode} onChange={setMode} options={["mixed", "normal", "risky"]} />
          <label className="range">
            <Zap size={16} />
            <input
              type="range"
              min="900"
              max="4200"
              step="100"
              value={speed}
              onChange={(event) => setSpeed(Number(event.target.value))}
            />
            <span>{(speed / 1000).toFixed(1)}s</span>
          </label>
        </div>
        <div className="storage-note">
          <span>{storage?.provider || "Checking storage"}</span>
          <strong>{storage?.database || "transaction_bank"}</strong>
        </div>
      </section>

      <section className="metric-grid">
        <Metric icon={Activity} label="Checked" value={stats.checked} />
        <Metric icon={ShieldAlert} label="Fraud hits" value={stats.fraud} tone="bad" />
        <Metric icon={AlertTriangle} label="High risk" value={stats.highRisk} tone="warn" />
        <Metric icon={Radar} label="Avg probability" value={`${Math.round(stats.avgProb * 100)}%`} />
      </section>

      <section className="main-grid">
        <div className="panel live-panel">
          <div className="panel-header">
            <h2>Live Transaction Result</h2>
            <span className={`risk-dot ${latest?.response?.risk_level || "Low"}`}>
              {latest?.response?.risk_level || "Waiting"}
            </span>
          </div>
          {latest ? (
            <div className="result-layout">
              <div className="score-ring">
                <span>{Math.round(latest.response.fraud_probability * 100)}%</span>
                <small>fraud probability</small>
              </div>
              <div className="feature-list">
                {latest.response.top_features.map((feature) => (
                  <div key={feature.feature} className="feature-row">
                    <span>{feature.feature}</span>
                    <div>
                      <i style={{ width: `${Math.min(feature.importance * 420, 100)}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="empty-state">Start the stream or check a transaction manually.</div>
          )}
        </div>

        <div className="panel chart-panel">
          <div className="panel-header">
            <h2>Probability Trend</h2>
          </div>
          <ResponsiveContainer width="100%" height={230}>
            <AreaChart data={probabilityTrend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Area type="monotone" dataKey="probability" stroke="#2563eb" fill="#bfdbfe" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="panel chart-panel">
          <div className="panel-header">
            <h2>Risk Distribution</h2>
          </div>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={riskBars}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {riskBars.map((entry) => (
                  <Cell key={entry.name} fill={riskColor(entry.name)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="bottom-grid">
        <form className="panel form-panel" onSubmit={submitManual}>
          <div className="panel-header">
            <h2>Manual Check</h2>
            <button type="button" className="ghost" onClick={() => setManual(buildTransaction("risky"))}>
              Fill risky
            </button>
          </div>
          <div className="form-grid">
            {Object.entries(manual).map(([key, value]) => (
              <label key={key}>
                <span>{key}</span>
                <input value={value} onChange={(event) => setManual({ ...manual, [key]: event.target.value })} />
              </label>
            ))}
          </div>
          <button className="primary" type="submit">
            <ShieldCheck size={18} />
            Check transaction
          </button>
        </form>

        <div className="panel table-panel">
          <div className="panel-header">
            <h2>Fraud Alerts</h2>
            <span>{alerts.length}</span>
          </div>
          <div className="alert-list">
            {alerts.slice(0, 10).map((item, index) => (
              <div className="alert-item" key={`${item.id || index}-${item.created_at}`}>
                <AlertTriangle size={18} />
                <div>
                  <strong>{item.risk_level} risk</strong>
                  <span>
                    ${Number(item.transaction_amount || 0).toFixed(2)} · {Math.round(item.fraud_probability * 100)}%
                  </span>
                </div>
              </div>
            ))}
            {!alerts.length && <div className="empty-state compact">No active alerts.</div>}
          </div>
        </div>
      </section>

      <section className="panel table-panel full">
        <div className="panel-header">
          <h2>Recent Transaction Checks</h2>
          <span>{transactions.length} rows</span>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Amount</th>
                <th>Product</th>
                <th>Prediction</th>
                <th>Probability</th>
                <th>Risk</th>
              </tr>
            </thead>
            <tbody>
              {transactions.slice(0, 18).map((item, index) => (
                <tr key={`${item.id || index}-${item.created_at}`}>
                  <td>{formatTime(item.created_at)}</td>
                  <td>${Number(item.transaction_amount || 0).toFixed(2)}</td>
                  <td>{item.product_code}</td>
                  <td>{item.prediction === 1 ? "Fraud" : "Legit"}</td>
                  <td>{Math.round(Number(item.fraud_probability || 0) * 100)}%</td>
                  <td>
                    <span className={`risk-chip ${item.risk_level}`}>{item.risk_level}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

function Metric({ icon: Icon, label, value, tone = "" }) {
  return (
    <div className={`metric ${tone}`}>
      <Icon size={22} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StatusPill({ icon: Icon, label, tone }) {
  return (
    <span className={`status-pill ${tone}`}>
      <Icon size={16} />
      {label}
    </span>
  );
}

function Segmented({ value, onChange, options }) {
  return (
    <div className="segmented">
      {options.map((option) => (
        <button
          type="button"
          key={option}
          className={option === value ? "active" : ""}
          onClick={() => onChange(option)}
        >
          {option}
        </button>
      ))}
    </div>
  );
}

function normalizeManual(data) {
  return Object.fromEntries(
    Object.entries(data).map(([key, value]) => {
      const numeric = Number(value);
      return [key, Number.isNaN(numeric) || value === "" ? value : numeric];
    })
  );
}

function pick(items) {
  return items[Math.floor(Math.random() * items.length)];
}

function randomNumber(min, max) {
  return Math.random() * (max - min) + min;
}

function randomInt(min, max) {
  return Math.floor(randomNumber(min, max + 1));
}

function formatTime(value) {
  if (!value) return "--";
  return new Date(value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function riskColor(risk) {
  return {
    Low: "#16a34a",
    Medium: "#d97706",
    High: "#dc2626",
    Critical: "#7f1d1d",
  }[risk] || "#64748b";
}

import React, { useState, useEffect } from "react";
import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
} from "recharts";
import {
  BarChart3, TrendingUp, Brain, Activity,
  ThumbsUp, ThumbsDown, Minus, AlertCircle,
  ExternalLink, RefreshCw, Layers,
} from "lucide-react";
import clsx from "clsx";
import { getInfo } from "../api";
const COLORS = {
  Positive: "#10b981",
  Negative: "#ef4444",
  Neutral:  "#f59e0b",
};

const SENTIMENT_ICONS = {
  Positive: ThumbsUp,
  Negative: ThumbsDown,
  Neutral:  Minus,
};

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center">
      <div className="w-16 h-16 bg-slate-100 rounded-2xl flex items-center
                      justify-center mb-4">
        <BarChart3 className="w-8 h-8 text-slate-300" />
      </div>
      <h3 className="text-base font-semibold text-slate-700 mb-1">
        No data yet
      </h3>
      <p className="text-sm text-slate-400 max-w-xs">
        Upload a CSV in the Bulk Upload tab to see analytics here.
        Results will automatically appear on this dashboard.
      </p>
    </div>
  );
}

function StatCard({ label, value, sub, color, icon: Icon }) {
  return (
    <div className="card flex items-center gap-4">
      <div className={clsx(
        "w-11 h-11 rounded-xl flex items-center justify-center flex-shrink-0",
        color.bg
      )}>
        <Icon className={clsx("w-5 h-5", color.text)} />
      </div>
      <div>
        <p className={clsx("text-2xl font-bold tabular-nums", color.text)}>{value}</p>
        <p className="text-xs text-slate-500 mt-0.5">{label}</p>
        {sub && <p className="text-xs text-slate-400">{sub}</p>}
      </div>
    </div>
  );
}

function PieLabel({ cx, cy, midAngle, innerRadius, outerRadius, percent }) {
  if (percent < 0.05) return null;
  const RADIAN = Math.PI / 180;
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);
  return (
    <text x={x} y={y} fill="white" textAnchor="middle"
          dominantBaseline="central" fontSize={12} fontWeight={600}>
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  );
}

function TopReviews({ results, sentiment }) {
  const cfg   = { Positive: "text-emerald-600 bg-emerald-50 border-emerald-200",
                  Negative: "text-red-600 bg-red-50 border-red-200",
                  Neutral:  "text-amber-600 bg-amber-50 border-amber-200" };
  const top   = results
    .filter(r => r.sentiment === sentiment)
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 3);

  if (top.length === 0) return (
    <p className="text-xs text-slate-400 italic py-2">No {sentiment} reviews found.</p>
  );

  return (
    <div className="space-y-2">
      {top.map((r, i) => (
        <div key={i} className={clsx(
          "p-3 rounded-lg border text-xs", cfg[sentiment]
        )}>
          <p className="line-clamp-2 leading-relaxed text-slate-700">
            "{r.review}"
          </p>
          <p className="mt-1 font-semibold">
            {Math.round(r.confidence * 100)}% confident
          </p>
        </div>
      ))}
    </div>
  );
}

function ModelInfoPanel() {
  const [info,    setInfo]    = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchInfo = async () => {
    setLoading(true);
    try {
      const data = await getInfo();
      setInfo(data);
    } catch {
      setInfo(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchInfo(); }, []);

  const rows = info ? [
    { label: "Model Name",    value: info.model?.model_name    || "—" },
    { label: "Version",       value: info.model?.model_version || "—" },
    { label: "Stage",         value: info.model?.model_stage   || "—" },
    { label: "API Version",   value: info.api?.version         || "—" },
    { label: "Uptime",        value: info.api?.uptime_seconds
        ? `${Math.floor(info.api.uptime_seconds / 60)}m`
        : "—" },
    { label: "Python",        value: info.system?.python_version || "—" },
    { label: "CPU %",         value: info.system?.cpu_percent != null
        ? `${info.system.cpu_percent}%` : "—" },
    { label: "Memory %",      value: info.system?.memory_percent != null
        ? `${info.system.memory_percent}%` : "—" },
  ] : [];

  return (
    <div className="card space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-indigo-500" />
          <h3 className="text-sm font-semibold text-slate-700">Model & System Info</h3>
        </div>
        <button
          onClick={fetchInfo}
          className="p-1.5 rounded-lg text-slate-400 hover:text-indigo-500
                     hover:bg-indigo-50 transition-all"
          title="Refresh"
        >
          <RefreshCw className={clsx("w-3.5 h-3.5", loading && "animate-spin")} />
        </button>
      </div>

      {loading ? (
        <div className="space-y-2">
          {[1,2,3,4].map(i => (
            <div key={i} className="h-5 bg-slate-100 rounded animate-pulse" />
          ))}
        </div>
      ) : info ? (
        <div className="space-y-2">
          {rows.map(r => (
            <div key={r.label} className="flex justify-between text-xs">
              <span className="text-slate-400">{r.label}</span>
              <span className="font-medium text-slate-700 tabular-nums">{r.value}</span>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-xs text-slate-400 italic">Could not fetch model info.</p>
      )}

      {/* External tool links */}
      <div className="pt-2 border-t border-slate-100 space-y-1.5">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
          MLOps Tools
        </p>
        {[
          { label: "MLflow UI",    url: "http://localhost:5000",  color: "text-blue-600"   },
          { label: "Airflow UI",   url: "http://localhost:8080",  color: "text-green-600"  },
          { label: "Grafana",      url: "http://localhost:3001",  color: "text-orange-600" },
          { label: "Prometheus",   url: "http://localhost:9090",  color: "text-red-600"    },
          { label: "API Docs",     url: "http://localhost:8000/docs", color: "text-indigo-600" },
        ].map(link => (
          <a
            key={link.label}
            href={link.url}
            target="_blank"
            rel="noreferrer"
            className={clsx(
              "flex items-center justify-between text-xs py-1 px-2",
              "rounded-lg hover:bg-slate-50 transition-all", link.color
            )}
          >
            <span className="font-medium">{link.label}</span>
            <ExternalLink className="w-3 h-3 opacity-60" />
          </a>
        ))}
      </div>
    </div>
  );
}

function ConfidenceHistogram({ results }) {
  const buckets = [
    { range: "50–60%", min: 0.50, max: 0.60 },
    { range: "60–70%", min: 0.60, max: 0.70 },
    { range: "70–80%", min: 0.70, max: 0.80 },
    { range: "80–90%", min: 0.80, max: 0.90 },
    { range: "90–100%",min: 0.90, max: 1.01 },
  ];

  const data = buckets.map(b => ({
    range:    b.range,
    count:    results.filter(r => r.confidence >= b.min && r.confidence < b.max).length,
    positive: results.filter(r => r.sentiment === "Positive" && r.confidence >= b.min && r.confidence < b.max).length,
    negative: results.filter(r => r.sentiment === "Negative" && r.confidence >= b.min && r.confidence < b.max).length,
    neutral:  results.filter(r => r.sentiment === "Neutral"  && r.confidence >= b.min && r.confidence < b.max).length,
  }));

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis dataKey="range" tick={{ fontSize: 11, fill: "#94a3b8" }} />
        <YAxis tick={{ fontSize: 11, fill: "#94a3b8" }} />
        <Tooltip
          contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }}
        />
        <Bar dataKey="positive" name="Positive" fill={COLORS.Positive} stackId="a" radius={[0,0,0,0]} />
        <Bar dataKey="neutral"  name="Neutral"  fill={COLORS.Neutral}  stackId="a" />
        <Bar dataKey="negative" name="Negative" fill={COLORS.Negative} stackId="a" radius={[4,4,0,0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export default function Dashboard({ bulkResults }) {
  const [activeTopTab, setActiveTopTab] = useState("Positive");

  // Use bulk results if available
  const results = bulkResults?.results || [];
  const summary = bulkResults?.summary || {};
  const total   = bulkResults?.total   || 0;

  if (results.length === 0) {
    return (
      <div className="max-w-6xl mx-auto">
        <div className="mb-6">
          <h2 className="text-xl font-bold text-slate-900">Analytics Dashboard</h2>
          <p className="text-sm text-slate-500 mt-1">
            Sentiment distribution, confidence analysis, and model insights.
          </p>
        </div>
        <EmptyState />
      </div>
    );
  }

  const pieData = Object.entries(summary).map(([name, value]) => ({ name, value }));
  const radarData = ["Positive", "Negative", "Neutral"].map(s => {
    const subset = results.filter(r => r.sentiment === s);
    const avg    = subset.length
      ? Math.round(subset.reduce((acc, r) => acc + r.confidence, 0) / subset.length * 100)
      : 0;
    return { sentiment: s, confidence: avg };
  });

  const avgConf = results.length
    ? Math.round(results.reduce((a, r) => a + r.confidence, 0) / results.length * 100)
    : 0;
  const inferenceMs = bulkResults?.inference_time_ms || 0;

  return (
    <div className="max-w-6xl mx-auto space-y-6">

      {/* Heading */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-slate-900">Analytics Dashboard</h2>
          <p className="text-sm text-slate-500 mt-1">
            Results from last bulk analysis · {total} reviews
          </p>
        </div>
        <div className="flex items-center gap-1.5 text-xs text-slate-400">
          <Activity className="w-3.5 h-3.5" />
          {inferenceMs}ms total inference
        </div>
      </div>

      {/* ── STAT CARDS ──────────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <StatCard
          label="Total Reviews"
          value={total}
          icon={Layers}
          color={{ bg: "bg-slate-100", text: "text-slate-700" }}
        />
        <StatCard
          label="Positive"
          value={summary.Positive || 0}
          sub={`${Math.round((summary.Positive || 0) / total * 100)}%`}
          icon={ThumbsUp}
          color={{ bg: "bg-emerald-100", text: "text-emerald-600" }}
        />
        <StatCard
          label="Negative"
          value={summary.Negative || 0}
          sub={`${Math.round((summary.Negative || 0) / total * 100)}%`}
          icon={ThumbsDown}
          color={{ bg: "bg-red-100", text: "text-red-600" }}
        />
        <StatCard
          label="Avg Confidence"
          value={`${avgConf}%`}
          sub="across all reviews"
          icon={TrendingUp}
          color={{ bg: "bg-indigo-100", text: "text-indigo-600" }}
        />
      </div>

      {/* ── CHARTS ROW ──────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Pie Chart */}
        <div className="card">
          <h3 className="text-sm font-semibold text-slate-700 mb-4">
            Sentiment Distribution
          </h3>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                outerRadius={80}
                dataKey="value"
                labelLine={false}
                label={PieLabel}
              >
                {pieData.map((entry) => (
                  <Cell key={entry.name} fill={COLORS[entry.name]} />
                ))}
              </Pie>
              <Tooltip
                formatter={(v, n) => [`${v} reviews (${Math.round(v/total*100)}%)`, n]}
                contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }}
              />
              <Legend
                iconType="circle"
                iconSize={8}
                formatter={(v) => <span style={{ fontSize: 12, color: "#64748b" }}>{v}</span>}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Confidence Histogram */}
        <div className="card">
          <h3 className="text-sm font-semibold text-slate-700 mb-4">
            Confidence Distribution
          </h3>
          <ConfidenceHistogram results={results} />
          <p className="text-xs text-slate-400 mt-2 text-center">
            Stacked by sentiment class
          </p>
        </div>

        {/* Radar Chart */}
        <div className="card">
          <h3 className="text-sm font-semibold text-slate-700 mb-4">
            Avg Confidence by Class
          </h3>
          <ResponsiveContainer width="100%" height={220}>
            <RadarChart data={radarData} cx="50%" cy="50%" outerRadius={70}>
              <PolarGrid stroke="#e2e8f0" />
              <PolarAngleAxis
                dataKey="sentiment"
                tick={{ fontSize: 11, fill: "#64748b" }}
              />
              <Radar
                name="Confidence"
                dataKey="confidence"
                stroke="#6366f1"
                fill="#6366f1"
                fillOpacity={0.25}
              />
              <Tooltip
                formatter={(v) => [`${v}%`, "Avg Confidence"]}
                contentStyle={{ fontSize: 12, borderRadius: 8 }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>

      </div>

      {/* ── BOTTOM ROW: Top Reviews + Model Info ────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Top Reviews by Sentiment */}
        <div className="lg:col-span-2 card space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-700">
              Highest Confidence Reviews
            </h3>
            {/* Sentiment tab switcher */}
            <div className="flex gap-1">
              {["Positive", "Negative", "Neutral"].map(s => {
                const Icon = SENTIMENT_ICONS[s];
                return (
                  <button
                    key={s}
                    onClick={() => setActiveTopTab(s)}
                    className={clsx(
                      "flex items-center gap-1 px-2.5 py-1 text-xs font-medium",
                      "rounded-lg border transition-all",
                      activeTopTab === s
                        ? s === "Positive" ? "bg-emerald-600 text-white border-emerald-600"
                          : s === "Negative" ? "bg-red-500 text-white border-red-500"
                          : "bg-amber-500 text-white border-amber-500"
                        : "bg-white text-slate-500 border-slate-200 hover:border-slate-300"
                    )}
                  >
                    <Icon className="w-3 h-3" />
                    {s}
                  </button>
                );
              })}
            </div>
          </div>
          <TopReviews results={results} sentiment={activeTopTab} />
        </div>

        {/* Model Info Panel */}
        <ModelInfoPanel />

      </div>

    </div>
  );
}
